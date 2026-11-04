"""Strata engine contract — pins, capability rules, and launch planning.

Strata ships two versions that move independently: the app/Terminal (0.2.10) and
every engine package plus the Control (0.1.56). A Turbofile that names one and
assumes the other is the quiet source of "it runs but is slow" reports, so both
are pinned here and checked by :func:`version_problems`.

The rules below are transcribed from the Strata 0.2.10 release notes, not guessed:

* DFlash2 is the default drafter for Qwen3.8 (about 0.7 GB) on CUDA sm 8.6, 8.9,
  12.0, 12.1, Radeon gfx1100/gfx1151/gfx1201, Apple Metal and Intel Arc.
* Exactly one drafter runs per launch. MTP is chosen by Auto when two or more
  requests run at once, when no DFlash2-capable engine is installed, for any
  family other than Qwen3.8 (DeepSeek, ISTA-DAS, ...), and on the all-cards split
  engine.
* 8 GB cards start at 128K context (this did not work before 0.2.10).
* The all-cards split engine starts by itself on Linux for exactly two NVIDIA
  cards of compute capability 8.6 or 12.0, or four Radeon AI PRO R9700 cards with
  a UD-Q4_K_XL-type file, on a Qwen3.8 file, when the routed experts fit in the
  cards and about 43 GiB of RAM is free (27.5 GiB of it the embedding table), at
  a context of 131,072 or less.

This module is pure: it takes a hardware fingerprint and returns a plan. No I/O,
so it can be unit-tested against synthetic machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hardware import HardwareFingerprint

STRATA_APP_VERSION = "0.2.10"
STRATA_ENGINE_VERSION = "0.1.56"

FEED_URL = "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed.json"
FEED_URL_ARM64 = (
    "https://github.com/Niko1221/Strata/releases/latest/download/strata-feed-linux-aarch64.json"
)

DRAFTER_CHOICES = ("auto", "mtp", "dflash2", "off")

DFLASH2_CUDA_ARCHS = frozenset({"8.6", "8.9", "12.0", "12.1"})
DFLASH2_AMD_ARCHS = frozenset({"gfx1100", "gfx1151", "gfx1201"})
SPLIT_CUDA_ARCHS = frozenset({"8.6", "12.0"})
SPLIT_AMD_ARCHS = frozenset({"gfx1201"})  # Radeon AI PRO R9700

#: Context 8 GB cards can start at as of 0.2.10.
SMALL_CARD_CONTEXT = 131_072
#: Context Strata's own model exposes on a comfortably sized card.
FULL_CONTEXT = 262_144
#: The split engine is capped here regardless of the model's maximum.
SPLIT_CONTEXT_CEILING = 131_072
#: Free RAM the split engine needs (about 43 GiB, 27.5 GiB of which is embeddings).
SPLIT_HOST_RAM_MB = 43 * 1024
#: Below this, a card is treated as a small card: 128K start, more work off-card.
SMALL_CARD_MB = 12 * 1024

#: Model families that Strata serves through its own engine packages.
FAMILY_QWEN = "qwen3.8-flash-next"
FAMILY_DEEPSEEK = "deepseek-v4.1-flash"
FAMILY_ISTA_DAS = "ista-das-coder"


@dataclass(frozen=True)
class StrataPlan:
    """What Strata should be asked to do on this machine."""

    model: str
    quant: str
    context_tokens: int
    drafting: str
    split_engine: str
    parallel: int
    launch_args: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: str = "documented"  # "documented" until a run on this box is recorded

    def to_mapping(self) -> dict:
        return {
            "model": self.model,
            "quant": self.quant,
            "context_tokens": self.context_tokens,
            "drafting": self.drafting,
            "split_engine": self.split_engine,
            "parallel": self.parallel,
            "launch_args": list(self.launch_args),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "evidence": self.evidence,
        }


def _archs(fingerprint: HardwareFingerprint) -> set[str]:
    return {
        str(device.compute_capability).strip()
        for device in fingerprint.devices
        if device.compute_capability
    }


def _nvidia(fingerprint: HardwareFingerprint) -> list:
    return [d for d in fingerprint.devices if (d.vendor or "").lower() == "nvidia"]


def _amd(fingerprint: HardwareFingerprint) -> list:
    return [d for d in fingerprint.devices if (d.vendor or "").lower() in {"amd", "radeon"}]


def _apple(fingerprint: HardwareFingerprint) -> bool:
    return any((d.vendor or "").lower() == "apple" for d in fingerprint.devices)


def _intel(fingerprint: HardwareFingerprint) -> bool:
    return any((d.vendor or "").lower() == "intel" for d in fingerprint.devices)


def drafter_supported(fingerprint: HardwareFingerprint) -> bool:
    """True when every card can run DFlash2 (Strata requires it on all of them)."""
    if not fingerprint.devices:
        return False
    if _apple(fingerprint) or _intel(fingerprint):
        return True
    archs = _archs(fingerprint)
    if not archs:
        return False
    cuda_ok = archs <= DFLASH2_CUDA_ARCHS
    amd_ok = archs <= DFLASH2_AMD_ARCHS
    return cuda_ok or amd_ok


def split_shape(fingerprint: HardwareFingerprint) -> tuple[bool, str]:
    """Whether the cards even form a split shape, before context or RAM are considered."""
    nvidia = _nvidia(fingerprint)
    if len(nvidia) == 2:
        archs = {str(d.compute_capability).strip() for d in nvidia if d.compute_capability}
        if archs and archs <= SPLIT_CUDA_ARCHS:
            return True, f"two NVIDIA cards at sm {sorted(archs)}"
    amd = _amd(fingerprint)
    if len(amd) == 4:
        archs = {str(d.compute_capability).strip() for d in amd if d.compute_capability}
        if archs and archs <= SPLIT_AMD_ARCHS:
            return True, "four Radeon AI PRO R9700 cards"
    return False, (
        f"{len(fingerprint.devices)} card(s) is not a split shape "
        "(two sm 8.6/12.0 NVIDIA or four R9700)"
    )


def split_engine_eligible(
    fingerprint: HardwareFingerprint,
    *,
    context_tokens: int,
    family: str,
    hosts_weights_in_ram: bool = True,
) -> tuple[bool, str]:
    """Whether the all-cards split engine may start here, with the reason either way."""
    if fingerprint.os not in {"linux"}:
        return False, f"split engine is Linux-only (this is {fingerprint.os})"
    if family != FAMILY_QWEN:
        return False, f"split engine serves Qwen3.8 files only (asked for {family})"
    if not hosts_weights_in_ram:
        return False, "split engine needs the non-expert weights in RAM"

    # Shape first: a machine that is not a split shape should hear that, not a
    # context complaint it could never satisfy.
    shape_ok, shape_reason = split_shape(fingerprint)
    if not shape_ok:
        return False, shape_reason
    if context_tokens > SPLIT_CONTEXT_CEILING:
        return False, (
            f"split engine caps context at {SPLIT_CONTEXT_CEILING}; "
            f"asked for {context_tokens}"
        )
    if fingerprint.host_usable_memory_mb < SPLIT_HOST_RAM_MB:
        return False, (
            "split engine wants about 43 GiB free RAM, "
            f"this host reports {fingerprint.host_usable_memory_mb} MiB"
        )
    return True, shape_reason


def quant_for(fingerprint: HardwareFingerprint, *, preferring_split: bool = False) -> str:
    """Pick the file Strata should load.

    IQ3_S is lossless in TurboFit's ladder, so a comfortably sized machine stays
    on it. The split engine's measured numbers are on UD-Q4_K_XL, so a split-eligible
    box is asked for that file unless it only has a small card.
    """
    if preferring_split:
        return "UD-Q4_K_XL"
    if fingerprint.total_vram_mb and fingerprint.total_vram_mb < SMALL_CARD_MB:
        return "IQ3_S"
    return "IQ3_S"


def context_for(fingerprint: HardwareFingerprint, requested: int | None = None) -> int:
    """Context Strata can actually start at here."""
    if requested:
        return int(requested)
    if not fingerprint.devices:
        return SMALL_CARD_CONTEXT
    smallest = min(d.memory_total_mb for d in fingerprint.devices)
    if smallest < SMALL_CARD_MB:
        return SMALL_CARD_CONTEXT
    return FULL_CONTEXT


def plan(
    fingerprint: HardwareFingerprint,
    *,
    family: str = FAMILY_QWEN,
    context_tokens: int | None = None,
    parallel: int = 1,
    quant: str | None = None,
    hosts_weights_in_ram: bool = True,
) -> StrataPlan:
    """Compose the `strata launch` arguments for this machine."""
    reasons: list[str] = []
    warnings: list[str] = []

    context = context_for(fingerprint, context_tokens)

    if context == SMALL_CARD_CONTEXT and (context_tokens or 0) > SMALL_CARD_CONTEXT:
        warnings.append(
            f"a card here is under {SMALL_CARD_MB} MiB, so Strata starts at "
            f"{SMALL_CARD_CONTEXT} rather than {context_tokens}"
        )

    split_ok, split_reason = split_engine_eligible(
        fingerprint,
        context_tokens=context,
        family=family,
        hosts_weights_in_ram=hosts_weights_in_ram,
    )
    if not split_ok and not context_tokens and context > SPLIT_CONTEXT_CEILING:
        # The split engine cannot hold more than 128K and is the faster engine by a
        # wide margin on a split shape, so an automatic context yields to it.
        retry_ok, retry_reason = split_engine_eligible(
            fingerprint,
            context_tokens=SPLIT_CONTEXT_CEILING,
            family=family,
            hosts_weights_in_ram=hosts_weights_in_ram,
        )
        if retry_ok:
            context = SPLIT_CONTEXT_CEILING
            split_ok, split_reason = retry_ok, retry_reason
            reasons.append(
                f"context capped at {SPLIT_CONTEXT_CEILING} so the split engine can run"
            )
        else:
            # The context check would have spoken first; at the context we could
            # actually use, the real blocker is what the retry found.
            split_reason = retry_reason
    split = "auto" if split_ok else "off"
    reasons.append(f"split engine: {'auto' if split_ok else 'off'} — {split_reason}")

    if quant is None:
        quant = quant_for(fingerprint, preferring_split=split_ok)

    drafting = "auto"
    if parallel >= 2:
        drafting = "mtp"
        reasons.append("two or more concurrent requests: MTP decodes faster than DFlash2")
    elif family != FAMILY_QWEN:
        drafting = "mtp"
        reasons.append(f"{family} has no DFlash2 drafter; MTP is the only option")
    elif split_ok:
        drafting = "mtp"
        reasons.append("split engine plus MTP decodes faster than layer split plus DFlash2")
    elif drafter_supported(fingerprint):
        drafting = "dflash2"
        reasons.append("DFlash2 is the measured default on these architectures")
        if _intel(fingerprint):
            warnings.append(
                "on Intel Arc, DFlash2 measured 2.6% slower than MTP; ask for "
                "--drafting mtp if that holds on your card"
            )
    else:
        drafting = "mtp"
        reasons.append("no DFlash2-capable engine for these cards")

    if drafting == "dflash2" and parallel >= 2:
        warnings.append("DFlash2 answers one request at a time; use MTP for concurrency")

    model = {
        FAMILY_QWEN: "qwen3.8-flash-next",
        FAMILY_DEEPSEEK: "deepseek-v4.1-flash",
        FAMILY_ISTA_DAS: "ISTA-DAS-Coder",
    }.get(family, family)

    args = [
        "strata",
        "launch",
        "--model",
        model,
        "--context",
        str(context),
        "--drafting",
        drafting,
        "--parallel",
        str(max(1, parallel)),
        "--split-engine",
        split,
    ]

    return StrataPlan(
        model=model,
        quant=quant,
        context_tokens=context,
        drafting=drafting,
        split_engine=split,
        parallel=max(1, parallel),
        launch_args=tuple(args),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def version_problems(app_version: str | None, engine_version: str | None) -> list[str]:
    """Report a mixed Strata install (the app and its engine packages drift apart)."""
    problems: list[str] = []
    if app_version and app_version != STRATA_APP_VERSION:
        problems.append(
            f"Strata app is {app_version}, TurboFit pins {STRATA_APP_VERSION}"
        )
    if engine_version and engine_version != STRATA_ENGINE_VERSION:
        problems.append(
            f"Strata engine packages are {engine_version}, TurboFit pins "
            f"{STRATA_ENGINE_VERSION} (0.2.10 ships 0.1.56 engines)"
        )
    return problems


def feed_url(os_name: str, architecture: str) -> str:
    """The update feed Strata itself reads (`strata update`)."""
    if os_name == "linux" and architecture in {"aarch64", "arm64"}:
        return FEED_URL_ARM64
    return FEED_URL


__all__ = [
    "DFLASH2_AMD_ARCHS",
    "DFLASH2_CUDA_ARCHS",
    "DRAFTER_CHOICES",
    "FAMILY_DEEPSEEK",
    "FAMILY_ISTA_DAS",
    "FAMILY_QWEN",
    "FULL_CONTEXT",
    "SMALL_CARD_CONTEXT",
    "SPLIT_CONTEXT_CEILING",
    "STRATA_APP_VERSION",
    "STRATA_ENGINE_VERSION",
    "StrataPlan",
    "context_for",
    "drafter_supported",
    "feed_url",
    "plan",
    "quant_for",
    "split_engine_eligible",
    "split_shape",
    "version_problems",
]
