"""Canonical TurboFit Check / Auto lineup.

A 9B must never be recommended. Routing:

- Apple / Metal: Qwen 3.8 27B GSQ-RCO UD-Q3_K_XL via llama.cpp at 24 GB+
  unified memory (Fit List winner, evidence
  qwen-3-8-27b-gsq-rco-iq3s-auto-262k); Cyber-Tiel Coder below 24 GB. The
  OrcaRouter Uncensored MLX band and the MLX 2-bit build (uploader: archival,
  quality collapse) are retired from the default Apple route.
- Integrated / RAM-only (non-Apple): Maple at 8 GB total, Cyber-Tiel Coder at 16 GB,
  and Qwen 3.8 27B GSQ-RCO at 24 GB+.
- Dedicated 8 GB: Maple, plus Cyber-Tiel Coder when host RAM can hold offloaded experts.
- Dedicated 16 GB+: GSQ-RCO GGUF, with Maple as the auxiliary at every tier.

The auxiliary is Maple Preview TQ2_0 (`maple-preview-tq2`) on every route:
it is the only entry in ``ALLOWED_AUX``. ``auto`` (share the main process)
stays a runtime mode, never a second aux model.

The Asha/Escha mixed 2-bit (EschaLabs/Qwen3.8-27B-Escha-W2) is a research
candidate only: custom SGLang kernels, no llama.cpp recipe, no TurboFit TPS.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from . import strata_tiers

ALLOWED_MAIN = (
    "maple-preview-tq2",
    "bonsai-27b",
    "bonsai-27b-q1",
    "bonsai-27b-1bit",
    "cyber-tiel-coder-35b-a3b-ud-iq3-xxs",
    "cyber-tiel-coder-35b-a3b-ud-q4-k-xl",
    "cyber-tiel-coder-35b-a3b-ud-q5-k-xl",
    "qwen3-8-27b-gsq-rco-iq3xxs",
    "qwen3-8-27b-gsq-rco-iq3s",
    "qwen3-8-27b-gsq-rco",
    "qwen3-8-27b-bf16",
    "qwen3-8-27b-q8",
    "qwen3-8-27b-uncensored-mlx-4bit",
    "qwen3-8-27b-uncensored-mlx-6bit",
    "qwen3-8-27b-uncensored-mlx-8bit",
    "cyber-tiel-coder-35b-a3b",
    "qwen3-8-flash-next-strata",
    "qwen3-8-flash-next-iq3xxs",
    "darwin-180b-rsi-strata",
    "arete-27b-q3",
    "arete-27b-q4",
    "arete-27b-q5",
    "arete-27b-q8",
    "nanbeige-33b-iq3xs",
    "nanbeige-33b-iq4xs",
    "flint-40a2-5b-mix12g",
    "flint-40a2-5b-mix24g",
    "deepseek-v4-1-flash-1017-iq1m",
    "glm-5-3-flash-maya-xs",
    "glm-5-3-flash-maya-s24",
    "glm-5-3-flash-maya-m",
    "glm-5-3-flash-maya-l",
)

ALLOWED_AUX = (
    "maple-preview-tq2",
)

_BANNED = re.compile(
    r"(?:^|[-_])9b(?:$|[-_])|qwen3\.5-9b|qwen3\.8-9b|mlx-2bit|uncensored-mlx-2",
    re.IGNORECASE,
)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def is_banned(alias: str) -> bool:
    return bool(_BANNED.search(normalize(alias)))


def is_allowed_main(alias: str) -> bool:
    token = normalize(alias)
    if is_banned(token):
        return False
    allowed = {normalize(item) for item in ALLOWED_MAIN}
    return token in allowed or any(token.startswith(item) and "9b" not in token for item in allowed)


def is_allowed_aux(alias: str) -> bool:
    """True only for Maple or an ``auto`` shared-main alias.

    ``auto``/``auto:<main>`` mean "no separate auxiliary process"; every real
    auxiliary is Maple. A non-Maple drafter is never allowed.
    """
    token = normalize(alias)
    if is_banned(token):
        return False
    return token in {normalize(item) for item in ALLOWED_AUX} or token.startswith("auto")


def expert_residency(*, host_ram_gb: float) -> dict[str, object]:
    """Where the auxiliary model's experts live, and what it costs the GPU.

    The auxiliary carries 40-85% of Hermes tokens, so it must be cheap: its
    weights stay memory-mapped on disk and the active experts stream into host
    RAM (>=32 GB) or straight off the SSD (below that). ``gpu_layers`` is 0 in
    both bands on purpose — the auxiliary keeps VRAM entirely free for the main
    model. A caller that wants the auxiliary on the GPU passes an explicit
    ``gpu_layers`` override; this default never claims a card.
    """
    if host_ram_gb >= 32:
        return {
            "mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False,
            "gpu_layers": 0,
        }
    return {
        "mode": "disk", "cpu_moe": True, "mmap": True, "mlock": False,
        "gpu_layers": 0,
    }


def apple_mlx_main(*, unified_ram_gb: float) -> str:
    """Fit List route for Apple unified memory.

    24 GB+ shared memory runs Qwen 3.8 27B GSQ-RCO UD-Q3_K_XL
    (evidence: qwen-3-8-27b-gsq-rco-iq3s-auto-262k), not the
    retired OrcaRouter Uncensored MLX band. Sub-24 GB Macs stay Cyber-Tiel Coder;
    the MLX 2-bit build is banned.
    """
    if unified_ram_gb >= 24:
        return "qwen3-8-27b-gsq-rco-iq3s"
    return "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"


def local_main_for_vram_gb(vram_gb: float) -> str:
    if vram_gb < 16:
        return "maple-preview-tq2"
    if vram_gb < 24:
        return "qwen3-8-27b-gsq-rco-iq3xxs"
    if vram_gb < 96:
        return "qwen3-8-27b-gsq-rco-iq3s"
    return "qwen3-8-27b-bf16"


def speed_main_for_vram_gb(vram_gb: float) -> str:
    """Faster MoE over dense 27B when the user picks speed."""
    if vram_gb < 16:
        return "maple-preview-tq2"
    if vram_gb < 24:
        return "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"
    # Above Q4 is not a default answer at any tier: more VRAM does not buy a
    # wider quant, it buys context headroom (see quant_policy).
    return "cyber-tiel-coder-35b-a3b-ud-q4-k-xl"


def speed_family_rank(alias: str) -> int:
    token = normalize(alias)
    # Q5_K_XL and Q4_K_XL share one rank: quant width is a ceiling, not a
    # scoring axis, so a wider build never outranks a narrower one.
    if "ud-q5-k-xl" in token or "ud-q4-k-xl" in token:
        return 45
    if token.startswith("cyber-tiel-coder"):
        return 40
    if token.startswith("maple"):
        return 30
    if "gsq-rco" in token or token.startswith("qwen3-8-27b"):
        return 5
    return 10


def shared_main_for_total_memory_gb(total_memory_gb: float) -> str:
    """Select by the one shared pool; never interpret it as dedicated VRAM."""
    if total_memory_gb < 16:
        return "maple-preview-tq2"
    if total_memory_gb < 24:
        return "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"
    return "qwen3-8-27b-gsq-rco-iq3s"


def local_aux_for_host(*, vram_gb: float, host_ram_gb: float) -> str:
    """The auxiliary is Maple Preview on every host.

    Maple TQ2_0 is tiny (~6 GB on disk), streams from host RAM / disk, and
    never blocks the main model's VRAM. There is no tier-based alternative and
    no non-Maple opt-in: the auxiliary is Maple across the board.
    """
    return "maple-preview-tq2"


def low_vram_moe_main() -> str:
    return "cyber-tiel-coder-35b-a3b-ud-iq3-xxs"


def recommendable_mains_for_hardware(hardware: object) -> frozenset[str]:
    """Mains Check may offer. Host-spill of a dense 27B is not a recommendation."""
    vram_gb = float(getattr(hardware, "total_vram_mb", 0) or 0) / 1024
    host_ram_gb = float(getattr(hardware, "system_ram_mb", 0) or 0) / 1024
    pool = str(getattr(hardware, "memory_pool_kind", "dedicated") or "dedicated")
    devices = getattr(hardware, "devices", ()) or ()
    backend = devices[0].backend if devices else "cpu"
    vendor = devices[0].vendor if devices else ""
    if pool == "cpu":
        pool = "unified"
        vram_gb = 0
    return frozenset(
        normalize(str(item["alias"]))
        for item in check_local_options(
            vram_gb=vram_gb,
            host_ram_gb=host_ram_gb,
            memory_pool=pool,
            backend=backend,
            vendor=vendor,
        )
    )


def main_matches_recommendable(alias: str, allowed: Iterable[str]) -> bool:
    token = normalize(alias)
    return any(token == item or token.startswith(f"{item}-") for item in allowed)


def strata_options(
    *,
    vram_gb: float,
    host_ram_gb: float,
    cpu_flags: Iterable[str] | None,
    backend: str = "cuda",
    root: str | None = None,
) -> tuple[list[dict[str, object]], strata_tiers.StrataTierView | None]:
    """Strata-served mains this dedicated host admits, ranked before llama.cpp.

    ``cpu_flags`` is the probed CPU flag set (``cpu_features.cpu_flags()``).
    ``None`` means "not interrogated": no Strata row is claimed and none is
    excluded, because neither answer would be honest. A host whose flags lack
    any of ``STRATA_REQUIRED`` gets no rows either — the recommendation surface
    carries the exclusion reason from :func:`strata_tiers.capability`, which
    names the missing flags.

    Strata is a Linux CUDA/ROCm engine per ``references/engine-serve-matrix.json``,
    so the Vulkan/Metal/CPU paths are untouched.
    """
    if backend not in {"cuda", "rocm"}:
        return [], None
    # Below the smallest band the raw figure is still the honest tier: a 4 GB
    # card must be refused with the documented floor, not silently ignored.
    tier_gb = strata_tiers.tier_for_vram(vram_gb) or int(vram_gb)
    if tier_gb <= 0:
        return [], None
    view = strata_tiers.tier_view(
        tier_gb=tier_gb,
        ram_bytes=max(0, int(host_ram_gb * strata_tiers.GIB)),
        cpu_flags=cpu_flags,
        root=root,
    )
    options: list[dict[str, object]] = []
    # The recommendation surface lists the strongest three admitted rows; the
    # tier report carries the full admitted set for audit.
    for placement in view.admitted[:3]:
        options.append({
            "role": "main",
            "alias": placement.model_id,
            "engine": "strata",
            "context": placement.context,
            "required_context": placement.context,
            "why": (
                view.reason if not options
                else placement.reason
            ),
            "residency": expert_residency(host_ram_gb=host_ram_gb),
        })
    return options, view


def check_local_options(
    *,
    vram_gb: float,
    host_ram_gb: float,
    memory_pool: str = "dedicated",
    backend: str = "cuda",
    vendor: str = "nvidia",
    cpu_flags: Iterable[str] | None = None,
    strata_root: str | None = None,
) -> list[dict[str, object]]:
    residency = expert_residency(host_ram_gb=host_ram_gb)
    apple = vendor == "apple" or backend == "metal"
    unified = memory_pool == "unified" or apple
    # A Strata-capable host gets its admitted MoE rows ahead of every llama.cpp
    # row at this tier: intelligence per tier, not tokens per second alone.
    preferred, _strata_view = (
        ([], None) if (apple or unified)
        else strata_options(
            vram_gb=vram_gb, host_ram_gb=host_ram_gb, cpu_flags=cpu_flags,
            backend=backend, root=strata_root,
        )
    )

    if apple:
        main = apple_mlx_main(unified_ram_gb=host_ram_gb)
        is_gsq_rco = main == "qwen3-8-27b-gsq-rco-iq3s"
        return [
            {
                "role": "main",
                "alias": main,
                "engine": "llama.cpp",
                "repo": None,
                "revision": None,
                "why": (
                    "Apple 24 GB+ unified: GSQ-RCO IQ3_S per the Fit List "
                    "(OrcaRouter Uncensored MLX band is retired)."
                    if is_gsq_rco
                    else "Apple under 24 GB unified: Cyber-Tiel Coder 35B-A3B, not MLX 2-bit."
                ),
                "residency": residency if main == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs" else None,
            },
            {
                "role": "aux",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": (
                    "Maple Preview 20B-A1B streams from host RAM / disk and "
                    "never blocks the main model's VRAM — default auxiliary."
                ),
                "residency": {"mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False},
            },
        ]

    if unified:
        main = shared_main_for_total_memory_gb(host_ram_gb)
        options = [
            {
                "role": "main",
                "alias": main,
                "engine": "llama.cpp",
                "why": (
                    "Integrated/unified memory uses total-memory bands: "
                    "Maple at 8 GB, Cyber-Tiel Coder at 16 GB, GSQ-RCO at 24 GB+."
                ),
                "residency": residency if main == "cyber-tiel-coder-35b-a3b-ud-iq3-xxs" else None,
            },
        ]
        # Dense exception: a 27B at q3+ fits a 16 GB unified pool, but it is not
        # the default because MoE decodes far faster on shared memory.
        if host_ram_gb >= 16:
            dense = "arete-27b-q4" if host_ram_gb >= 24 else "arete-27b-q3"
            options.append(
                {
                    "role": "main",
                    "alias": dense,
                    "engine": "llama.cpp",
                    "why": (
                        "Dense-exception option: Arete 27B at q3+ fits the unified "
                        "pool, but expect fewer tokens/sec than the MoE default."
                    ),
                }
            )
        return options + [
            {
                "role": "aux",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": (
                    "Maple Preview 20B-A1B streams from host RAM / disk and "
                    "never blocks the main model's VRAM — default auxiliary."
                ),
                "residency": {"mode": "ram", "cpu_moe": True, "mmap": True, "mlock": False},
            },
        ]

    tier_gb = strata_tiers.tier_for_vram(vram_gb)
    ram_bytes = max(0, int(host_ram_gb * strata_tiers.GIB))
    return strata_ranked(
        preferred,
        _dedicated_options(vram_gb, host_ram_gb, residency),
        tier_gb=tier_gb,
        ram_bytes=ram_bytes,
        root=strata_root,
        view=_strata_view,
    )


def _dedicated_options(
    vram_gb: float, host_ram_gb: float, residency: dict[str, object]
) -> list[dict[str, object]]:
    """The llama.cpp band ladder for a dedicated host, before the 64K gate."""
    if vram_gb < 16:
        options: list[dict[str, object]] = [
            {
                "role": "main",
                "alias": "maple-preview-tq2",
                "engine": "llama.cpp",
                "why": "20B-A1B Maple TQ2_0 fits within the dedicated 8 GB VRAM band.",
            },
        ]
        if vram_gb >= 12:
            # Arete 27B Q3_K_S is 12.5 GiB with its projector: it cannot hold 64K
            # on a 12 GB card, so the 12 GB rung is the 27B GSQ-RCO IQ3_XXS
            # (10.3 GiB), which does.
            options.append(
                {
                    "role": "main",
                    "alias": "qwen3-8-27b-gsq-rco-iq3xxs",
                    "engine": "llama.cpp",
                    "why": (
                        "Qwen 3.8 27B GSQ-RCO IQ3_XXS (~10.3 GiB) holds 64K on a "
                        "12 GB card; a dense Q3 at 12.5 GiB cannot."
                    ),
                }
            )
        if host_ram_gb >= 16:
            options.append(
                {
                    "role": "main",
                    "alias": "cyber-tiel-coder-35b-a3b-ud-iq3-xxs",
                    "engine": "llama.cpp",
                    "why": (
                        "35A3B MoE alternative when host RAM can hold offloaded "
                        f"experts via {residency['mode']}-backed mmap."
                    ),
                    "residency": residency,
                }
            )
        return options
    options = [
        {
            "role": "main",
            "alias": local_main_for_vram_gb(vram_gb),
            "engine": "llama.cpp",
            "why": "Intelligence / balanced: GSQ-RCO / Qwen band for this VRAM.",
        },
        {
            "role": "main",
            "alias": speed_main_for_vram_gb(vram_gb),
            "engine": "llama.cpp",
            "why": "Speed: Cyber-Tiel Coder MoE instead of dense 27B.",
        },
    ]
    # Named alternatives across the matrix. These never displace the band default;
    # they are the "every model this machine can run" set from the Fit List.
    if vram_gb < 24:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q4",
                "engine": "llama.cpp",
                "why": "Arete 27B Q4_K_M (~18 GiB) at 128K context on a 16-23 GB card.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix12g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-12G: MoE decode speed at a 12 GB footprint.",
            }
        )
    elif vram_gb < 64:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q4",
                "engine": "llama.cpp",
                "why": "Arete 27B Q4_K_M (~18 GiB) at 262K context: the dense quality pick.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "nanbeige-33b-iq4xs",
                "engine": "llama.cpp",
                "why": "Nanbeige 4.3 33B IQ4_XS (~17 GiB): a 33B dense alternative.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix24g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-24G for the larger card in this band.",
            }
        )
    else:
        options.append(
            {
                "role": "main",
                "alias": "arete-27b-q8",
                "engine": "llama.cpp",
                "why": "Arete 27B Q8_0 (~27 GiB): near-lossless dense at 64 GB+.",
            }
        )
        options.append(
            {
                "role": "main",
                "alias": "flint-40a2-5b-mix24g",
                "engine": "llama.cpp",
                "why": "Flint 40A2.5B Mix-24G for the larger card in this band.",
            }
        )
    options.append(
        {
            "role": "aux",
            "alias": local_aux_for_host(vram_gb=vram_gb, host_ram_gb=host_ram_gb),
            "engine": "llama.cpp",
            "why": "Default aux is Maple Preview when total VRAM can hold it next to main.",
            "residency": residency,
        }
    )
    return options


def _known_artifact(alias: str, root: str | None) -> bool:
    import json
    from pathlib import Path

    base = Path(root) if root is not None else strata_tiers.default_root()
    payload = json.loads((base / "references/model-footprints.json").read_text(encoding="utf-8"))
    return normalize(alias) in (payload.get("models") or {})


def holds_floor(
    alias: str, *, tier_gb: int, ram_bytes: int, root: str | None = None
) -> bool | None:
    """Whether a main holds >= 64K at this tier; ``None`` when it is unmeasured.

    Offload counts: an MoE that legitimately keeps experts in RAM passes, while
    a dense build that only fits by spilling cannot. An artifact the footprint
    table does not know is ``None`` (unknown), never a silent refusal.
    """
    return strata_tiers.floor_state(
        normalize(alias), tier_gb=tier_gb, ram_bytes=ram_bytes, root=root,
    )


def floor_refusals(
    *,
    vram_gb: float,
    host_ram_gb: float,
    memory_pool: str = "dedicated",
    backend: str = "cuda",
    vendor: str = "nvidia",
    strata_root: str | None = None,
) -> list[dict[str, object]]:
    """Banded mains this tier would have offered that cannot hold the 64K floor.

    Kept separate from :func:`check_local_options` so the operator's report can
    say *which* recommendation the floor deleted, not merely that the tier got
    shorter.
    """
    apple = vendor == "apple" or backend == "metal"
    unified = memory_pool == "unified" or apple
    if apple or unified:
        return []
    tier_gb = strata_tiers.tier_for_vram(vram_gb)
    if tier_gb <= 0:
        return []
    ram_bytes = max(0, int(host_ram_gb * strata_tiers.GIB))
    refusals: list[dict[str, object]] = []
    for item in _dedicated_options(vram_gb, host_ram_gb, expert_residency(host_ram_gb=host_ram_gb)):
        if item.get("role") != "main":
            continue
        alias = str(item["alias"])
        if holds_floor(alias, tier_gb=tier_gb, ram_bytes=ram_bytes, root=strata_root) is False:
            refusals.append({
                "model": normalize(alias),
                "reason": (
                    f"below the {65_536:,} token (64K) floor: no placement at "
                    f"{tier_gb} GB holds 65,536 tokens"
                ),
            })
    return refusals


def strata_ranked(
    preferred: list[dict[str, object]],
    options: list[dict[str, object]],
    *,
    tier_gb: int,
    ram_bytes: int,
    root: str | None,
    view: strata_tiers.StrataTierView | None,
) -> list[dict[str, object]]:
    """Strata-first ordering when admitted, with sub-floor mains dropped.

    Strata rows lead only when the view says they hold at least the context of
    the best llama.cpp option at this tier. When llama.cpp holds more context,
    its rows stay ahead and the view reason names that, instead of quietly
    ranking a shorter-context model first.
    """
    kept = [
        item for item in options
        if item.get("role") != "main"
        or item.get("engine") == "strata"
        or holds_floor(
            str(item["alias"]), tier_gb=tier_gb, ram_bytes=ram_bytes, root=root,
        ) is not False
    ]
    if preferred and view is not None and view.preferred_over_llamacpp:
        return preferred + kept
    return kept + preferred
def filter_aliases(aliases: Iterable[str], *, role: str) -> list[str]:
    checker = is_allowed_main if role == "main" else is_allowed_aux
    return [alias for alias in aliases if checker(alias)]
