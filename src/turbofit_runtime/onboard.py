"""First run: scan this host, recommend one model, fetch only that.

TurboFit's promise is scan → recommend → download, not "download the library".
Before this module the pieces existed but nothing joined them: the fit view
could say what was missing, and ``scripts/download-artifacts`` could fetch a
family, but the first thing a new user ran was the downloader over families
they had not chosen.

So the plan built here has three properties, in order of importance:

* **One recommendation.** It is the measured tier winner for this host's
  capacity band when a measurement exists. When none does, the plan says so and
  offers the fallback chain's first rung that fits this machine, labelled
  ``portable-fit`` -- never a benchmarked claim it cannot back.
* **Only what was chosen.** The plan lists the recommendation's artifacts with
  byte sizes, and *separately* the optional fallback rungs. Nothing outside
  that set is ever in the download list; there is no ``--all`` on this path,
  and the CLI rejects one.
* **No assumption of presence.** Every artifact is re-checked with the
  campaign's own "installed" definition (:mod:`turbofit_runtime.artifacts`), so
  a complete file is skipped and reported, a stub is replaced, and an absent
  file is counted in the bytes the user is about to spend.

Downloading is delegated to ``scripts/download-artifacts`` -- the shard-aware,
SHA-verifying materializer -- so this module cannot invent a second way to put
bytes on disk.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from .artifacts import human_bytes, inspect_file, parse_shard
from .chain import load_chain
from .cpu_features import cpu_flags, strata_supported
from .experts import available_ram_bytes
from .fit_view import (
    STATUS_ENGINE,
    STATUS_MEMORY,
    STATUS_MISSING,
    STATUS_READY,
    STATUS_UNRESOLVABLE,
    component_artifacts,
    load_fit_view,
)
from .hardware import HardwareFingerprint, probe_hardware
from .tier_report import native_tier

SCHEMA = "turbofit.onboard-plan/v1"
RESULT_SCHEMA = "turbofit.onboard-result/v1"
ROOT = Path(__file__).resolve().parents[2]

GIB = 2 ** 30

#: What the plan is doing, in one word, so a caller never parses a message.
PLAN_READY = "ready-to-download"     # artifacts are missing; the plan fetches them
PLAN_COMPLETE = "already-installed"  # nothing to fetch, everything verified
PLAN_BLOCKED = "blocked"             # cannot recommend, or cannot run here
PLAN_UNMEASURED = "no-measurements"

POLICY_NONE = "none"
POLICY_ALL = "all"

_FALLBACK_NONE_WORDS = {"", "none", "off", "no", "false", "0"}

_DOWNLOADER: Any = None


class FallbackPolicyError(ValueError):
    """``--include-fallback`` was not ``none``, ``all`` or a rung number."""


def parse_fallback_policy(value: str | int | None) -> str | int:
    """Normalise the fallback selector. ``none`` is the default, always."""
    if value is None:
        return POLICY_NONE
    if isinstance(value, bool):
        raise FallbackPolicyError("include_fallback must be none, all, or a rung number")
    if isinstance(value, int):
        if value < 1:
            raise FallbackPolicyError("include_fallback rung numbers start at 1")
        return value
    text = str(value).strip().lower()
    if text in _FALLBACK_NONE_WORDS:
        return POLICY_NONE
    if text in {POLICY_ALL, "*"}:
        return POLICY_ALL
    if text.isdigit():
        number = int(text)
        if number < 1:
            raise FallbackPolicyError("include_fallback rung numbers start at 1")
        return number
    raise FallbackPolicyError(
        f"include_fallback must be none, all, or a rung number (got {value!r})"
    )


# ---------------------------------------------------------------------------
# Host scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostScan:
    """What this machine is, stated once and reused by the whole plan."""

    hardware: HardwareFingerprint
    tier_gb: int
    tier_source: str
    lane: str
    per_device_gb: float
    ram_bytes: int
    ram_source: str
    ram_available_bytes: int | None
    flags: frozenset[str] | None

    @property
    def device_count(self) -> int:
        return len(self.hardware.devices)

    @property
    def strata(self) -> bool | None:
        return strata_supported(self.flags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "os": self.hardware.os,
            "architecture": self.hardware.architecture,
            "devices": [
                {
                    "index": device.index,
                    "name": device.name,
                    "vendor": device.vendor,
                    "backend": device.backend,
                    "memory_total_mb": device.memory_total_mb,
                }
                for device in self.hardware.devices
            ],
            "device_count": self.device_count,
            "topology_key": self.hardware.topology_key,
            "memory_pool": self.hardware.memory_pool_kind,
            "lane": self.lane,
            "per_device_gb": round(self.per_device_gb, 2),
            "total_memory_gb": round(
                (
                    self.hardware.total_usable_memory_mb
                    if self.hardware.shared_memory_pool
                    else self.hardware.total_vram_mb
                )
                / 1024,
                2,
            ),
            "system_ram_gb": round(self.hardware.system_ram_mb / 1024, 2),
            "ram_bytes": self.ram_bytes,
            "ram_gb": round(self.ram_bytes / GIB, 2),
            "ram_source": self.ram_source,
            "ram_available_bytes": self.ram_available_bytes,
            "cpu_flags_known": self.flags is not None,
            "strata_supported": self.strata,
        }


def scan_host(
    hardware: HardwareFingerprint | None = None,
    *,
    ram_bytes: int | None = None,
    flags: frozenset[str] | None = None,
    tier_gb: int | None = None,
    probe_flags: bool = True,
) -> HostScan:
    """Detect the host and the capacity band it belongs to.

    The band comes from :func:`turbofit_runtime.tier_report.native_tier` -- the
    same function the hardware-tier report uses -- so the first-run download and
    the published tier table can never disagree about which tier this is.
    """
    fingerprint = hardware if hardware is not None else probe_hardware()
    if tier_gb is not None:
        if isinstance(tier_gb, bool) or int(tier_gb) < 1:
            raise ValueError("tier_gb must be a positive integer")
        tier, tier_source = int(tier_gb), "explicit"
    else:
        tier, tier_source = native_tier(fingerprint), "host"

    available = available_ram_bytes()
    if ram_bytes is not None:
        if int(ram_bytes) <= 0:
            raise ValueError("ram_bytes must be positive")
        ram, ram_source = int(ram_bytes), "explicit"
    elif available:
        ram, ram_source = int(available), "host-memavailable"
    else:
        ram, ram_source = int(fingerprint.system_ram_mb) * 1024, "host-installed"

    flagset = flags
    if flagset is None and probe_flags:
        flagset = cpu_flags()
    per_device = (
        min(device.memory_total_mb for device in fingerprint.devices) / 1024
        if fingerprint.devices
        else float(tier)
    )
    return HostScan(
        hardware=fingerprint,
        tier_gb=tier,
        tier_source=tier_source,
        lane=fingerprint.memory_pool_kind,
        per_device_gb=per_device,
        ram_bytes=ram,
        ram_source=ram_source,
        ram_available_bytes=available,
        flags=flagset,
    )


# ---------------------------------------------------------------------------
# Artifact planning
# ---------------------------------------------------------------------------


def _manifest_index(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    by_destination: dict[str, dict] = {}
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        destination = str(row.get("destination") or "")
        if not destination:
            continue
        entry = dict(row)
        by_destination[destination] = entry
        by_name.setdefault(PurePosixPath(destination).name, []).append(entry)
    return by_destination, by_name


def _fetch_row_for(
    path: str,
    *,
    model_root: Path,
    index: tuple[dict[str, dict], dict[str, list[dict]]],
) -> dict | None:
    """The pinned manifest row that can materialize *path*, if one exists.

    Matched by the manifest's own destination relative to the model root, then
    by basename. An ambiguous basename match is refused rather than guessed: a
    wrong match would fetch a different quant than the one the recipe runs.
    """
    by_destination, by_name = index
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(model_root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = ""
    if relative and relative in by_destination:
        return by_destination[relative]
    matches = by_name.get(candidate.name) or []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # Case and separators drift between a recipe and the manifest
        # (``Q4_K_XL`` vs ``Q4-K-XL``); a normalised basename still means the
        # same file, while a different quant stays unmatched.
        wanted = _normalised_name(candidate.name)
        matches = [
            row
            for name, rows in by_name.items()
            if _normalised_name(name) == wanted
            for row in rows
        ]
    if len(matches) == 1:
        return matches[0]
    return None


def _normalised_name(name: str) -> str:
    return "".join(char for char in str(name).lower() if char.isalnum())


def _full_size(manifest_row: Mapping[str, Any] | None, footprints: Mapping[str, Any]) -> tuple[int | None, str]:
    """Full, on-disk size of the artifact a row stands for, and its basis.

    A manifest row for shard 1 of N pins *that shard's* size, so summing rows
    understates a sharded model. ``model-footprints.json`` carries the measured
    whole-set size, and that is what the user is actually about to download.
    """
    if not manifest_row:
        return None, "unknown"
    raw = manifest_row.get("size_bytes")
    pinned = int(raw) if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0 else None
    models = footprints.get("models") if isinstance(footprints, Mapping) else None
    # The measured whole-set size belongs to the model's *weights*, not to every
    # file that shares its family list (a 2 GiB mmproj is not a 23 GiB model).
    # It is the right answer only for the shard set head or for a single-file
    # artifact whose pinned size already agrees with it.
    for family in manifest_row.get("families") or ():
        record = (models or {}).get(str(family))
        weight = record.get("weight_bytes") if isinstance(record, Mapping) else None
        if not weight:
            continue
        weight = int(weight)
        if parse_shard(str(manifest_row.get("path") or "")) is not None:
            if pinned and pinned > weight * 1.05:
                continue
            return weight, "measured-footprint"
        if pinned and abs(pinned - weight) <= max(1024, weight // 20):
            return pinned, "manifest-pinned"
    if pinned:
        return pinned, "manifest-pinned"
    return None, "unknown"


def _artifact_entry(
    artifact: Mapping[str, Any],
    *,
    model_root: Path,
    index: tuple[dict[str, dict], dict[str, list[dict]]],
    footprints: Mapping[str, Any],
) -> dict[str, Any]:
    """One planned artifact: what it is, whether it is here, what it costs."""
    path = str(artifact.get("path") or "")
    fetch_row = _fetch_row_for(path, model_root=model_root, index=index)
    present = bool(artifact.get("present"))
    on_disk = int(artifact.get("size_bytes") or 0)
    full_bytes, basis = _full_size(fetch_row, footprints)
    if full_bytes is None and isinstance(artifact.get("expected_bytes"), int):
        full_bytes, basis = int(artifact["expected_bytes"]), "manifest-pinned"

    if present:
        download_bytes, action, state = 0, "present", "complete"
    elif full_bytes is None:
        # Unknown size: still planned (the fetch proves it), but the total is
        # labelled rather than silently counted as zero.
        download_bytes = 0
        action = "download" if fetch_row else "unfetchable"
        state = str(artifact.get("reason") or "missing")
    else:
        download_bytes = max(0, full_bytes - on_disk)
        action = "download" if fetch_row else "unfetchable"
        state = str(artifact.get("reason") or "missing")

    entry: dict[str, Any] = {
        "role": str(artifact.get("role") or "main:model"),
        "path": path,
        "present": present,
        "state": state,
        "on_disk_bytes": on_disk,
        "full_bytes": full_bytes,
        "bytes_basis": basis,
        "download_bytes": download_bytes,
        "action": action,
    }
    shard = parse_shard(PurePosixPath(path).name) if path else None
    if shard is not None:
        entry["shard_count"] = shard[1]
        entry["note"] = (
            f"part of a {shard[1]}-shard set; the fetch resolves and verifies "
            "every shard, and the size shown is the whole measured set"
        )
    if fetch_row is not None:
        entry["download"] = {
            "repo_id": str(fetch_row.get("repo_id") or ""),
            "revision": str(fetch_row.get("revision") or ""),
            "path": str(fetch_row.get("path") or ""),
            "destination": str(fetch_row.get("destination") or ""),
            "sha256": str(fetch_row.get("sha256") or ""),
            "size_bytes": int(fetch_row.get("size_bytes") or 0),
            "families": [str(name) for name in (fetch_row.get("families") or [])],
        }
    else:
        entry["download"] = None
        entry["unfetchable_reason"] = (
            "no pinned manifest entry names this file, so TurboFit has no "
            "repository to fetch it from and no hash to verify it against"
        )
    return entry


def _summarise(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(entries)
    to_download = [row for row in rows if row.get("action") == "download"]
    return {
        "artifact_count": len(rows),
        "artifacts_to_download": len(to_download),
        "artifacts_present": sum(1 for row in rows if row.get("present")),
        "download_bytes": sum(int(row.get("download_bytes") or 0) for row in to_download),
        "library_bytes": sum(int(row.get("full_bytes") or 0) for row in rows),
        "unknown_size_files": [
            row["path"] for row in to_download if row.get("full_bytes") is None
        ],
    }


def _entries_for(component: Any, *, model_root: Path, index: Any, footprints: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _artifact_entry(row, model_root=model_root, index=index, footprints=footprints)
        for row in component_artifacts(component, model_root)
    ]


# ---------------------------------------------------------------------------
# Chain rungs
# ---------------------------------------------------------------------------


def _strata_block(flavor: str, scan: HostScan) -> str:
    if flavor == "strata" and scan.strata is False:
        return "Strata needs AVX2, which this CPU does not expose."
    return ""


def _chain_levels(chain: Any, scan: HostScan) -> tuple[list[Any], int]:
    # A host with no accelerator has no dedicated VRAM to shed into: the
    # unified/CPU lane is the one whose rungs can actually run there.
    shared = scan.lane in {"unified", "cpu"}
    lane = scan.lane if scan.lane in chain.lanes else chain.default_lane(unified=shared)
    models = [level for level in chain.levels(lane) if level.kind == "model" and level.model]
    context_rungs = sum(1 for level in chain.levels(lane) if level.kind == "context")
    return models, context_rungs


def _chain_rung(
    level: Any,
    *,
    scan: HostScan,
    book: Any,
    model_root: Path,
    index: Any,
    footprints: Mapping[str, Any],
) -> dict[str, Any]:
    """One chain model rung: resolvable, sized, selectable only if it fits here."""
    needs = int(getattr(level, "needs_mb_per_card", 0) or 0)
    fits = needs == 0 or needs <= scan.per_device_gb * 1024
    base = {
        "index": None,
        "chain_level": level.id,
        "model": level.model,
        "engine": level.engine,
        "context": int(level.context or 0),
        "evidence": level.evidence,
        "needs_mb_per_card": needs,
        "fits_per_card": fits,
        "selectable": False,
        "included": False,
        "artifacts": [],
        "artifact_count": 0,
        "artifacts_to_download": 0,
        "artifacts_present": 0,
        "download_bytes": 0,
        "incremental_bytes": 0,
        "library_bytes": 0,
    }
    try:
        component = book.resolve_component(
            str(level.model), role="main", gpu="0", port=11605, context=int(level.context or 0)
        )
    except (ValueError, KeyError) as exc:
        return {**base, "reason": f"no launch recipe resolves this rung ({type(exc).__name__}: {exc})"}
    blocked = _strata_block(str(getattr(component, "engine", "") or ""), scan)
    if blocked:
        return {**base, "main_family": component.family, "reason": blocked}
    entries = _entries_for(component, model_root=model_root, index=index, footprints=footprints)
    if not entries:
        return {
            **base,
            "main_family": component.family,
            "reason": "the resolved recipe names no on-disk artifact to fetch",
        }
    totals = _summarise(entries)
    unfetchable = [row for row in entries if row.get("action") == "unfetchable"]
    rung = {
        **base,
        "main_family": component.family,
        "main_alias": component.alias,
        "artifacts": entries,
        "selectable": fits and not unfetchable,
        **totals,
        "incremental_bytes": totals["download_bytes"],
    }
    if unfetchable:
        rung["reason"] = (
            "the recipe resolves files no pinned manifest entry can fetch "
            f"({', '.join(PurePosixPath(row['path']).name for row in unfetchable)}); "
            "nothing here can be downloaded automatically"
        )
        return rung
    if not fits:
        rung["reason"] = (
            f"needs {needs / 1024:.1f} GB per card; this machine has {scan.per_device_gb:.1f} GB"
        )
    return rung


def _chain_rungs(
    chain: Any,
    *,
    scan: HostScan,
    book: Any,
    model_root: Path,
    index: Any,
    footprints: Mapping[str, Any],
    exclude: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Every chain model rung for this host, numbered over the selectable ones."""
    levels, context_rungs = _chain_levels(chain, scan)
    rungs: list[dict[str, Any]] = []
    position = 0
    for level in levels:
        if exclude and str(level.model) == exclude:
            continue
        rung = _chain_rung(
            level, scan=scan, book=book, model_root=model_root, index=index, footprints=footprints
        )
        if rung.get("selectable"):
            position += 1
            rung["index"] = position
        rungs.append(rung)
    return rungs, context_rungs


def _recommendation_from_rung(rung: Mapping[str, Any]) -> dict[str, Any]:
    """A chain rung promoted to the recommendation slot, without overclaiming."""
    return {
        "source": "fallback-chain",
        "evidence": "portable-fit",
        "status": STATUS_MISSING if rung.get("artifacts_to_download") else STATUS_READY,
        "configuration_id": None,
        "main": rung.get("model"),
        "main_family": rung.get("main_family"),
        "main_alias": rung.get("main_alias"),
        "context": rung.get("context"),
        "intelligence_score": None,
        "throughput_tps": None,
        "level": None,
        "engine": {
            "name": rung.get("engine") or "unknown engine",
            "flavor": rung.get("engine") or "",
            "binary": "",
            "installed": None,
            "usable": True,
            "reason": None,
        },
        "placement": {
            "kind": "gpu",
            "n_cpu_moe": 0,
            "ram_requirement": "",
            "label": (
                f"the fallback chain lists this rung at "
                f"{int(rung.get('needs_mb_per_card') or 0) / 1024:.1f} GB per card"
            ),
        },
        "auxiliary": None,
        "artifacts": list(rung.get("artifacts") or []),
        "artifacts_to_download": rung.get("artifacts_to_download", 0),
        "artifacts_present": rung.get("artifacts_present", 0),
        "download_bytes": rung.get("download_bytes", 0),
        "library_bytes": rung.get("library_bytes", 0),
        "unknown_size_files": rung.get("unknown_size_files") or [],
    }


def _recommendation_from_tier(
    entry: Mapping[str, Any],
    *,
    model_root: Path,
    index: Any,
    footprints: Mapping[str, Any],
) -> dict[str, Any] | None:
    rec = entry.get("recommendation")
    if not isinstance(rec, Mapping):
        return None
    entries = [
        _artifact_entry(row, model_root=model_root, index=index, footprints=footprints)
        for row in rec.get("artifacts") or []
        if isinstance(row, Mapping)
    ]
    totals = _summarise(entries)
    return {
        "source": "tier-measurement",
        "evidence": "measured-on-box",
        "status": entry.get("status"),
        "configuration_id": rec.get("configuration_id"),
        "main": rec.get("main"),
        "main_family": rec.get("main_family"),
        "main_alias": rec.get("main_alias"),
        "context": rec.get("context"),
        "intelligence_score": rec.get("intelligence_score"),
        "throughput_tps": rec.get("throughput_tps"),
        "level": rec.get("level"),
        "engine": rec.get("engine"),
        "placement": dict(rec.get("placement") or {}),
        "auxiliary": (rec.get("auxiliary") or {}).get("id"),
        "artifacts": entries,
        **totals,
    }


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _load_inputs(
    root: str | Path | None,
    *,
    recipe_book: Any,
    model_root: str | Path | None,
) -> tuple[Path, Any, Path, dict, dict, tuple]:
    base = Path(root or ROOT)
    book = recipe_book
    if book is None:
        from .recipes import RecipeBook

        book = RecipeBook.load(base / "references/model-recipes.json")
    if model_root is not None:
        book.model_root = Path(model_root).expanduser()
    weights = Path(
        model_root
        if model_root is not None
        else getattr(book, "model_root", Path.home() / "Models" / "storage" / "gguf")
    ).expanduser()
    manifest = json.loads((base / "references/artifact-manifest.json").read_text(encoding="utf-8"))
    footprints = json.loads((base / "references/model-footprints.json").read_text(encoding="utf-8"))
    return base, book, weights, manifest, footprints, _manifest_index(manifest.get("artifacts") or [])


def _tier_blocker(tier_entry: Mapping[str, Any]) -> str:
    """Why the measured recommendation cannot be used here, if it cannot."""
    status = tier_entry.get("status")
    recommendation = tier_entry.get("recommendation") or {}
    engine = recommendation.get("engine") or {}
    if status == STATUS_MEMORY:
        return str(tier_entry.get("message") or "this configuration cannot be loaded on this machine.")
    if status == STATUS_UNRESOLVABLE:
        return str(tier_entry.get("message") or "this configuration cannot be resolved into a launch recipe.")
    if engine.get("flavor") == "strata" and engine.get("usable") is False:
        return str(engine.get("reason") or "Strata cannot decode on this CPU.")
    return ""


def _renumber(rungs: list[dict[str, Any]]) -> None:
    position = 0
    for rung in rungs:
        if rung.get("selectable"):
            position += 1
            rung["index"] = position
        else:
            rung["index"] = None


def build_plan(
    root: str | Path | None = None,
    *,
    hardware: HardwareFingerprint | None = None,
    ram_bytes: int | None = None,
    flags: frozenset[str] | None = None,
    tier_gb: int | None = None,
    probe_flags: bool = True,
    include_fallback: str | int | None = POLICY_NONE,
    model_root: str | Path | None = None,
    recipe_book: Any = None,
    view: Mapping[str, Any] | None = None,
    chain: Any = None,
    binary_installed: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Scan → recommend → the exact list of bytes this run would fetch.

    Pure planning: nothing here touches the network or writes a file. Call
    :func:`execute_plan` with the result to actually fetch.
    """
    policy = parse_fallback_policy(include_fallback)
    base, book, weights, manifest, footprints, index = _load_inputs(
        root, recipe_book=recipe_book, model_root=model_root
    )
    scan = scan_host(
        hardware, ram_bytes=ram_bytes, flags=flags, tier_gb=tier_gb, probe_flags=probe_flags
    )
    if view is None:
        view = load_fit_view(
            base,
            tier_gb=scan.tier_gb,
            flags=scan.flags,
            probe_flags=False,
            binary_installed=binary_installed,
            recipe_book=book,
        )
    chain_doc = chain or load_chain(base / "references/fallback-chain.json")

    tiers = [tier for tier in view.get("tiers") or [] if isinstance(tier, Mapping)]
    tier_entry = next(
        (tier for tier in tiers if int(tier.get("hardware_tier_gb") or 0) == scan.tier_gb), None
    )
    if tier_entry is None and len(tiers) == 1:
        tier_entry = tiers[0]
    tier_measured = bool(tier_entry and tier_entry.get("measured"))
    recommendation, blocker, all_rungs, context_rungs = _choose_recommendation(
        tier_entry=tier_entry,
        tier_measured=tier_measured,
        chain_doc=chain_doc,
        scan=scan,
        book=book,
        weights=weights,
        index=index,
        footprints=footprints,
    )
    if recommendation is None:
        return _blocked_plan(
            base=base,
            scan=scan,
            view=view,
            tier_measured=tier_measured,
            blocker=blocker,
            tier_entry=tier_entry,
            policy=policy,
            rungs=all_rungs,
            context_rungs=context_rungs,
        )

    rungs, included, fallback, download_entries, selected, unfetchable, status = _plan_fallback(
        all_rungs=all_rungs,
        context_rungs=context_rungs,
        recommendation=recommendation,
        policy=policy,
        scan=scan,
        chain_doc=chain_doc,
        book=book,
        weights=weights,
        index=index,
        footprints=footprints,
    )
    total_bytes = sum(int(entry.get("download_bytes") or 0) for entry in download_entries)
    plan = {
        "ok": True,
        "schema": SCHEMA,
        "root": str(base),
        "mode": "dry-run",
        "status": status,
        "host": scan.as_dict(),
        "tier": {
            "hardware_tier_gb": scan.tier_gb,
            "tier_source": scan.tier_source,
            "measured": tier_measured,
        },
        "cpu": dict((view or {}).get("cpu") or {}),
        "recommendation": recommendation,
        "fallback": fallback,
        "download": {
            "entries": download_entries,
            "destinations": [
                str((entry.get("download") or {}).get("destination")) for entry in download_entries
            ],
            "families": sorted(
                {
                    family
                    for entry in download_entries
                    for family in ((entry.get("download") or {}).get("families") or [])
                }
            ),
            "bytes": total_bytes,
        },
        "total": {
            "download_bytes": total_bytes,
            "download_gib": round(total_bytes / GIB, 2),
            "library_bytes": sum(int(entry.get("full_bytes") or 0) for entry in selected),
            "artifacts_to_download": len(download_entries),
            "artifacts_present": sum(1 for entry in selected if entry.get("present")),
            "unknown_size_files": [
                entry["path"] for entry in download_entries if entry.get("full_bytes") is None
            ],
            "unfetchable": unfetchable,
            "rungs_included": [int(rung["index"]) for rung in included],
        },
        "message": _plan_message(
            scan=scan,
            recommendation=recommendation,
            fallback=fallback,
            status=status,
            download_bytes=total_bytes,
            unfetchable=unfetchable,
        ),
        "next": _next_steps(status, bool(included), policy),
    }
    return plan


def _choose_recommendation(
    *,
    tier_entry: Mapping[str, Any] | None,
    tier_measured: bool,
    chain_doc: Any,
    scan: HostScan,
    book: Any,
    weights: Path,
    index: Any,
    footprints: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]], int]:
    """The measured tier winner, else the first chain rung that fits here."""
    rungs: list[dict[str, Any]] = []
    context_rungs = 0
    if tier_measured and tier_entry is not None:
        blocker = _tier_blocker(tier_entry)
        if blocker:
            return None, blocker, rungs, context_rungs
        return (
            _recommendation_from_tier(
                tier_entry, model_root=weights, index=index, footprints=footprints
            ),
            "",
            rungs,
            context_rungs,
        )
    rungs, context_rungs = _chain_rungs(
        chain_doc, scan=scan, book=book, model_root=weights, index=index, footprints=footprints
    )
    first = next((rung for rung in rungs if rung.get("selectable")), None)
    if first is None:
        return None, "", rungs, context_rungs
    return _recommendation_from_rung(first), "", rungs, context_rungs


def _plan_fallback(
    *,
    all_rungs: list[dict[str, Any]],
    context_rungs: int,
    recommendation: Mapping[str, Any],
    policy: str | int,
    scan: HostScan,
    chain_doc: Any,
    book: Any,
    weights: Path,
    index: Any,
    footprints: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str], str]:
    """The optional rungs: what they cost, which are included, and the fetch list."""
    if not all_rungs:
        all_rungs, context_rungs = _chain_rungs(
            chain_doc, scan=scan, book=book, model_root=weights, index=index, footprints=footprints
        )
    # The recommendation is one model; the chain is not allowed to re-offer it.
    own_families = {
        str(recommendation.get(field) or "")
        for field in ("main", "main_family", "configuration_id")
    }
    rungs = [rung for rung in all_rungs if str(rung.get("model") or "") not in own_families]
    _renumber(rungs)

    # A rung may need the very same file the recommendation already brings (a
    # measured winner is often also a chain rung). Those bytes are counted once,
    # and the rung says where they came from.
    claimed = {
        (entry.get("download") or {}).get("destination")
        for entry in recommendation.get("artifacts") or []
        if entry.get("download")
    }
    for rung in rungs:
        for entry in rung.get("artifacts") or []:
            destination = (entry.get("download") or {}).get("destination")
            if destination and destination in claimed:
                entry["shared_with"] = "recommendation"
                entry["download_bytes"] = 0
                if entry.get("action") == "download":
                    entry["action"] = "shared"
                continue
            if destination:
                claimed.add(destination)
        rung.update(_summarise(rung.get("artifacts") or []))
        rung["incremental_bytes"] = sum(
            int(entry.get("download_bytes") or 0)
            for entry in rung.get("artifacts") or []
            if entry.get("action") == "download"
        )

    offered = [rung for rung in rungs if rung.get("selectable")]
    for rung in offered:
        rung["included"] = policy == POLICY_ALL or (
            isinstance(policy, int) and int(rung["index"]) <= policy
        )
    included = [rung for rung in offered if rung.get("included")]

    download_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in list(recommendation.get("artifacts") or []) + [
        item for rung in included for item in rung.get("artifacts") or []
    ]:
        if entry.get("action") != "download":
            continue
        destination = (entry.get("download") or {}).get("destination")
        if not destination or destination in seen:
            continue
        seen.add(destination)
        download_entries.append(entry)

    selected = list(recommendation.get("artifacts") or []) + [
        item for rung in included for item in rung.get("artifacts") or []
    ]
    unfetchable = sorted(
        {str(entry.get("path")) for entry in selected if entry.get("action") == "unfetchable"}
    )
    if download_entries:
        status = PLAN_READY
    elif unfetchable:
        # Missing, and unfetchable: calling that "already installed" would be a lie.
        status = PLAN_BLOCKED
    else:
        status = PLAN_COMPLETE

    included_bytes = sum(int(rung.get("incremental_bytes") or 0) for rung in included)
    fallback = {
        "policy": policy,
        "lane": scan.lane,
        "context_rungs": context_rungs,
        "context_note": (
            f"{context_rungs} further chain steps shed context on the same weights and "
            "need no download"
            if context_rungs
            else ""
        ),
        "rungs": rungs,
        "offered_bytes": sum(int(rung.get("incremental_bytes") or 0) for rung in offered),
        "included_bytes": included_bytes,
        "included_rungs": [int(rung["index"]) for rung in included],
        "message": (
            "No fallback rungs are included: this is the recommendation only. "
            "Add --include-fallback 1 (or all) to fetch the chain a step further."
            if not included
            else "Included fallback rungs "
            + ", ".join(str(rung["index"]) for rung in included)
            + f" ({human_bytes(included_bytes)} more)."
        ),
    }
    return rungs, included, fallback, download_entries, selected, unfetchable, status


def _blocked_plan(
    *,
    base: Path,
    scan: HostScan,
    view: Mapping[str, Any] | None,
    tier_measured: bool,
    blocker: str,
    tier_entry: Mapping[str, Any] | None,
    policy: Any,
    rungs: list[dict[str, Any]] | None = None,
    context_rungs: int = 0,
) -> dict[str, Any]:
    if blocker:
        status = PLAN_BLOCKED
        message = blocker
    elif tier_entry is not None and tier_entry.get("status") == STATUS_ENGINE:
        status = PLAN_BLOCKED
        message = str(tier_entry.get("message") or "the engine cannot run on this machine.")
    else:
        status = PLAN_UNMEASURED
        message = (
            f"No measurements yet on {scan.tier_gb} GB hardware, and no fallback-chain "
            "rung both fits this machine and ships a launch recipe, so there is nothing "
            "honest to recommend or download yet."
        )
    return {
        "ok": False,
        "schema": SCHEMA,
        "root": str(base),
        "mode": "dry-run",
        "status": status,
        "host": scan.as_dict(),
        "tier": {
            "hardware_tier_gb": scan.tier_gb,
            "tier_source": scan.tier_source,
            "measured": tier_measured,
        },
        "cpu": dict((view or {}).get("cpu") or {}),
        "recommendation": None,
        "fallback": {
            "policy": policy,
            "lane": scan.lane,
            "rungs": list(rungs or []),
            "context_rungs": int(context_rungs),
            "offered_bytes": 0,
            "included_bytes": 0,
            "included_rungs": [],
            "message": "No chain rungs are offered because no model can be recommended here.",
        },
        "download": {"entries": [], "destinations": [], "families": [], "bytes": 0},
        "total": {
            "download_bytes": 0,
            "download_gib": 0.0,
            "library_bytes": 0,
            "artifacts_to_download": 0,
            "artifacts_present": 0,
            "unknown_size_files": [],
            "unfetchable": [],
            "rungs_included": [],
        },
        "message": message,
        "next": [],
    }


def _plan_message(
    *,
    scan: HostScan,
    recommendation: Mapping[str, Any],
    fallback: Mapping[str, Any],
    status: str,
    download_bytes: int,
    unfetchable: Iterable[str] = (),
) -> str:
    name = str(recommendation.get("main") or recommendation.get("main_family") or "the recommended model")
    context = int(recommendation.get("context") or 0)
    label = f" at {context // 1024}K" if context else ""
    prefix = ""
    if recommendation.get("evidence") != "measured-on-box":
        prefix = (
            f"No measurements yet on {scan.tier_gb} GB hardware, so this is the documented "
            "fallback-chain rung that fits, not a benchmarked winner: "
        )
    if status == PLAN_COMPLETE:
        size = int(recommendation.get("library_bytes") or 0)
        verified = f" ({human_bytes(size)} verified)" if size else ""
        body = (
            f"everything {name}{label} needs is already on this machine"
            f"{verified}; nothing will be downloaded."
        )
        return prefix + body[0].upper() + body[1:]
    if status == PLAN_BLOCKED and list(unfetchable):
        missing = list(unfetchable)
        return prefix + (
            f"{name}{label} cannot be downloaded automatically: "
            f"{len(missing)} file(s) it needs have no pinned manifest entry, so they "
            "must be placed in the model directory by hand."
        )
    rungs = [rung for rung in fallback.get("rungs") or [] if rung.get("included")]
    rung_phrase = (
        f" plus {len(rungs)} fallback {'rung' if len(rungs) == 1 else 'rungs'}" if rungs else ""
    )
    sentence = (
        f"download {human_bytes(download_bytes)} for {name}{label}{rung_phrase}; "
        f"the recommendation itself needs "
        f"{human_bytes(int(recommendation.get('library_bytes') or 0))}."
    )
    sentence = sentence[0].upper() + sentence[1:]
    placement = recommendation.get("placement") or {}
    if placement.get("label"):
        sentence += f" Placement: {placement['label']}."
    missing = list(unfetchable)
    if missing:
        sentence += (
            f" {len(missing)} file(s) have no pinned download and must be supplied by hand."
        )
    return prefix + sentence


def _next_steps(status: str, included: bool, policy: Any) -> list[str]:
    if status == PLAN_COMPLETE:
        return ["scripts/turbofit-run"]
    steps = ["scripts/turbofit-onboard --download"]
    if not included:
        steps.append("scripts/turbofit-onboard --download --include-fallback 1")
    elif policy == POLICY_ALL:
        steps.append("scripts/turbofit-onboard --download --include-fallback all")
    return steps


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _downloader() -> Any:
    """The one materializer: ``scripts/download-artifacts``.

    Imported lazily so planning (and every test) stays offline and free of the
    Hugging Face client.
    """
    global _DOWNLOADER
    if _DOWNLOADER is None:
        script = ROOT / "scripts/download-artifacts"
        loader = SourceFileLoader("turbofit_download_artifacts", str(script))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(loader.name, module)
        loader.exec_module(module)
        _DOWNLOADER = module
    return _DOWNLOADER


def default_model_root() -> Path:
    return Path(os.environ.get("TURBOFIT_MODEL_ROOT", "~/Models/storage/gguf")).expanduser()


def _download_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in plan.get("download", {}).get("entries") or []:
        source = entry.get("download") or {}
        if not source.get("repo_id") or not source.get("path"):
            continue
        rows.append(
            {
                "destination": source["destination"],
                "families": list(source.get("families") or []),
                "repo_id": source["repo_id"],
                "revision": source["revision"],
                "path": source["path"],
                "sha256": source.get("sha256") or "",
                "size_bytes": int(source.get("size_bytes") or 0),
            }
        )
    return rows


def _recheck(plan: Mapping[str, Any], model_root: Path) -> list[dict[str, Any]]:
    """Re-verify planned artifacts on disk after a fetch.

    A downloader reporting success is not evidence that the file is installed;
    the campaign's own size/stub check is.
    """
    results = []
    for entry in plan.get("download", {}).get("entries") or []:
        destination = (entry.get("download") or {}).get("destination")
        if not destination:
            continue
        target = model_root / destination
        expected = entry.get("full_bytes") if entry.get("bytes_basis") == "manifest-pinned" else None
        state = inspect_file(target, expected)
        results.append(
            {
                "destination": destination,
                "present": state.present,
                "size_bytes": state.size_bytes,
                "reason": state.reason,
            }
        )
    return results


def execute_plan(
    plan: Mapping[str, Any],
    *,
    model_root: str | Path | None = None,
    fetcher: Callable[[list[dict[str, Any]], Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fetch exactly the plan's rows, then re-verify them on disk."""
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"not an onboarding plan: {plan.get('schema')!r}")
    root = Path(model_root).expanduser() if model_root is not None else default_model_root()
    rows = _download_rows(plan)
    if not rows:
        return {
            "ok": bool(plan.get("ok")),
            "schema": RESULT_SCHEMA,
            "mode": "download",
            "status": plan.get("status"),
            "downloaded": 0,
            "already_present": 0,
            "families": {},
            "failed": {},
            "missing_bytes": 0,
            "verified": True,
            "model_root": str(root),
            "plan": plan,
            "message": str(plan.get("message") or "nothing to download."),
        }
    fetch = fetcher or (lambda items, destination: _downloader().download_families(items, root=destination))
    summary = fetch(rows, root)
    verification = _recheck(plan, root)
    unverified = [row for row in verification if not row["present"]]
    planned_bytes = sum(
        int(entry.get("download_bytes") or 0) for entry in plan.get("download", {}).get("entries") or []
    )
    failed = dict(summary.get("failed") or {})
    ok = not failed and not unverified
    name = str(
        (plan.get("recommendation") or {}).get("main")
        or (plan.get("recommendation") or {}).get("main_family")
        or "the recommended model"
    )
    if ok:
        message = (
            f"Fetched {human_bytes(planned_bytes)} and verified it on disk: {name} is ready to run."
        )
    else:
        message = (
            f"{len(failed)} family/families failed and {len(unverified)} artifact(s) are still "
            f"not complete on disk; nothing was silently marked installed."
        )
    return {
        "ok": ok,
        "schema": RESULT_SCHEMA,
        "mode": "download",
        "status": plan.get("status"),
        "downloaded": int(summary.get("artifacts_fetched") or 0),
        "already_present": int(
            summary.get("artifacts", 0) - summary.get("artifacts_fetched", 0)
        ),
        "families": dict(summary.get("families_detail") or {}),
        "failed": failed,
        "missing_bytes": int(summary.get("missing_bytes") or 0),
        "expected_bytes": planned_bytes,
        "verified": not unverified,
        "verification": verification,
        "model_root": str(root),
        "recommendation": plan.get("recommendation"),
        "message": message,
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# Human rendering
# ---------------------------------------------------------------------------


def render_plan(plan: Mapping[str, Any]) -> str:
    """The plan as a first-run user reads it, not as JSON."""
    host = plan.get("host") or {}
    lines = ["TurboFit first run — this machine"]
    topology = host.get("topology_key") or "unknown"
    if host.get("device_count"):
        lines.append(
            f"  host: {host.get('device_count')}x{float(host.get('per_device_gb') or 0):.0f} GB VRAM "
            f"({topology}, {host.get('memory_pool')}) · {host.get('system_ram_gb')} GB RAM · "
            f"{host.get('ram_gb')} GB available"
        )
    else:
        lines.append(
            f"  host: no accelerator detected, CPU inference · {host.get('system_ram_gb')} GB RAM"
        )
    tier = plan.get("tier") or {}
    lines.append(
        f"  tier: {tier.get('hardware_tier_gb')} GB · "
        + ("measured" if tier.get("measured") else "no measurements yet")
    )
    cpu = (plan.get("cpu") or {}).get("statement")
    if cpu:
        lines.append(f"  cpu: {cpu}")

    recommendation = plan.get("recommendation")
    if not recommendation:
        lines += ["", str(plan.get("message") or "Nothing can be recommended here.")]
        return "\n".join(lines)

    lines += ["", "recommendation"]
    context = int(recommendation.get("context") or 0)
    headline = str(recommendation.get("main") or recommendation.get("main_family"))
    if context:
        headline += f" at {context // 1024}K"
    speed = recommendation.get("throughput_tps")
    intelligence = recommendation.get("intelligence_score")
    scores = []
    if intelligence is not None:
        scores.append(f"intelligence {float(intelligence):.1f}")
    if speed is not None:
        scores.append(f"{float(speed):.1f} tok/s")
    if scores:
        headline += "  (" + ", ".join(scores) + ")"
    lines.append(f"  {headline}")
    placement = recommendation.get("placement") or {}
    if placement.get("label"):
        lines.append(f"  placement: {placement['label']}")
    engine = recommendation.get("engine") or {}
    if engine.get("name"):
        state = "installed" if engine.get("installed") else "binary not installed yet"
        lines.append(f"  engine: {engine.get('name')} ({state})")

    artifacts = recommendation.get("artifacts") or []
    download = [row for row in artifacts if row.get("action") == "download"]
    present = [row for row in artifacts if row.get("present")]
    lines.append(
        f"  download {human_bytes(int(recommendation.get('download_bytes') or 0))} "
        f"in {len(download)} file(s); {len(present)} already verified on disk"
    )
    for row in artifacts:
        size = human_bytes(row.get("full_bytes"))
        lines.append(f"    - {PurePosixPath(row['path']).name} — {size} — {row.get('state')}")

    fallback = plan.get("fallback") or {}
    rungs = fallback.get("rungs") or []
    if rungs:
        lines += ["", "optional fallback-chain rungs"]
        for rung in rungs:
            if not rung.get("selectable"):
                lines.append(
                    f"  {rung.get('chain_level')}. {rung.get('model')} — not offered: {rung.get('reason')}"
                )
                continue
            flag = "INCLUDED" if rung.get("included") else "not included"
            lines.append(
                f"  {rung.get('index')}. {rung.get('model')} at "
                f"{int(rung.get('context') or 0) // 1024}K — "
                f"{human_bytes(int(rung.get('incremental_bytes') or 0))} — "
                f"{rung.get('evidence')} — {flag}"
            )
    if fallback.get("context_note"):
        lines.append(f"  ({fallback['context_note']})")

    total = plan.get("total") or {}
    lines += [
        "",
        f"total download: {human_bytes(int(total.get('download_bytes') or 0))}"
        + (
            f" (plus {len(total.get('unknown_size_files') or [])} file(s) of unknown size)"
            if total.get("unknown_size_files")
            else ""
        ),
        f"status: {plan.get('status')}",
        str(plan.get("message") or ""),
    ]
    for step in plan.get("next") or []:
        lines.append(f"next: {step}")
    return "\n".join(line for line in lines if line is not None)


__all__ = [
    "FallbackPolicyError",
    "HostScan",
    "PLAN_BLOCKED",
    "PLAN_COMPLETE",
    "PLAN_READY",
    "PLAN_UNMEASURED",
    "POLICY_ALL",
    "POLICY_NONE",
    "RESULT_SCHEMA",
    "SCHEMA",
    "build_plan",
    "default_model_root",
    "execute_plan",
    "parse_fallback_policy",
    "render_plan",
    "scan_host",
]
