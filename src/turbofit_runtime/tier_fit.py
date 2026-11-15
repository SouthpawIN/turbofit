"""Tier fit arithmetic for the hardware-tier tournaments.

The tournaments are the ranking surface of the intelligence campaign: every
catalog main model that can actually hold a 64K ctx in a memory band is listed
against one fixed auxiliary (Maple TQ2_0), so ``select_turbofit_list_candidates``
can measure intelligence and speed for all of them and pick one winner.

Fit is derived from the repo's own files, not from hand-picked winners:

* ``references/model-footprints.json`` — per-catalog-main weight bytes, built
  from ``references/artifact-manifest.json`` (pinned revision + SHA) and the
  pinned shard set of each repo.
* ``references/fit-ladder.md`` — "Required: model_size + KV_cache <= tier_size"
  with ``KV_cache ~ 2-4 GB at 262K``.  We take the documented upper bound and
  scale it linearly: 1 GiB at 64K, 2 GiB at 128K, 4 GiB at 256K, 16 GiB at 1M.
* ``references/recommend.py`` — the documented 1% VRAM reporting tolerance
  (16 MiB floor at 8 GB) absorbs the reporting delta at every tier boundary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

FOOTPRINTS_SCHEMA = "turbofit.model-footprints/v1"
AUXILIARY_ID = "maple-preview-tq2"
MIN_CONTEXT = 65_536
CONTEXT_SLUGS = {65_536: "64k", 131_072: "128k", 262_144: "262k", 1_048_576: "1m"}
KV_GIB = {65_536: 1, 131_072: 2, 262_144: 4, 1_048_576: 16}
TIERS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
GIB = 1024 ** 3
_SHARD = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.(?P<ext>gguf|bin)$")
# Mirrors recommend.DEDICATED_VRAM_REPORTING_TOLERANCE_* without importing the
# whole recommendation stack into this pure-arithmetic module.
_REPORTING_TOLERANCE_MB = 16
_REPORTING_TOLERANCE_FRACTION = 0.01


def tier_budget_bytes(vram_gb: int) -> int:
    """Bytes of a band including the documented VRAM reporting tolerance."""
    tolerance_mb = max(
        _REPORTING_TOLERANCE_MB,
        int(round(vram_gb * 1024 * _REPORTING_TOLERANCE_FRACTION)),
    ) if vram_gb > 8 else _REPORTING_TOLERANCE_MB
    return vram_gb * GIB + tolerance_mb * 1024 * 1024


def kv_bytes(context: int) -> int:
    if context not in KV_GIB:
        raise ValueError(f"unsupported tournament context: {context}")
    return KV_GIB[context] * GIB


def fits(weight_bytes: int, context: int, vram_gb: int) -> bool:
    """True when weights + KV cache fit the band at this context."""
    if context < MIN_CONTEXT:
        return False
    return weight_bytes + kv_bytes(context) <= tier_budget_bytes(vram_gb)


def parse_shard(path: str) -> tuple[str, int, int, str] | None:
    """Return ``(prefix, index, count, extension)`` for a sharded filename."""
    match = _SHARD.match(PurePosixPath(path).name)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("index")), int(match.group("count")), match.group("ext")


def _siblings_from_manifest(artifacts: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Group manifest rows by repo, keyed by their own path -> size."""
    by_repo: dict[str, dict[str, int]] = {}
    for artifact in artifacts:
        by_repo.setdefault(str(artifact["repo_id"]), {})[str(artifact["path"])] = int(artifact["size_bytes"])
    return by_repo


def weight_bytes_for_family(
    artifacts: list[Mapping[str, Any]],
    siblings: Mapping[str, Mapping[str, int]],
) -> tuple[int, bool]:
    """Total weight bytes for one catalog main.

    ``siblings`` maps repo_id -> {path: size} for the complete pinned listing.
    Manifest rows are authoritative when present; missing shards are filled from
    the listing. The boolean is False when a declared shard is absent from both,
    which is recorded so the footprint file can be audited.
    """
    total = 0
    complete = True
    manifest_sizes = {
        (str(item["repo_id"]), str(item["path"])): int(item["size_bytes"])
        for item in artifacts
    }
    # A family can list several shards of the same set (or only the first).
    # Group them so each set is summed exactly once.
    shard_sets: dict[tuple[str, str, str, int, str], Mapping[str, Any]] = {}
    for artifact in artifacts:
        shard = parse_shard(str(artifact["path"]))
        if shard is None:
            total += int(artifact["size_bytes"])
            continue
        prefix, _index, count, extension = shard
        parent = str(PurePosixPath(str(artifact["path"])).parent)
        shard_sets[(str(artifact["repo_id"]), parent, prefix, count, extension)] = artifact
    for (repo_id, parent, prefix, count, extension) in shard_sets:
        listing = siblings.get(repo_id) or {}
        for index in range(1, count + 1):
            name = f"{prefix}-{index:05d}-of-{count:05d}.{extension}"
            path = str(PurePosixPath(parent) / name) if parent != "." else name
            size = listing.get(path)
            if size is None:
                size = manifest_sizes.get((repo_id, path))
            if size is None:
                complete = False
                continue
            total += int(size)
    return total, complete


def build_footprints(
    catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
    siblings_provider: Callable[[str], Mapping[str, int]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the per-main weight table from the artifact manifest.

    ``siblings_provider(repo_id)`` returns the complete pinned listing for a
    repo. Without it the files listed in the manifest are summed as-is, which
    understates a repo whose manifest only tracks the first shard.
    """
    artifacts = [item for item in manifest.get("artifacts") or [] if isinstance(item, Mapping)]
    manifest_siblings = _siblings_from_manifest(artifacts)
    cache: dict[str, Mapping[str, int]] = {}

    def listing(repo_id: str) -> Mapping[str, int]:
        if repo_id in manifest_siblings and (siblings_provider is None):
            return manifest_siblings[repo_id]
        if repo_id not in cache:
            if siblings_provider is not None:
                cache[repo_id] = siblings_provider(repo_id)
            else:
                cache[repo_id] = manifest_siblings.get(repo_id, {})
        return cache[repo_id]

    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in artifacts:
        for family in artifact.get("families") or []:
            by_family.setdefault(str(family), []).append(artifact)

    models: dict[str, dict[str, Any]] = {}
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []):
            continue
        artifacts_for_model = by_family.get(str(model["id"])) or []
        if not artifacts_for_model:
            raise ValueError(f"no artifact manifest rows for catalog main {model['id']}")
        repos = {str(item["repo_id"]) for item in artifacts_for_model}
        listing_map = {repo: listing(repo) for repo in repos}
        weight_bytes, complete = weight_bytes_for_family(artifacts_for_model, listing_map)
        if weight_bytes <= 0:
            raise ValueError(f"computed zero weight bytes for catalog main {model['id']}")
        models[str(model["id"])] = {
            "weight_bytes": weight_bytes,
            "shard_set_complete": complete,
            "repos": sorted(repos),
        }
    return {
        "schema": FOOTPRINTS_SCHEMA,
        "generated_at": generated_at or "",
        "source": "references/artifact-manifest.json at pinned revisions; shard sets completed from the pinned repo listing",
        "models": models,
    }


def load_footprints(root: str | Path) -> dict[str, Any]:
    payload = json.loads((Path(root) / "references/model-footprints.json").read_text(encoding="utf-8"))
    if payload.get("schema") != FOOTPRINTS_SCHEMA:
        raise ValueError("invalid model footprints root")
    return payload


def fitting_contexts(
    model: Mapping[str, Any],
    weight_bytes: int,
    vram_gb: int,
    configuration_ids: set[str],
) -> list[int]:
    """Contexts this main can hold in the band, that exist in the matrix."""
    supported = model.get("supported_contexts")
    contexts = list(supported) if supported else [65_536, 131_072, 262_144, 1_048_576]
    return [
        int(context)
        for context in contexts
        if fits(weight_bytes, int(context), vram_gb)
        and f"{model['id']}--{AUXILIARY_ID}--{CONTEXT_SLUGS[int(context)]}" in configuration_ids
    ]


def select_candidates(
    catalog: Mapping[str, Any],
    configurations: Mapping[str, Any],
    footprints: Mapping[str, Any],
    vram_gb: int,
) -> list[str]:
    """Every fitting catalog main at Maple aux, ordered for the campaign queue."""
    rows = configurations.get("rows")
    if not isinstance(rows, list):
        raise ValueError("configuration matrix must contain rows")
    configuration_ids = {str(row["id"]) for row in rows if isinstance(row, Mapping)}
    weights = footprints.get("models") or {}
    mains: list[tuple[str, int, list[int]]] = []
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []):
            continue
        record = weights.get(str(model["id"]))
        if record is None:
            raise ValueError(f"missing weight footprint for catalog main {model['id']}")
        contexts = fitting_contexts(model, int(record["weight_bytes"]), vram_gb, configuration_ids)
        if contexts:
            mains.append((str(model["id"]), int(record["weight_bytes"]), contexts))
    mains.sort(key=lambda item: (-max(item[2]), -item[1], item[0]))
    ordered: list[str] = []
    for model_id, _weight, contexts in mains:
        for context in sorted(contexts):
            ordered.append(f"{model_id}--{AUXILIARY_ID}--{CONTEXT_SLUGS[context]}")
    return ordered
