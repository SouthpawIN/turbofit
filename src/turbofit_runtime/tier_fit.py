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
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from .experts import plan_expert_offload

FOOTPRINTS_SCHEMA = "turbofit.model-footprints/v1"
AUXILIARY_ID = "maple-preview-tq2"
MIN_CONTEXT = 65_536
CONTEXT_SLUGS = {65_536: "64k", 131_072: "128k", 262_144: "262k", 1_048_576: "1m"}
KV_GIB = {65_536: 1, 131_072: 2, 262_144: 4, 1_048_576: 16}
TIERS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
GIB = 1024 ** 3

# --- RAM dimension -----------------------------------------------------------
# Placement used to be one-dimensional: a model either fit a VRAM band or it did
# not.  With expert offload (``--n-cpu-moe``) that is wrong for any host with
# enough RAM: the experts can live in system RAM while attention, KV and dense
# weights stay on the GPU.
#
# The committed artefacts cannot assume one machine's RAM, so they carry an
# explicit assumption — ``DEFAULT_RAM_GIB`` — and every generator accepts
# ``--ram-gb`` / honours ``TURBOFIT_RAM_GIB`` so another host gets its own
# answer.  128 GiB is the documented default because it is a common workstation
# ceiling and comfortably exercises offload at 48-96 GiB tiers; it is an
# *available* budget (the MemAvailable equivalent), not a total installed figure.
DEFAULT_RAM_GIB = 128
RAM_GIB_ENV = "TURBOFIT_RAM_GIB"
# Mirrors experts.plan_expert_offload's own reserves rather than re-deriving them.
RAM_RESERVE_BYTES = 2048 * 1024 * 1024


def ram_budget_bytes(ram_gb: float) -> int:
    """Bytes of system RAM offered to the engine for one launch."""
    if ram_gb <= 0:
        raise ValueError("ram_gb must be positive")
    return int(ram_gb * GIB)


def default_ram_gib(env: Mapping[str, str] | None = None) -> float:
    """RAM assumption: ``TURBOFIT_RAM_GIB`` when set, else the documented default."""
    source = env if env is not None else os.environ
    raw = (source.get(RAM_GIB_ENV) or "").strip()
    if not raw:
        return float(DEFAULT_RAM_GIB)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{RAM_GIB_ENV} must be a number of GiB, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{RAM_GIB_ENV} must be positive, got {raw!r}")
    return value
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


@dataclass(frozen=True)
class Placement:
    """Where one configuration's memory lives, at one tier's VRAM band.

    ``placement == "gpu"`` means every byte is on the card(s): the historical,
    fully-VRAM answer.  ``placement == "offload"`` means the model exceeds the
    band and is only placeable because a stated number of bytes of MoE experts
    can live in system RAM while the rest stays on the GPU.  The two are never
    collapsed: a caller that wants "runs entirely on GPUs" filters on
    ``placement``.
    """

    configuration_id: str
    main: str
    context: int
    placement: str
    total_bytes: int
    vram_bytes: int
    ram_bytes: int
    n_cpu_moe: int
    expert_layers: int
    expert_bytes: int
    evidence: str
    reason: str

    @property
    def ram_gib(self) -> float:
        return self.ram_bytes / GIB

    @property
    def vram_gib(self) -> float:
        return self.vram_bytes / GIB

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["vram_gib"] = round(self.vram_gib, 2)
        data["ram_gib"] = round(self.ram_gib, 2)
        data["total_gib"] = round(self.total_bytes / GIB, 2)
        data["expert_gib"] = round(self.expert_bytes / GIB, 2)
        return data


def placement_for(
    model_id: str,
    record: Mapping[str, Any],
    context: int,
    vram_gb: int,
    ram_bytes: int,
    *,
    measured: bool = True,
) -> Placement | None:
    """Place one catalog main at one context in one band, with system RAM.

    ``record`` is a ``model-footprints.json`` entry: ``weight_bytes`` always, and
    ``expert_bytes``/``expert_layers`` when the artifact's GGUF header has been
    measured.  The offload branch is not a second fit model: it calls
    ``experts.plan_expert_offload`` — the same arithmetic the launcher uses to
    emit ``--n-cpu-moe`` — with the tier's VRAM and the RAM budget as its inputs.

    Returns ``None`` when the model cannot be placed at this context, including
    when it is MoE but its expert share has not been measured (an unmeasured
    model must not be advertised as offloadable).
    """
    if context < MIN_CONTEXT:
        return None
    weight_bytes = int(record["weight_bytes"])
    total_bytes = weight_bytes + kv_bytes(context)
    configuration_id = f"{model_id}--{AUXILIARY_ID}--{CONTEXT_SLUGS[context]}"

    if fits(weight_bytes, context, vram_gb):
        return Placement(
            configuration_id=configuration_id, main=model_id, context=context,
            placement="gpu", total_bytes=total_bytes, vram_bytes=total_bytes,
            ram_bytes=0, n_cpu_moe=0, expert_layers=0, expert_bytes=0,
            evidence="weights-and-kv-fit-vram",
            reason=f"{weight_bytes / GIB:.1f} GiB of weights + "
                   f"{kv_bytes(context) / GIB:.0f} GiB KV fits the {vram_gb} GB band",
        )

    expert_bytes = record.get("expert_bytes")
    expert_layers = record.get("expert_layers")
    if not measured or not expert_bytes or not expert_layers:
        return None
    plan = plan_expert_offload(
        total_bytes=total_bytes,
        expert_bytes=int(expert_bytes),
        expert_layers=int(expert_layers),
        vram_bytes=tier_budget_bytes(vram_gb),
        ram_bytes=ram_bytes,
    )
    if not plan.feasible or plan.n_cpu_moe <= 0:
        return None
    return Placement(
        configuration_id=configuration_id, main=model_id, context=context,
        placement="offload", total_bytes=total_bytes, vram_bytes=plan.vram_bytes,
        ram_bytes=plan.ram_bytes, n_cpu_moe=plan.n_cpu_moe,
        expert_layers=plan.expert_layers, expert_bytes=int(expert_bytes),
        evidence="measured-expert-footprint",
        reason=plan.reason,
    )


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
    files, complete = artifact_files_for_family(artifacts, siblings)
    return sum(size for _repo, _path, size in files), complete


def artifact_files_for_family(
    artifacts: Iterable[Mapping[str, Any]],
    siblings: Mapping[str, Mapping[str, int]],
) -> tuple[list[tuple[str, str, int]], bool]:
    """Every pinned file that makes up one catalog main, with its real size.

    This is the single shard-enumeration path: the footprint total, the expert
    measurement and any audit all see the same file set.
    """
    manifest_sizes = {
        (str(item["repo_id"]), str(item["path"])): int(item["size_bytes"])
        for item in artifacts
    }
    files: dict[tuple[str, str], int] = {}
    complete = True
    shard_sets: dict[tuple[str, str, str, int, str], Mapping[str, Any]] = {}
    for artifact in artifacts:
        repo_id = str(artifact["repo_id"])
        path = str(artifact["path"])
        shard = parse_shard(path)
        if shard is None:
            files[(repo_id, path)] = int(artifact["size_bytes"])
            continue
        prefix, _index, count, extension = shard
        parent = str(PurePosixPath(path).parent)
        shard_sets[(repo_id, parent, prefix, count, extension)] = artifact
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
            files[(repo_id, path)] = int(size)
    return [(repo, path, size) for (repo, path), size in sorted(files.items())], complete


def build_footprints(
    catalog: Mapping[str, Any],
    manifest: Mapping[str, Any],
    siblings_provider: Callable[[str], Mapping[str, int]] | None = None,
    generated_at: str | None = None,
    expert_provider: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Build the per-main weight table from the artifact manifest.

    ``siblings_provider(repo_id)`` returns the complete pinned listing for a
    repo. Without it the files listed in the manifest are summed as-is, which
    understates a repo whose manifest only tracks the first shard.

    ``expert_provider(model)`` returns a measured expert record
    (``expert_bytes``, ``expert_layers``, optional ``source``/``measured_at``)
    for one catalog main, or None when its GGUF header has not been measured.
    The record is attached to the model entry — placing a model as an offload
    option without it would be a claim, not a measurement.
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
        record: dict[str, Any] = {
            "weight_bytes": weight_bytes,
            "shard_set_complete": complete,
            "repos": sorted(repos),
            "moe": bool(model.get("moe")),
        }
        expert = expert_provider(model) if expert_provider is not None else None
        if expert is not None:
            expert_bytes = int(expert["expert_bytes"])
            expert_layers = int(expert["expert_layers"])
            if expert_bytes <= 0 or expert_layers <= 0:
                raise ValueError(f"measured zero expert footprint for catalog main {model['id']}")
            if expert_bytes > weight_bytes:
                raise ValueError(
                    f"measured experts exceed total weights for catalog main {model['id']}: "
                    f"{expert_bytes} > {weight_bytes}"
                )
            record["expert_bytes"] = expert_bytes
            record["expert_layers"] = expert_layers
            measurement: dict[str, Any] = {
                "method": "gguf-header",
                "source": str(expert.get("source") or ""),
            }
            if expert.get("measured_at"):
                measurement["measured_at"] = str(expert["measured_at"])
            if expert.get("total_bytes") is not None:
                measurement["measured_total_bytes"] = int(expert["total_bytes"])
                # A measured shard set that disagrees with the manifest listing
                # means the artifact changed shape; say so instead of averaging.
                measurement["matches_weight_bytes"] = int(expert["total_bytes"]) == weight_bytes
            record["expert_measurement"] = measurement
        models[str(model["id"])] = record
    return {
        "schema": FOOTPRINTS_SCHEMA,
        "generated_at": generated_at or "",
        "source": "references/artifact-manifest.json at pinned revisions; shard sets completed from the pinned repo listing",
        "expert_source": "GGUF tensor tables read from the pinned revision (HTTP Range headers or local files); models without expert_bytes are MoE but unmeasured",
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


def _configuration_ids(configurations: Mapping[str, Any]) -> set[str]:
    rows = configurations.get("rows")
    if not isinstance(rows, list):
        raise ValueError("configuration matrix must contain rows")
    return {str(row["id"]) for row in rows if isinstance(row, Mapping)}


def select_placements(
    catalog: Mapping[str, Any],
    configurations: Mapping[str, Any],
    footprints: Mapping[str, Any],
    vram_gb: int,
    ram_bytes: int,
    *,
    measured: bool = True,
) -> list[Placement]:
    """Every placeable catalog main at Maple aux, full-GPU and offloaded.

    Ordering is unchanged from the VRAM-only ladder: mains by the largest
    context they can hold, then by weight, then by id; contexts ascending.
    """
    configuration_ids = _configuration_ids(configurations)
    weights = footprints.get("models") or {}
    grouped: list[tuple[int, int, str, list[Placement]]] = []
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []):
            continue
        model_id = str(model["id"])
        record = weights.get(model_id)
        if record is None:
            raise ValueError(f"missing weight footprint for catalog main {model_id}")
        supported = model.get("supported_contexts")
        contexts = list(supported) if supported else [65_536, 131_072, 262_144, 1_048_576]
        placements = []
        for context in sorted(int(item) for item in contexts):
            if f"{model_id}--{AUXILIARY_ID}--{CONTEXT_SLUGS[context]}" not in configuration_ids:
                continue
            placement = placement_for(
                model_id, record, context, vram_gb, ram_bytes, measured=measured,
            )
            if placement is not None:
                placements.append(placement)
        if placements:
            # Keep the historical full-GPU ordering: a main is ranked by the
            # largest context it runs *entirely on the GPU* (offload-only mains
            # sort after those, by their largest offload context).
            gpu_contexts = [item.context for item in placements if item.placement == "gpu"]
            offload_contexts = [item.context for item in placements if item.placement == "offload"]
            rank_context = max(gpu_contexts or offload_contexts)
            placements.sort(key=lambda item: (item.placement != "gpu", item.context))
            grouped.append((
                -rank_context,
                -int(record["weight_bytes"]),
                model_id,
                placements,
            ))
    grouped.sort(key=lambda item: (item[0], item[1], item[2]))
    return [placement for _c, _w, _id, items in grouped for placement in items]


def select_candidates(
    catalog: Mapping[str, Any],
    configurations: Mapping[str, Any],
    footprints: Mapping[str, Any],
    vram_gb: int,
    ram_bytes: int | None = None,
) -> list[str]:
    """Catalog mains that run *entirely on the GPU* at this band.

    RAM is accepted for signature compatibility but never promotes a model into
    this list: an offloaded configuration is a different placement and is
    returned by :func:`select_offload_candidates`.
    """
    placements = select_placements(
        catalog, configurations, footprints, vram_gb,
        ram_budget_bytes(default_ram_gib()) if ram_bytes is None else ram_bytes,
    )
    return [item.configuration_id for item in placements if item.placement == "gpu"]


def select_offload_candidates(
    catalog: Mapping[str, Any],
    configurations: Mapping[str, Any],
    footprints: Mapping[str, Any],
    vram_gb: int,
    ram_bytes: int,
) -> list[dict[str, Any]]:
    """Catalog mains that only fit this band with experts in system RAM."""
    placements = select_placements(
        catalog, configurations, footprints, vram_gb, ram_bytes,
    )
    return [item.as_dict() for item in placements if item.placement == "offload"]


def unmeasured_moe_mains(
    catalog: Mapping[str, Any], footprints: Mapping[str, Any]
) -> list[str]:
    """MoE catalog mains whose expert share has not been measured.

    They are not placeable as offload options, and saying so is cheaper than
    silently omitting them.
    """
    weights = footprints.get("models") or {}
    unmeasured = []
    for model in catalog.get("models") or []:
        if "main" not in (model.get("roles") or []) or not model.get("moe"):
            continue
        record = weights.get(str(model["id"])) or {}
        if not record.get("expert_bytes") or not record.get("expert_layers"):
            unmeasured.append(str(model["id"]))
    return sorted(unmeasured)
