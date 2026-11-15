"""Strict hardware-tier tournament plans for evidence-backed promotion.

A tier has two disjoint candidate sets:

* ``candidates`` — configurations that run entirely on the tier's GPUs;
* ``offload_candidates`` — configurations that exceed the band and only run with
  a stated number of expert bytes in system RAM (``--n-cpu-moe``).

They are separate lists on purpose: a reader must be able to tell "fits on the
card" from "runs with 30 GiB of experts in RAM" without parsing a reason string.

Candidacy is also the ranking surface of the intelligence campaign, so every
candidate — on the card or offloaded — is subject to the quantization policy in
:mod:`turbofit_runtime.quant_policy`:

* at or below the Q4 class -> ordinary candidate, no note;
* Q8/full precision -> candidate only when it fits the band (fully on the GPU,
  or marked accuracy-tier), with the reason recorded in ``quant_exceptions``;
* above Q4 (Q5/Q6) -> not a candidate unless the catalog entry carries a
  recorded ``quantization_justification``; otherwise the family is listed in
  ``quant_excluded`` with the reason, so it disappears visibly rather than
  silently.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .quant_policy import (
    ALLOWED_EXCEPTION,
    EXCLUDED_BY_DEFAULT,
    REASON_JUSTIFIED,
    classify_entry,
    entry_quantization,
)

SCHEMA = "turbofit.hardware-tier-tournaments/v2"
TIERS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)
EVIDENCE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OFFLOAD_FIELDS = {
    "configuration_id", "placement", "total_gib", "vram_gib", "ram_gib",
    "n_cpu_moe", "expert_layers", "evidence", "reason",
}
RAM_ASSUMPTION_FIELDS = {"gib", "source", "override", "reserve_gib"}
TIER_FIELDS = {
    "id",
    "vram_gb",
    "physical_evidence_required",
    "candidates",
    "offload_candidates",
    "quant_exceptions",
    "quant_excluded",
    "winner",
}
EXCEPTION_FIELDS = {"main", "quantization", "quant_class", "quant_token", "reason"}
EXCLUSION_FIELDS = {
    "main", "family", "quantization", "quant_class", "quant_token", "reason", "contexts",
}


def load_tournaments(
    path: str | Path,
    configurations: Mapping[str, Any],
    catalog: Any = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_tournaments(payload, configurations, catalog)
    return payload


def _validate_offload(entry: Mapping[str, Any], tier_id: str) -> str:
    if not isinstance(entry, Mapping) or set(entry) != OFFLOAD_FIELDS:
        raise ValueError(f"invalid offload candidate fields: {tier_id}")
    identifier = str(entry["configuration_id"])
    if entry["placement"] != "offload":
        raise ValueError(f"offload candidate must be labelled offload: {tier_id}")
    if int(entry["n_cpu_moe"]) < 1:
        raise ValueError(f"offload candidate must move at least one expert layer: {tier_id}")
    if float(entry["ram_gib"]) <= 0 or float(entry["vram_gib"]) <= 0:
        raise ValueError(f"offload candidate must state both RAM and VRAM bytes: {tier_id}")
    if not str(entry["evidence"]).strip() or not str(entry["reason"]).strip():
        raise ValueError(f"offload candidate must carry evidence and a reason: {tier_id}")
    return identifier


def _main_by_configuration(configurations: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["id"]): str(row["main"])
        for row in configurations.get("rows", [])
        if isinstance(row, Mapping) and "id" in row and "main" in row
    }


def _models_by_id(catalog: Any) -> dict[str, Any]:
    if isinstance(catalog, Mapping):
        return {str(item["id"]): item for item in catalog.get("models") or []}
    return {str(item.id): item for item in getattr(catalog, "models", ())}


def _entry_field(entry: Any, name: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(name)
    return getattr(entry, name, None)


def validate_tournaments(
    payload: Mapping[str, Any],
    configurations: Mapping[str, Any],
    catalog: Any = None,
) -> None:
    if set(payload) != {"schema", "ranking", "ram_assumption", "tiers"} or payload.get("schema") != SCHEMA:
        raise ValueError("invalid hardware tier tournament root")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking or len(ranking) != len(set(ranking)):
        raise ValueError("hardware tier ranking must be a unique non-empty list")
    assumption = payload.get("ram_assumption")
    if not isinstance(assumption, Mapping) or set(assumption) != RAM_ASSUMPTION_FIELDS:
        raise ValueError("invalid ram_assumption record")
    if float(assumption["gib"]) <= 0 or float(assumption["reserve_gib"]) < 0:
        raise ValueError("ram_assumption must state a positive budget and a non-negative reserve")
    if not str(assumption["source"]).strip() or not str(assumption["override"]).strip():
        raise ValueError("ram_assumption must document its default and override")
    configuration_ids = {item["id"] for item in configurations.get("rows", [])}
    main_by_configuration = _main_by_configuration(configurations)
    models = _models_by_id(catalog) if catalog is not None else {}
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or tuple(item.get("vram_gb") for item in tiers) != TIERS:
        raise ValueError("hardware tiers must be 8/12/16/24/32/48/64/96/128/192/256/384 GB in order")
    for item in tiers:
        expected_id = f"hardware-{item['vram_gb']}gb"
        if set(item) != TIER_FIELDS:
            raise ValueError(f"invalid tier fields: {expected_id}")
        if item["id"] != expected_id or item["physical_evidence_required"] is not True:
            raise ValueError(f"invalid tier identity or evidence policy: {expected_id}")
        candidates = item["candidates"]
        if not isinstance(candidates, list) or not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"tier candidates must be unique and non-empty: {expected_id}")
        missing = sorted(set(candidates) - configuration_ids)
        if missing:
            raise ValueError(f"unknown tier candidates for {expected_id}: {missing}")
        offload = item["offload_candidates"]
        if not isinstance(offload, list):
            raise ValueError(f"tier offload_candidates must be a list: {expected_id}")
        offload_ids = [_validate_offload(entry, expected_id) for entry in offload]
        if len(offload_ids) != len(set(offload_ids)):
            raise ValueError(f"tier offload candidates must be unique: {expected_id}")
        unknown = sorted(set(offload_ids) - configuration_ids)
        if unknown:
            raise ValueError(f"unknown tier offload candidates for {expected_id}: {unknown}")
        overlap = sorted(set(offload_ids) & set(candidates))
        if overlap:
            raise ValueError(f"a configuration cannot be both full-GPU and offloaded: {overlap}")
        exceptions = item["quant_exceptions"]
        excluded = item["quant_excluded"]
        if not isinstance(exceptions, list) or not isinstance(excluded, list):
            raise ValueError(f"quant policy audit must be lists: {expected_id}")
        exception_mains: set[str] = set()
        for record in exceptions:
            if not isinstance(record, Mapping) or set(record) != EXCEPTION_FIELDS:
                raise ValueError(f"invalid quant exception record: {expected_id}")
            if record["quant_class"] not in {ALLOWED_EXCEPTION, EXCLUDED_BY_DEFAULT}:
                raise ValueError(f"invalid quant exception class: {expected_id}")
            if record["quant_class"] == EXCLUDED_BY_DEFAULT and not str(
                record["reason"]
            ).startswith(f"{REASON_JUSTIFIED}:"):
                raise ValueError(
                    f"an above-ceiling candidate in quant_exceptions must carry a "
                    f"recorded justification: {expected_id}"
                )
            if not str(record["reason"]).strip():
                raise ValueError(f"quant exception reason is required: {expected_id}")
            exception_mains.add(str(record["main"]))
        candidate_mains = {main_by_configuration.get(candidate) for candidate in candidates}
        offered_mains = candidate_mains | {
            main_by_configuration.get(identifier) for identifier in offload_ids
        }
        for record in excluded:
            if not isinstance(record, Mapping) or set(record) != EXCLUSION_FIELDS:
                raise ValueError(f"invalid quant exclusion record: {expected_id}")
            if record["quant_class"] not in {ALLOWED_EXCEPTION, EXCLUDED_BY_DEFAULT}:
                raise ValueError(f"quant exclusions must be held out by policy: {expected_id}")
            if not str(record["reason"]).strip():
                raise ValueError(f"quant exclusion reason is required: {expected_id}")
            if record["main"] in offered_mains:
                raise ValueError(
                    f"excluded quant {record['quantization']} is still a candidate: {expected_id}"
                )
        if models:
            # A candidate above the ceiling must carry a catalog justification and
            # must appear in this tier's exception audit, so the reason is in the file.
            for identifier in list(candidates) + list(offload_ids):
                main_id = main_by_configuration.get(identifier, str(identifier).split("--", 1)[0])
                entry = models.get(main_id)
                if entry is None:
                    continue
                verdict = classify_entry(entry)
                if verdict.is_excluded and not str(
                    _entry_field(entry, "quantization_justification") or ""
                ).strip():
                    raise ValueError(
                        f"tier {expected_id} offers excluded quant {entry_quantization(entry)} "
                        f"without a recorded justification: {main_id}"
                    )
                if not verdict.is_allowed and main_id not in exception_mains:
                    raise ValueError(
                        f"tier {expected_id} offers {entry_quantization(entry)} without a quant "
                        f"exception reason: {main_id}"
                    )
        winner = item["winner"]
        if winner is None:
            continue
        if not isinstance(winner, Mapping) or set(winner) != {"configuration", "evidence", "hardware_fingerprint"}:
            raise ValueError(f"invalid winner record: {expected_id}")
        if winner["configuration"] not in set(candidates) | set(offload_ids):
            raise ValueError(f"winner is not a tier candidate: {expected_id}")
        if not EVIDENCE_RE.fullmatch(str(winner["evidence"])):
            raise ValueError(f"winner evidence must be sha256-bound: {expected_id}")
        if not str(winner["hardware_fingerprint"]).strip():
            raise ValueError(f"winner hardware fingerprint is required: {expected_id}")


def candidate_ids_for_tier(payload: Mapping[str, Any], hardware_tier_gb: int) -> tuple[str, ...]:
    """Full-GPU candidates only; offload options are a different placement."""
    tier = next((item for item in payload["tiers"] if item["vram_gb"] == hardware_tier_gb), None)
    if tier is None:
        raise ValueError(f"unknown hardware tier: {hardware_tier_gb}")
    return tuple(str(item) for item in tier["candidates"])


def offload_candidates_for_tier(payload: Mapping[str, Any], hardware_tier_gb: int) -> tuple[dict[str, Any], ...]:
    tier = next((item for item in payload["tiers"] if item["vram_gb"] == hardware_tier_gb), None)
    if tier is None:
        raise ValueError(f"unknown hardware tier: {hardware_tier_gb}")
    return tuple(dict(item) for item in tier["offload_candidates"])


def candidate_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for tier in payload["tiers"]:
        for candidate in tier["candidates"]:
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
    return tuple(result)


def offload_candidate_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for tier in payload["tiers"]:
        for entry in tier["offload_candidates"]:
            identifier = str(entry["configuration_id"])
            if identifier not in seen:
                result.append(identifier)
                seen.add(identifier)
    return tuple(result)


def excluded_mains(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Tier id -> mains held out by the quant ceiling, for reporting."""
    return {
        str(tier["id"]): [str(record["main"]) for record in tier["quant_excluded"]]
        for tier in payload["tiers"]
    }
