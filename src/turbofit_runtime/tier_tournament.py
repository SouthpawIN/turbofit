"""Strict hardware-tier tournament plans for evidence-backed promotion."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "turbofit.hardware-tier-tournaments/v1"
TIERS = (8, 16, 24, 48, 64, 96, 200, 300)
EVIDENCE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def load_tournaments(
    path: str | Path,
    configurations: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_tournaments(payload, configurations)
    return payload


def validate_tournaments(payload: Mapping[str, Any], configurations: Mapping[str, Any]) -> None:
    if set(payload) != {"schema", "ranking", "tiers"} or payload.get("schema") != SCHEMA:
        raise ValueError("invalid hardware tier tournament root")
    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking or len(ranking) != len(set(ranking)):
        raise ValueError("hardware tier ranking must be a unique non-empty list")
    configuration_ids = {item["id"] for item in configurations.get("rows", [])}
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or tuple(item.get("vram_gb") for item in tiers) != TIERS:
        raise ValueError("hardware tiers must be 8/16/24/48/64/96/200/300 GB in order")
    for item in tiers:
        expected_id = f"hardware-{item['vram_gb']}gb"
        if set(item) != {"id", "vram_gb", "physical_evidence_required", "candidates", "winner"}:
            raise ValueError(f"invalid tier fields: {expected_id}")
        if item["id"] != expected_id or item["physical_evidence_required"] is not True:
            raise ValueError(f"invalid tier identity or evidence policy: {expected_id}")
        candidates = item["candidates"]
        if not isinstance(candidates, list) or not candidates or len(candidates) != len(set(candidates)):
            raise ValueError(f"tier candidates must be unique and non-empty: {expected_id}")
        missing = sorted(set(candidates) - configuration_ids)
        if missing:
            raise ValueError(f"unknown tier candidates for {expected_id}: {missing}")
        winner = item["winner"]
        if winner is None:
            continue
        if not isinstance(winner, Mapping) or set(winner) != {"configuration", "evidence", "hardware_fingerprint"}:
            raise ValueError(f"invalid winner record: {expected_id}")
        if winner["configuration"] not in candidates:
            raise ValueError(f"winner is not a tier candidate: {expected_id}")
        if not EVIDENCE_RE.fullmatch(str(winner["evidence"])):
            raise ValueError(f"winner evidence must be sha256-bound: {expected_id}")
        if not str(winner["hardware_fingerprint"]).strip():
            raise ValueError(f"winner hardware fingerprint is required: {expected_id}")


def candidate_ids_for_tier(payload: Mapping[str, Any], hardware_tier_gb: int) -> tuple[str, ...]:
    tier = next((item for item in payload["tiers"] if item["vram_gb"] == hardware_tier_gb), None)
    if tier is None:
        raise ValueError(f"unknown hardware tier: {hardware_tier_gb}")
    return tuple(str(item) for item in tier["candidates"])


def candidate_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for tier in payload["tiers"]:
        for candidate in tier["candidates"]:
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
    return tuple(result)
