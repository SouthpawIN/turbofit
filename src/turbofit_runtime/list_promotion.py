"""Promote exact-tier benchmark winners into the TurboFit List tournament record."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


def retain_or_replace_winner(
    existing: Mapping[str, Any] | None,
    selected: Mapping[str, Any] | None,
    candidates: Sequence[str],
) -> dict[str, str] | None:
    """Never erase committed evidence merely because local scratch state is absent."""
    if selected is not None:
        return {key: str(selected[key]) for key in ("configuration", "evidence", "hardware_fingerprint")}
    if existing is not None and existing.get("configuration") in candidates:
        return {key: str(existing[key]) for key in ("configuration", "evidence", "hardware_fingerprint")}
    return None


def select_tier_winner(
    candidates: Sequence[str],
    physical: Mapping[str, Mapping[str, Any]],
    intelligence: Mapping[str, Mapping[str, Any]],
    *,
    hardware_tier_gb: int,
) -> dict[str, str] | None:
    eligible: list[tuple[tuple[float, float, float, str], str, Mapping[str, Any]]] = []
    for identifier in candidates:
        runtime = physical.get(identifier)
        score = intelligence.get(identifier)
        if not isinstance(runtime, Mapping) or not isinstance(score, Mapping):
            continue
        evidence = str(runtime.get("evidence") or "")
        fingerprint = str(runtime.get("hardware_fingerprint") or "")
        if (
            runtime.get("status") != "success"
            or runtime.get("current_recipe") is not True
            or _SHA.fullmatch(evidence) is None
            or not fingerprint
            or score.get("hardware_tier_gb") != hardware_tier_gb
        ):
            continue
        try:
            balanced = float(score["balanced_score"])
            intelligence_value = float(score["intelligence_score"])
            throughput = float(score["throughput_tps"])
        except (KeyError, TypeError, ValueError):
            continue
        if balanced <= 0 or intelligence_value <= 0 or throughput <= 0:
            continue
        eligible.append(((balanced, intelligence_value, throughput, identifier), identifier, runtime))
    if not eligible:
        return None
    _, identifier, runtime = max(eligible, key=lambda item: item[0])
    return {
        "configuration": identifier,
        "evidence": str(runtime["evidence"]),
        "hardware_fingerprint": str(runtime["hardware_fingerprint"]),
    }
