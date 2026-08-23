"""Generate the evidence-only TurboFit List from hardware-tier winners."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

TIERS = (8, 16, 24, 48, 64, 96, 200, 300)


def _winner(tier: Mapping[str, Any]) -> dict[str, Any] | None:
    recommendations = tier.get("recommendations")
    if not isinstance(recommendations, Mapping):
        return None
    candidate = recommendations.get("balanced")
    if not isinstance(candidate, Mapping):
        return None
    fit = candidate.get("fit")
    if (
        tier.get("status") != "physically-validated"
        or not isinstance(fit, Mapping)
        or fit.get("physically_demonstrated") is not True
        or not fit.get("evidence")
        or candidate.get("intelligence_score") is None
        or candidate.get("measured_tps") is None
        or candidate.get("balanced_score") is None
    ):
        return None
    main = candidate.get("main") if isinstance(candidate.get("main"), Mapping) else {}
    auxiliary = candidate.get("auxiliary") if isinstance(candidate.get("auxiliary"), Mapping) else {}
    return {
        "configuration_id": candidate.get("configuration_id"),
        "main": main.get("name"),
        "auxiliary": auxiliary.get("name"),
        "context": candidate.get("context"),
        "intelligence_score": candidate.get("intelligence_score"),
        "measured_tps": candidate.get("measured_tps"),
        "balanced_score": candidate.get("balanced_score"),
        "evidence": fit.get("evidence"),
        "hardware_fingerprint": fit.get("hardware_fingerprint"),
    }


def build_turbofit_list(report: Mapping[str, Any]) -> dict[str, Any]:
    tiers = report.get("tiers")
    if not isinstance(tiers, list):
        raise ValueError("hardware tier report must contain tiers")
    by_capacity = {item.get("capacity_gb"): item for item in tiers if isinstance(item, Mapping)}
    levels = []
    for capacity in TIERS:
        tier = by_capacity.get(capacity)
        if tier is None:
            raise ValueError(f"hardware tier report lacks {capacity} GB")
        winner = _winner(tier)
        levels.append({
            "hardware_level_gb": capacity,
            "tier_id": tier.get("id"),
            "status": "winner" if winner else "pending-benchmarks",
            "winner": winner,
        })
    return {
        "schema": "turbofit.list/v1",
        "name": "TurboFit List",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_schema": report.get("schema"),
        "policy": "exact hardware + current recipe + physical evidence + intelligence + TPS",
        "levels": levels,
    }


def render_turbofit_list(payload: Mapping[str, Any]) -> str:
    lines = [
        "# TurboFit List",
        "",
        "The **TurboFit List** is made only from benchmark winners at each physical hardware level. **TurboFit Check** is the system scan-to-configuration process that matches a user's machine to this evidence; it is not a model leaderboard.",
        "",
        "| Hardware level | Winner | Context | Intelligence | TPS | Balanced | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in payload.get("levels", []):
        winner = item.get("winner")
        if not isinstance(winner, Mapping):
            lines.append(f"| {item['hardware_level_gb']} GB | — | — | — | — | — | pending benchmarks |")
            continue
        pair = f"{winner.get('main')} + {winner.get('auxiliary')}"
        lines.append(
            f"| {item['hardware_level_gb']} GB | `{pair}` | {winner.get('context')} | "
            f"{winner.get('intelligence_score')} | {winner.get('measured_tps')} | "
            f"{round(float(winner.get('balanced_score')), 3)} | winner |"
        )
    lines.extend([
        "",
        "A blank level is honest: no current-recipe winner has completed the exact physical and intelligence campaigns for that hardware class yet.",
        "",
    ])
    return "\n".join(lines)
