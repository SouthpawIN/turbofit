"""Rank measured configurations into a recommendation per hardware tier.

The tier report states what *should* fit. This states what *was measured*, so the
recommendation for a slot is evidence rather than arithmetic on file sizes.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA = "turbofit.intelligence-leaderboard/v1"


def _rank_key(record: Mapping[str, Any]) -> tuple[float, float, str]:
    # Intelligence first, then throughput, then a stable tiebreak so the document
    # is deterministic across runs and machines.
    return (
        -float(record.get("intelligence_score") or 0.0),
        -float(record.get("throughput_tps") or 0.0),
        str(record.get("configuration_id") or ""),
    )


def latest_per_configuration(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the newest measurement for each configuration."""
    newest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        cid = str(record.get("configuration_id") or "")
        if not cid:
            continue
        current = newest.get(cid)
        if current is None or str(record.get("recorded_at") or "") >= str(current.get("recorded_at") or ""):
            newest[cid] = record
    return [dict(record) for record in newest.values()]


def build_leaderboard(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One winner and its options per hardware tier, from real measurements."""
    rows = latest_per_configuration(records)
    by_tier: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        tier = row.get("hardware_tier_gb")
        if tier is None:
            continue
        by_tier.setdefault(int(tier), []).append(row)

    tiers = []
    for tier in sorted(by_tier):
        ranked = sorted(by_tier[tier], key=_rank_key)
        best = ranked[0]
        tiers.append(
            {
                "hardware_tier_gb": tier,
                "recommended": {
                    "configuration_id": best.get("configuration_id"),
                    "main": best.get("main"),
                    "auxiliary": best.get("auxiliary"),
                    "context": best.get("context"),
                    "intelligence_score": best.get("intelligence_score"),
                    "throughput_tps": best.get("throughput_tps"),
                    "balanced_score": best.get("balanced_score"),
                    "level": best.get("benchmark_level"),
                    "recorded_at": best.get("recorded_at"),
                },
                "options": [
                    {
                        "configuration_id": row.get("configuration_id"),
                        "main": row.get("main"),
                        "intelligence_score": row.get("intelligence_score"),
                        "throughput_tps": row.get("throughput_tps"),
                        "level": row.get("benchmark_level"),
                    }
                    for row in ranked[1:]
                ],
                "measured": len(ranked),
            }
        )
    return {"schema": SCHEMA, "tiers": tiers, "measured_configurations": len(rows)}


def render_markdown(board: Mapping[str, Any]) -> str:
    lines = ["# Intelligence leaderboard", ""]
    if not board.get("tiers"):
        lines.append("No measured configurations yet.")
        return "\n".join(lines) + "\n"
    lines.append("Ranked by intelligence at the configuration's own hardware tier; throughput breaks ties.")
    for tier in board["tiers"]:
        best = tier["recommended"]
        lines += ["", f"## {tier['hardware_tier_gb']} GB", "",
                  f"**Recommended:** {best['main']} at {best['context']} — "
                  f"intelligence {best['intelligence_score']:.1f}, {best['throughput_tps']:.2f} tok/s "
                  f"({best['level']})"]
        if tier["options"]:
            lines += ["", "| option | intelligence | tok/s | level |", "|---|---|---|---|"]
            for opt in tier["options"]:
                lines.append(
                    f"| {opt['main']} | {float(opt['intelligence_score'] or 0):.1f} | "
                    f"{float(opt['throughput_tps'] or 0):.2f} | {opt['level']} |"
                )
    return "\n".join(lines) + "\n"
