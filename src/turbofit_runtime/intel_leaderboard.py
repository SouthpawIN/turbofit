"""Rank measured configurations into a recommendation per hardware tier.

The tier report states what *should* fit. This states what *was measured*, so the
recommendation for a slot is evidence rather than arithmetic on file sizes.

Recommendation honours the quantization policy: a measured Q5/Q6 configuration
is never the default pick, even when it scores highest.  It stays in ``options``
so the measurement is not hidden, and the tier records why it was skipped.  A
Q8/full-precision measurement is eligible with its reason attached.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .quant_policy import classify_entry, default_admits, entry_quantization

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


def _catalog_index(catalog: Any) -> dict[str, Any]:
    if catalog is None:
        return {}
    if isinstance(catalog, Mapping):
        return {str(item["id"]): item for item in catalog.get("models") or []}
    return {str(item.id): item for item in getattr(catalog, "models", ())}


def _entry_for(record: Mapping[str, Any], index: Mapping[str, Any]) -> Any:
    for key in ("catalog_main", "main"):
        identifier = record.get(key)
        if identifier is not None and str(identifier) in index:
            return index[str(identifier)]
    return None


def _quant_fields(entry: Any) -> dict[str, Any]:
    if entry is None:
        return {}
    return {
        "quantization": entry_quantization(entry),
        "quant_class": classify_entry(entry).quant_class,
    }


def build_leaderboard(
    records: Iterable[Mapping[str, Any]],
    catalog: Any = None,
) -> dict[str, Any]:
    """One winner and its options per hardware tier, from real measurements.

    ``catalog`` (a model-catalog payload or ``ModelCatalog``) lets the policy
    classify each measured main.  Without it the leaderboard is unchanged.
    """
    rows = latest_per_configuration(records)
    index = _catalog_index(catalog)
    by_tier: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        tier = row.get("hardware_tier_gb")
        if tier is None:
            continue
        by_tier.setdefault(int(tier), []).append(row)

    tiers = []
    for tier in sorted(by_tier):
        ranked = sorted(by_tier[tier], key=_rank_key)
        admitted: list[tuple[dict[str, Any], Any, str]] = []
        skipped: list[dict[str, Any]] = []
        for row in ranked:
            entry = _entry_for(row, index)
            ok, reason = (True, "")
            if entry is not None:
                # A measured row ran at this tier, so its weights + KV fit here.
                ok, reason = default_admits(entry, fits_fully_in_vram=True)
            if ok:
                admitted.append((row, entry, reason))
            else:
                skipped.append({
                    "configuration_id": row.get("configuration_id"),
                    "main": row.get("main"),
                    "reason": reason,
                    **_quant_fields(entry),
                })
        if not admitted:
            tiers.append({
                "hardware_tier_gb": tier,
                "recommended": None,
                "reason": "every measured configuration is outside the quantization policy",
                "options": [],
                "quant_skipped": skipped,
                "measured": len(ranked),
            })
            continue
        best, best_entry, best_reason = admitted[0]
        recommended = {
            "configuration_id": best.get("configuration_id"),
            "main": best.get("main"),
            "auxiliary": best.get("auxiliary"),
            "context": best.get("context"),
            "intelligence_score": best.get("intelligence_score"),
            "throughput_tps": best.get("throughput_tps"),
            "balanced_score": best.get("balanced_score"),
            "level": best.get("benchmark_level"),
            "recorded_at": best.get("recorded_at"),
            **_quant_fields(best_entry),
        }
        if best_entry is not None and not classify_entry(best_entry).is_allowed:
            recommended["quant_exception_reason"] = best_reason
        tiers.append(
            {
                "hardware_tier_gb": tier,
                "recommended": recommended,
                "options": [
                    {
                        "configuration_id": row.get("configuration_id"),
                        "main": row.get("main"),
                        "intelligence_score": row.get("intelligence_score"),
                        "throughput_tps": row.get("throughput_tps"),
                        "level": row.get("benchmark_level"),
                        **_quant_fields(entry),
                    }
                    for row, entry, _reason in admitted[1:]
                ]
                + [
                    {
                        "configuration_id": record["configuration_id"],
                        "main": record.get("main"),
                        "intelligence_score": None,
                        "throughput_tps": None,
                        "level": None,
                        "quantization": record.get("quantization"),
                        "quant_class": record.get("quant_class"),
                        "not_default": record["reason"],
                    }
                    for record in skipped
                ],
                "quant_skipped": skipped,
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
        if best is None:
            lines += ["", f"## {tier['hardware_tier_gb']} GB", "",
                      f"**No recommendation:** {tier.get('reason', 'no eligible measurement')}"]
            continue
        lines += ["", f"## {tier['hardware_tier_gb']} GB", "",
                  f"**Recommended:** {best['main']} at {best['context']} — "
                  f"intelligence {best['intelligence_score']:.1f}, {best['throughput_tps']:.2f} tok/s "
                  f"({best['level']})"]
        if best.get("quant_exception_reason"):
            lines.append(f"*Quant exception:* {best['quant_exception_reason']}")
        if tier["options"]:
            lines += ["", "| option | intelligence | tok/s | level |", "|---|---|---|---|"]
            for opt in tier["options"]:
                if opt.get("not_default"):
                    lines.append(
                        f"| {opt['main']} (not default: {opt['not_default']}) | — | — | — |"
                    )
                    continue
                lines.append(
                    f"| {opt['main']} | {float(opt['intelligence_score'] or 0):.1f} | "
                    f"{float(opt['throughput_tps'] or 0):.2f} | {opt['level']} |"
                )
    return "\n".join(lines) + "\n"
