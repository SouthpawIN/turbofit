"""Render measured configurations in Hermes Index terms for upstream publication.

Nous Research's Hermes Index is the mean pass@1 across four suites run inside
Hermes Agent, reported with cost per task. Local models have no dollar cost, so
this reports seconds per task instead — the same axis, honest currency.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

SUITES = ("hermes-bench", "terminal-bench-4", "terminal-bench-science",
          "skills-bench", "deep-swe", "agentic-pair")


def _latest(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    newest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        cid = str(record.get("configuration_id") or "")
        if cid and (
            cid not in newest
            or str(record.get("recorded_at") or "") >= str(newest[cid].get("recorded_at") or "")
        ):
            newest[cid] = record
    return [dict(r) for r in newest.values()]


def index_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One table row: what was measured, how many suites back it, what it costs."""
    measured = [str(name) for name in (record.get("suites") or []) if name]
    tps = float(record.get("throughput_tps") or 0.0)
    seconds_per_task = round(1.0 / tps, 4) if tps > 0 else None
    return {
        "model": record.get("main"),
        "configuration": record.get("configuration_id"),
        "hardware_tier_gb": record.get("hardware_tier_gb"),
        "context": record.get("context"),
        "auxiliary": record.get("auxiliary"),
        "index": record.get("combined_score") or record.get("intelligence_score"),
        "suites_measured": len(measured),
        "suites_total": len(SUITES),
        "tokens_per_second": tps,
        "seconds_per_task_at_1k": seconds_per_task,
        "level": record.get("benchmark_level"),
    }


def build_table(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [index_row(r) for r in _latest(records)]
    rows.sort(key=lambda r: (-float(r["index"] or 0), str(r["configuration"])))
    return {"rows": rows, "measured": len(rows)}


def render_markdown(table: Mapping[str, Any], *, title: str = "TurboFit — Hermes-Index-compatible results") -> str:
    rows = table.get("rows") or []
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["No measured configurations yet."]) + "\n"
    lines += [
        "Index = mean pass@1 over the suites actually measured (Hermes Index averages four;",
        "`suites` shows how many back each row, so a partial run is never presented as full).",
        "Local inference has no per-task cost, so `s/task@1k` replaces dollars: 1000 tokens ÷ tok/s.",
        "",
        "| Model | Tier | Ctx | Aux | Index | Suites | tok/s | s/task@1k | Level |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {model} | {tier} GB | {ctx} | {aux} | {index:.1f} | {sm}/{st} | {tps:.1f} | {spt} | {lvl} |".format(
                model=r["model"], tier=r["hardware_tier_gb"], ctx=r["context"], aux=r["auxiliary"],
                index=float(r["index"] or 0), sm=r["suites_measured"], st=r["suites_total"],
                tps=float(r["tokens_per_second"] or 0),
                spt=f"{r['seconds_per_task_at_1k']:.2f}" if r["seconds_per_task_at_1k"] else "—",
                lvl=r["level"],
            )
        )
    return "\n".join(lines) + "\n"
