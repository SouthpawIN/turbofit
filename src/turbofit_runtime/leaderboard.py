"""Benchmark leaderboard: best model + serveable engines per hardware tier.

Built deterministically from two committed reference sources:

  - ``references/hardware-tier-report.json``  (``turbofit.hardware-tier-report/v1``)
  - ``references/engine-serve-matrix.json``   (``turbofit.engine-serve-matrix/v1``)

Selection rules (documented in the rendered page):

1. A tier with ``status == "physically-validated"`` and a
   ``recommendations.measured_winner`` uses that winner (``basis="measured"``).
2. Otherwise the top catalog candidate by
   ``(intelligence_score desc, balanced_score desc, context desc,
   configuration_id asc)`` is used (``basis="catalog-candidate"``). Catalog
   tiers carry no measured scores, so context is the effective tiebreak —
   the candidate with the most capability headroom.
3. Serve engines resolve ONLY by exact match of the winner's ``main.id``
   against an engine-matrix pair key (the fit-list slug). Unmatched models
   fall back to ``["native"]`` with ``engine_source="default"`` — no fuzzy
   mapping, no invented engines.

The rendered markdown is wrapped in generated-content markers so wiki tooling
can replace it without disturbing surrounding hand-written prose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEADERBOARD_SCHEMA = "turbofit.benchmark-leaderboard/v1"
TIER_REPORT_SCHEMA = "turbofit.hardware-tier-report/v1"
ENGINE_MATRIX_SCHEMA = "turbofit.engine-serve-matrix/v1"

GENERATED_BEGIN = "<!-- turbofit:generated:benchmark-leaderboard:begin -->"
GENERATED_END = "<!-- turbofit:generated:benchmark-leaderboard:end -->"

DEFAULT_ENGINES = ("native",)


def load_tier_report(path: Path) -> dict[str, Any]:
    """Load and validate the hardware tier report."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hardware tier report must be a JSON object")
    if payload.get("schema") != TIER_REPORT_SCHEMA:
        raise ValueError(
            f"unexpected tier report schema: {payload.get('schema')!r} "
            f"(want {TIER_REPORT_SCHEMA!r})"
        )
    tiers = payload.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("hardware tier report has no tiers")
    for tier in tiers:
        if not isinstance(tier, dict) or tier.get("id") is None:
            raise ValueError("tier report contains a tier without an id")
        candidates = tier.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"tier {tier.get('id')!r} has no candidates")
    return payload


def load_engine_matrix(path: Path) -> dict[str, Any]:
    """Load and validate the engine/serve matrix."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("engine matrix must be a JSON object")
    if payload.get("schema") != ENGINE_MATRIX_SCHEMA:
        raise ValueError(
            f"unexpected engine matrix schema: {payload.get('schema')!r} "
            f"(want {ENGINE_MATRIX_SCHEMA!r})"
        )
    pairs = payload.get("pairs")
    if not isinstance(pairs, dict):
        raise ValueError("engine matrix has no pairs")
    return payload


def resolve_engines(model_id: str, matrix: dict[str, Any]) -> dict[str, Any]:
    """Resolve serve engines for a winner model.

    Exact fit-list-slug match only. Anything else is ``default`` (native),
    never a guessed mapping.
    """
    pair = matrix.get("pairs", {}).get(model_id)
    if isinstance(pair, dict):
        eligible = pair.get("eligible_engines")
        if isinstance(eligible, list) and eligible:
            return {
                "engines": list(eligible),
                "artifact": pair.get("artifact"),
                "priority": list(pair.get("priority") or []),
                "apple_alternate": pair.get("apple_alternate"),
                "hf_engines": list(pair.get("hf_engines") or []),
                "engine_source": "matrix",
            }
    return {
        "engines": list(DEFAULT_ENGINES),
        "artifact": None,
        "priority": [],
        "apple_alternate": None,
        "hf_engines": [],
        "engine_source": "default",
    }


def _catalog_rank(c: dict[str, Any]) -> tuple:
    intelligence = c.get("intelligence_score")
    balanced = c.get("balanced_score")
    context = c.get("context") or 0
    return (
        -(intelligence if isinstance(intelligence, (int, float)) else -1),
        -(balanced if isinstance(balanced, (int, float)) else -1),
        -int(context),
        str(c.get("configuration_id") or ""),
    )


def select_winner(tier: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Pick the tier winner and its evidence basis (see module docstring)."""
    recommendations = tier.get("recommendations") or {}
    winner = recommendations.get("measured_winner")
    if (
        tier.get("status") == "physically-validated"
        and isinstance(winner, dict)
        and winner.get("main")
    ):
        return winner, "measured"
    candidates = [c for c in tier.get("candidates", []) if isinstance(c, dict)]
    if not candidates:
        raise ValueError(f"tier {tier.get('id')!r} has no selectable candidates")
    return min(candidates, key=_catalog_rank), "catalog-candidate"


def _winner_payload(c: dict[str, Any], basis: str) -> dict[str, Any]:
    main = c.get("main") or {}
    fit = c.get("fit") or {}
    measured_tps = c.get("measured_tps")
    intelligence = c.get("intelligence_score")
    return {
        "configuration_id": c.get("configuration_id"),
        "model": main.get("id"),
        "model_name": main.get("name"),
        "quantization": main.get("quantization"),
        "context": c.get("context"),
        "basis": basis,
        "measured_tps": measured_tps if isinstance(measured_tps, (int, float)) else None,
        "intelligence_score": intelligence if isinstance(intelligence, (int, float)) else None,
        "evidence": fit.get("evidence"),
        "hardware_fingerprint": fit.get("hardware_fingerprint"),
    }


def build_leaderboard(report: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic leaderboard document (``LEADERBOARD_SCHEMA``)."""
    tiers = sorted(report["tiers"], key=lambda t: int(t.get("capacity_gb") or 0))
    rows: list[dict[str, Any]] = []
    for tier in tiers:
        winner_c, basis = select_winner(tier)
        engines = resolve_engines(winner_c["main"]["id"], matrix)
        candidates = [c for c in tier.get("candidates", []) if isinstance(c, dict)]
        ranked = sorted(candidates, key=_catalog_rank)
        winner_cfg = winner_c.get("configuration_id")
        runners_up = [
            c.get("configuration_id")
            for c in ranked
            if c.get("configuration_id") and c.get("configuration_id") != winner_cfg
        ][:2]
        topology = tier.get("topology") or {}
        rows.append({
            "tier_id": tier.get("id"),
            "capacity_gb": tier.get("capacity_gb"),
            "status": tier.get("status"),
            "topology": topology.get("topology")
            or (topology.get("total_vram_gb") if topology else None),
            "winner": _winner_payload(winner_c, basis),
            "engines": engines["engines"],
            "artifact": engines["artifact"],
            "priority": engines["priority"],
            "apple_alternate": engines["apple_alternate"],
            "hf_engines": engines["hf_engines"],
            "engine_source": engines["engine_source"],
            "runners_up": runners_up,
        })
    current = report.get("current_hardware") or {}
    return {
        "schema": LEADERBOARD_SCHEMA,
        "generated_from": {
            "tier_report": report.get("schema"),
            "engine_matrix": matrix.get("schema"),
            "tier_report_host": {
                "os": current.get("os"),
                "architecture": current.get("architecture"),
                "devices": current.get("device_memory_mb"),
                "native_tier_gb": current.get("native_tier_gb"),
            },
        },
        "tiers": rows,
    }


def _fmt(value: Any, default: str = "—") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_leaderboard_markdown(leaderboard: dict[str, Any]) -> str:
    """Render the leaderboard to markdown, wrapped in generated markers."""
    host = leaderboard.get("generated_from", {}).get("tier_report_host", {})
    lines: list[str] = []
    lines.append(GENERATED_BEGIN)
    lines.append("")
    lines.append("# Benchmark leaderboard — best model + engines per tier")
    lines.append("")
    lines.append(
        "Deterministic output of the committed tier report joined to the "
        "engine/serve matrix. Regenerated by the runtime tests and the "
        "dashboard endpoint; do not hand-edit between the markers."
    )
    lines.append("")
    if host:
        lines.append(
            f"Tier report host: {host.get('os', 'unknown')} {host.get('architecture', '')} · "
            f"{host.get('devices') or 'unknown'} accelerators · native tier {host.get('native_tier_gb', '?')} GB."
        )
        lines.append("")
    lines.append("| Tier | Best model | Quant | Context | Evidence | TPS | Serve engines |")
    lines.append("|------|-----------|-------|---------|----------|-----|---------------|")
    for row in leaderboard["tiers"]:
        w = row["winner"]
        engines = ", ".join(row["engines"])
        if row["engine_source"] == "default":
            engines += " (default)"
        evidence = "measured" if w["basis"] == "measured" else "catalog"
        lines.append(
            f"| {_fmt(row['capacity_gb'])} GB | {w.get('model_name') or w.get('model') or '—'} "
            f"| {w.get('quantization') or '—'} | {_fmt(w.get('context'))} "
            f"| {evidence} | {_fmt(w.get('measured_tps'), 'pending')} "
            f"| {engines} |"
        )
    lines.append("")
    lines.append("## Per-tier detail")
    for row in leaderboard["tiers"]:
        w = row["winner"]
        lines.append("")
        lines.append(f"### {_fmt(row['capacity_gb'])} GB — {row.get('status', 'unknown')}")
        basis_note = (
            "measured winner from physically-validated evidence"
            if w["basis"] == "measured"
            else "top catalog candidate (no measured scores on this tier yet)"
        )
        lines.append(
            f"- Winner: **{w.get('model_name') or w.get('model') or '—'}** "
            f"({w.get('configuration_id') or '—'}) · {basis_note}"
        )
        if w.get("measured_tps") is not None:
            lines.append(f"- Measured: {w['measured_tps']:.2f} tok/s"
                         + (f" · intelligence {w['intelligence_score']:.3f}"
                            if w.get("intelligence_score") is not None else ""))
        if w.get("evidence"):
            lines.append(f"- Evidence: `{w['evidence']}` (hash-bound)"
                         + (f" · fingerprint `{w['hardware_fingerprint']}`"
                            if w.get("hardware_fingerprint") else ""))
        artifact = f" · {row['artifact']}" if row.get("artifact") else ""
        priority = f" · priority {row['priority'][0]}" if row.get("priority") else ""
        lines.append(
            f"- Serve: {', '.join(row['engines'])} ({row['engine_source']}){artifact}{priority}"
        )
        if row.get("apple_alternate"):
            lines.append(f"- Apple alternate: {row['apple_alternate']}")
        if row.get("hf_engines"):
            lines.append(f"- HF serving (not local): {', '.join(row['hf_engines'])}")
        if row.get("runners_up"):
            lines.append(f"- Runners-up: {', '.join(row['runners_up'])}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("1. Tiers with `physically-validated` status use their `measured_winner`; "
                 "all other tiers use the top catalog candidate by intelligence score, "
                 "then balanced score, then context (catalog tiers carry no measured scores).")
    lines.append("2. Serve engines resolve only by exact fit-list-slug match against the "
                 "engine/serve matrix pair keys. Unmatched models default to `native` and "
                 "are pending matrix coverage — never a guessed mapping.")
    lines.append("3. Sources (repo `references/`): `hardware-tier-report.json` "
                 f"({TIER_REPORT_SCHEMA}) and `engine-serve-matrix.json` ({ENGINE_MATRIX_SCHEMA}).")
    lines.append("")
    lines.append(GENERATED_END)
    return "\n".join(lines) + "\n"


def publish_leaderboard(
    report_path: Path, matrix_path: Path, target_path: Path
) -> Path:
    """Build, render, and atomically write the generated page. Idempotent."""
    report = load_tier_report(report_path)
    matrix = load_engine_matrix(matrix_path)
    markdown = render_leaderboard_markdown(build_leaderboard(report, matrix))
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    target = publish_leaderboard(
        args.repo_root / "references" / "hardware-tier-report.json",
        args.repo_root / "references" / "engine-serve-matrix.json",
        args.target,
    )
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
