"""Hermes-Index results table for the README, generated from measured evidence.

The Hermes Index is Nous Research's definition of model intelligence: the mean
pass@1 across four suites run inside Hermes Agent (Hermes Bench, TerminalBench 4,
TerminalBench Science, SkillsBench), reported together with the cost of a task.
TurboFit adopts that definition and nothing else: a row is published only when at
least one of those four suites was actually measured.

Two consequences shape this renderer:

* Local inference has no dollar cost, so the price column becomes **seconds per
  task** at a 1,000-token task (``1000 / tok/s``).
* A row always carries ``suites measured / total``, and every hardware tier is
  listed even when nothing was measured on it, so a partial campaign is never
  presented as a full index and an unmeasured tier never silently disappears.

Rows come only from ``references/intelligence-scores.json``; the markdown is
wrapped in generated-content markers and must not be hand-edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCORES_SCHEMA = "turbofit.intelligence-scores/v1"
TIER_REPORT_SCHEMA = "turbofit.hardware-tier-report/v1"

GENERATED_BEGIN = "<!-- turbofit:generated:hermes-index:begin -->"
GENERATED_END = "<!-- turbofit:generated:hermes-index:end -->"
DEFAULT_ANCHOR = "## Keeping it current"

#: The four suites that make up a complete Hermes Index. These names match
#: ``turbofit_runtime.intelligence.HERMES_INDEX_SUITES`` exactly so a record can
#: be checked against either renderer.
HERMES_INDEX_SUITES = (
    "hermes-bench",
    "terminal-bench-4",
    "terminal-bench-science",
    "skills-bench",
)
#: Retired suites. They may exist in archived evidence, never in a live row.
RETIRED_SUITES = ("deep-swe", "agentic-pair")
#: 32K is below spec: a measurement at a smaller window is never published.
MIN_CONTEXT = 65536
#: Local inference has no dollar cost; time per 1,000-token task replaces it.
TOKENS_PER_TASK = 1000


def load_scores(path: Path) -> dict[str, Any]:
    """Load and validate ``turbofit.intelligence-scores/v1``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("intelligence scores must be a JSON object")
    if payload.get("schema") != SCORES_SCHEMA:
        raise ValueError(
            f"unexpected intelligence scores schema: {payload.get('schema')!r} "
            f"(want {SCORES_SCHEMA!r})"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("intelligence scores has no records list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("intelligence record must be an object")
        if not str(record.get("configuration_id") or "").strip():
            raise ValueError("intelligence record has no configuration_id")
        names = {str(m.get("name")) for m in record.get("measurements") or [] if isinstance(m, dict)}
        stale = sorted(names & set(RETIRED_SUITES))
        if stale:
            raise ValueError(
                f"record {record.get('configuration_id')!r} carries retired suite(s) "
                f"{', '.join(stale)}: the index is the four Hermes suites only"
            )
    archived = payload.get("archived") or []
    if not isinstance(archived, list):
        raise ValueError("intelligence scores has a non-list 'archived' block")
    return payload


def _tier_bands(report_path: Path | None) -> list[int]:
    """Tier memory bands, widened by any tier that actually has a measurement."""
    bands: set[int] = set()
    if report_path is not None and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") == TIER_REPORT_SCHEMA:
            bands = {
                int(tier["capacity_gb"])
                for tier in report.get("tiers") or []
                if isinstance(tier, dict) and tier.get("capacity_gb") is not None
            }
    return sorted(bands)


def _quant(record: Mapping[str, Any]) -> str:
    quants = [
        str(q) for q in record.get("quantizations") or [] if str(q) != "route-to-main"
    ]
    return quants[0] if quants else "—"


def _aux(record: Mapping[str, Any]) -> str:
    mode = str(record.get("aux_mode") or "—")
    auxiliary = str(record.get("auxiliary") or "")
    main = str(record.get("main") or "")
    if not auxiliary or auxiliary == main:
        return mode
    return f"{mode} · `{auxiliary}`"


def _suites(record: Mapping[str, Any]) -> tuple[int, int, str, tuple[str, ...]]:
    """Measured Hermes-suite count, total, names, and any retired suites seen."""
    raw = [str(m.get("name")) for m in record.get("measurements") or [] if m.get("name")]
    retired = tuple(sorted({name for name in raw if name in RETIRED_SUITES}))
    names = sorted({name for name in raw if name in HERMES_INDEX_SUITES})
    return len(names), len(HERMES_INDEX_SUITES), ", ".join(names) or "none", retired


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _row(record: Mapping[str, Any]) -> dict[str, Any]:
    context = record.get("context")
    if not isinstance(context, int) or context < MIN_CONTEXT:
        raise ValueError(
            f"record {record.get('configuration_id')!r} has context {context!r}; "
            f"the published floor is {MIN_CONTEXT}"
        )
    measured, total, names, retired = _suites(record)
    tps = _finite(record.get("throughput_tps"))
    # An index exists only if a Hermes suite was measured. A record that carries
    # only retired suites (a legacy DeepSWE row) renders as "no measurements
    # yet" instead of republishing its old composite.
    index = _finite(record.get("intelligence_score")) if measured else None
    return {
        "configuration_id": record.get("configuration_id"),
        "model": str(record.get("main") or record.get("configuration_id")),
        "quant": _quant(record),
        "tier": int(record["hardware_tier_gb"]),
        "context": context,
        "aux": _aux(record),
        "index": index,
        "suites_measured": measured,
        "suites_total": total,
        "suites": f"{measured}/{total}",
        "suite_names": names,
        "retired_suites": retired,
        "tps": tps,
        "seconds_per_task": (TOKENS_PER_TASK / tps) if tps else None,
        "level": str(record.get("benchmark_level") or "—"),
    }


def build_rows(scores: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per measured configuration, newest measurement per configuration.

    A record with no Hermes-suite measurement has no index and therefore no row:
    it is reported through the tier lines as ``no measurements yet``.
    """
    newest: dict[str, Mapping[str, Any]] = {}
    for record in scores.get("records") or []:
        identifier = str(record.get("configuration_id"))
        current = newest.get(identifier)
        if current is None or str(record.get("recorded_at") or "") >= str(
            current.get("recorded_at") or ""
        ):
            newest[identifier] = record
    rows = [_row(record) for record in newest.values()]
    rows = [row for row in rows if row["suites_measured"] > 0]
    rows.sort(key=lambda r: (r["tier"], -(r["index"] or -1.0), r["configuration_id"]))
    return rows


def group_by_tier(
    rows: Iterable[Mapping[str, Any]], tiers: Iterable[int]
) -> list[tuple[int, list[dict[str, Any]]]]:
    """Attach every row to its tier band, and keep empty bands as empty lists."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["tier"]), []).append(dict(row))
    for tier in tiers:
        grouped.setdefault(int(tier), [])
    return [(tier, grouped[tier]) for tier in sorted(grouped)]


def _number(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def render_section(scores: Mapping[str, Any], tiers: Sequence[int]) -> str:
    """Render the README section, wrapped in generated markers."""
    rows = build_rows(scores)
    lines: list[str] = [GENERATED_BEGIN, ""]
    lines.append("## Intelligence — what actually runs, and how fast")
    lines.append("")
    lines.append(
        "The **Hermes Index** is the mean pass@1 across the four suites Hermes Agent "
        "runs — Hermes Bench, TerminalBench 4, TerminalBench Science and SkillsBench — "
        "reported with the cost of a task. Local inference has no dollar cost, so the "
        "cost column is **seconds per task** at a 1,000-token task, and the suite "
        "column always reads *measured / total*: a partial run is never a full index."
    )
    lines.append("")
    lines.append(
        "| Model | Tier | Context | Aux | Index | Suites | tok/s | s/task @1k | Measurement |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for tier, tier_rows in group_by_tier(rows, tiers):
        if not tier_rows:
            continue
        for row in tier_rows:
            lines.append(
                f"| `{row['model']}` {row['quant']} | {row['tier']} GB "
                f"| {row['context']:,} | {row['aux']} "
                f"| {_number(row['index'], 1)} | {row['suites']} "
                f"| {_number(row['tps'], 2)} | {_number(row['seconds_per_task'], 2)} "
                f"| {row['level']} |"
            )
    unmeasured = [tier for tier, tier_rows in group_by_tier(rows, tiers) if not tier_rows]
    if unmeasured:
        listed = ", ".join(f"{tier} GB" for tier in unmeasured)
        lines.append(
            f"| — | {listed} | — | — | **no measurements yet** | 0/4 | — | — | — |"
        )
    lines.append("")
    lines.append(
        "A suite that cannot be run here is reported as **unavailable** with its reason rather "
        "than as a zero, so `suites` shows exactly how much of the index a row carries. "
        "Hermes Bench is run by Nous and is not publicly reproducible; TerminalBench 4, "
        "TerminalBench Science and SkillsBench are wired through their own harnesses against a "
        "TurboFit-served local endpoint. `s/task @1k` is `1,000 ÷ tok/s`. The `no measurements "
        "yet` row lists every tier the campaign has not reached — those tiers stay visible "
        "instead of being dropped."
    )
    lines.append("")
    archived = [item for item in scores.get("archived") or [] if isinstance(item, Mapping)]
    if archived:
        ids = ", ".join(f"`{item.get('configuration_id')}`" for item in archived)
        lines.append(
            f"Superseded and not published: {ids} — scored under the retired "
            "`deep-swe`/`agentic-pair` definition, which no longer contributes to any index."
        )
        lines.append("")
    lines.append(
        "Source: `references/intelligence-scores.json`; regenerate with "
        "`tools/turbofit-intel-leaderboard`. Do not hand-edit between the markers."
    )
    lines.append("")
    lines.append(GENERATED_END)
    return "\n".join(lines) + "\n"


def replace_generated(text: str, section: str, anchor: str = DEFAULT_ANCHOR) -> str:
    """Insert or replace the generated block in a README, leaving prose intact."""
    if GENERATED_BEGIN in text or GENERATED_END in text:
        if text.count(GENERATED_BEGIN) != 1 or text.count(GENERATED_END) != 1:
            raise ValueError(f"malformed generated markers: {GENERATED_BEGIN}")
        before, remainder = text.split(GENERATED_BEGIN, 1)
        _, after = remainder.split(GENERATED_END, 1)
        return before + section.rstrip("\n") + after
    if anchor not in text:
        raise ValueError(f"README insertion anchor missing: {anchor}")
    return text.replace(anchor, section + "\n" + anchor, 1)


def render_readme(
    repo_root: Path, readme: Path, anchor: str = DEFAULT_ANCHOR
) -> str:
    """Return the README text with the section regenerated."""
    repo_root = Path(repo_root)
    scores = load_scores(repo_root / "references" / "intelligence-scores.json")
    tiers = list(_tier_bands(repo_root / "references" / "hardware-tier-report.json"))
    tiers = sorted(set(tiers) | {row["tier"] for row in build_rows(scores)})
    text = Path(readme).read_text(encoding="utf-8")
    return replace_generated(text, render_section(scores, tiers), anchor)


def publish(repo_root: Path, readme: Path, anchor: str = DEFAULT_ANCHOR) -> Path:
    """Rewrite the README in place with a fresh section. Idempotent atomic write."""
    target = Path(readme)
    rendered = render_readme(repo_root, target, anchor)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    tmp.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the README Hermes-Index table.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--readme", type=Path, default=None)
    parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    args = parser.parse_args(argv)
    readme = args.readme or (args.repo_root / "README.md")
    target = publish(args.repo_root, readme, args.anchor)
    print(f"wrote hermes-index section to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
