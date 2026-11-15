"""Contract tests for the generated Hermes-Index README section.

The README is generated from the wiki hub, so the measured table has to be
regenerated from ``references/intelligence-scores.json`` after every sync — it can
never be hand-edited. These tests pin the two properties that matter most:

* every tier is visible even when nothing was measured on it, rendered as
  ``no measurements yet`` rather than quietly dropped, and
* ``suites measured / total`` is always shown, so a partial campaign is never
  presented as a full Hermes Index.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.intelligence_readme import (
    GENERATED_BEGIN,
    GENERATED_END,
    HERMES_INDEX_SUITES,
    MIN_CONTEXT,
    SCORES_SCHEMA,
    TOKENS_PER_TASK,
    build_rows,
    load_scores,
    render_readme,
    render_section,
    replace_generated,
)

ROOT = Path(__file__).parents[1]
SCORES = ROOT / "references" / "intelligence-scores.json"
TIER_REPORT = ROOT / "references" / "hardware-tier-report.json"
README = ROOT / "README.md"

HEADER = "| Model | Tier | Context | Aux | Index | Suites | tok/s | s/task @1k | Measurement |"


def _record(**overrides):
    record = {
        "configuration_id": "fixture-model--auto--262k",
        "main": "fixture-model",
        "auxiliary": "auto:fixture-model",
        "aux_mode": "shared-main",
        "benchmark_level": "promotion",
        "context": 262144,
        "hardware_tier_gb": 48,
        "intelligence_score": 75.0,
        "measurements": [
            {"name": "deep-swe", "score": 0.5, "tasks_passed": 1, "tasks_total": 2},
            {"name": "agentic-pair", "score": 1.0, "tasks_passed": 16, "tasks_total": 16},
        ],
        "quantizations": ["UD-Q4_K_M", "route-to-main"],
        "recorded_at": "2026-08-23T05:38:24.951167+00:00",
        "throughput_tps": 40.0,
    }
    record.update(overrides)
    return record


def _repo(tmp_path: Path, records, tiers=(8, 48, 96)) -> Path:
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "intelligence-scores.json").write_text(
        json.dumps(
            {
                "schema": SCORES_SCHEMA,
                "records": records,
                "required_benchmarks": ["deep-swe", "agentic-pair"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "references" / "hardware-tier-report.json").write_text(
        json.dumps(
            {
                "schema": "turbofit.hardware-tier-report/v1",
                "tiers": [{"id": f"hardware-{gb}gb", "capacity_gb": gb} for gb in tiers],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Fixture\n\nintro\n\n## Keeping it current\n\ntail\n", encoding="utf-8")
    return tmp_path


def test_fixture_section_is_generated_between_markers(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record()])
    text = render_readme(repo, repo / "README.md")

    assert text.count(GENERATED_BEGIN) == 1
    assert text.count(GENERATED_END) == 1
    assert text.index(GENERATED_BEGIN) < text.index(HEADER) < text.index(GENERATED_END)
    # Inserted above the anchor, prose untouched.
    assert text.index(GENERATED_END) < text.index("## Keeping it current")
    assert text.startswith("# Fixture\n\nintro\n")

    assert HEADER in text
    assert "`fixture-model` UD-Q4_K_M" in text
    assert "| 48 GB | 262,144 |" in text
    assert "| 25.00 |" in text
    assert "| 2/4 |" in text
    assert "| promotion |" in text


def test_no_measurement_tier_renders_not_dropped(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record()], tiers=(8, 48, 96))
    section = render_section(load_scores(repo / "references" / "intelligence-scores.json"), [8, 48, 96])

    assert "no measurements yet" in section
    unmeasured_row = next(line for line in section.splitlines() if "no measurements yet" in line)
    for gb in (8, 96):
        assert f"{gb} GB" in unmeasured_row
    assert "48 GB" not in unmeasured_row
    # The measured tier is not mislabelled.
    measured_row = next(line for line in section.splitlines() if "fixture-model" in line)
    assert "no measurements yet" not in measured_row


def test_tier_with_measurement_but_no_report_band_is_kept(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record(hardware_tier_gb=200)], tiers=(8, 48))
    text = render_readme(repo, repo / "README.md")
    assert "| 200 GB |" in text
    unmeasured_row = next(line for line in text.splitlines() if "no measurements yet" in line)
    assert "200 GB" not in unmeasured_row


def test_suites_total_is_the_full_hermes_index(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record()])
    rows = build_rows(load_scores(repo / "references" / "intelligence-scores.json"))
    assert rows[0]["suites_measured"] == 2
    assert rows[0]["suites_total"] == 4
    assert rows[0]["suites"] == "2/4"
    assert 2 / 4 < 1.0  # a partial run is never a full index


def test_seconds_per_task_is_one_thousand_over_tps(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record(throughput_tps=25.0)])
    rows = build_rows(load_scores(repo / "references" / "intelligence-scores.json"))
    assert rows[0]["seconds_per_task"] == pytest.approx(40.0)


def test_missing_throughput_renders_dash_not_zero(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record(throughput_tps=None)])
    section = render_section(load_scores(repo / "references" / "intelligence-scores.json"), [48])
    row = next(line for line in section.splitlines() if "fixture-model" in line)
    assert row.count("| — |") >= 1


def test_context_below_floor_is_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record(context=32768)])
    with pytest.raises(ValueError, match=str(MIN_CONTEXT)):
        build_rows(load_scores(repo / "references" / "intelligence-scores.json"))


def test_unknown_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"schema": "nope", "records": []}), encoding="utf-8")
    with pytest.raises(ValueError, match=SCORES_SCHEMA):
        load_scores(path)


def test_replacement_is_idempotent_and_preserves_prose(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record()])
    scores = load_scores(repo / "references" / "intelligence-scores.json")
    section = render_section(scores, [8, 48, 96])
    once = render_readme(repo, repo / "README.md")
    assert replace_generated(once, section) == once
    assert once.count(GENERATED_BEGIN) == 1
    assert once.count("## Keeping it current") == 1


def test_missing_anchor_is_an_error_when_no_markers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="anchor"):
        replace_generated("# no anchor here\n", GENERATED_BEGIN + "\nx\n" + GENERATED_END)


def test_rendered_output_carries_no_device_names(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [_record()])
    section = render_section(load_scores(repo / "references" / "intelligence-scores.json"), [8, 48])
    for banned in ("3090", "4090", "5090", "RTX", "Xeon", "Threadripper"):
        assert banned not in section
    assert "GB" in section  # tiers are generic memory bands


def test_empty_campaign_lists_every_tier(tmp_path: Path) -> None:
    repo = _repo(tmp_path, [], tiers=(8, 48, 96))
    text = render_readme(repo, repo / "README.md")
    row = next(line for line in text.splitlines() if "no measurements yet" in line)
    for gb in (8, 48, 96):
        assert f"{gb} GB" in row
    assert HEADER in text


def test_committed_readme_section_matches_regeneration() -> None:
    """The committed block must be exactly what the renderer produces today."""
    committed = README.read_text(encoding="utf-8")
    assert GENERATED_BEGIN in committed and GENERATED_END in committed
    assert render_readme(ROOT, README) == committed


def test_committed_scores_render_a_row_per_measured_tier() -> None:
    scores = load_scores(SCORES)
    rows = build_rows(scores)
    section = render_section(scores, sorted({row["tier"] for row in rows} | {8, 96}))
    assert "no measurements yet" in section  # 8 GB / 96 GB are unmeasured
    for row in rows:
        assert f"| {row['tier']} GB |" in section
    assert TIER_REPORT.is_file()
