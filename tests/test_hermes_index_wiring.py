"""The index is the four Hermes suites — provenance tests for that switch.

DeepSWE and the agentic pair remain usable harnesses, but nothing in the shipped
repository may turn their numbers into an index. These tests pin the wiring:
the campaign runs the Hermes suites, the level parameters describe suites, and
the committed score file archives the retired record instead of publishing it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.intelligence import HERMES_INDEX_SUITES, RETIRED_BENCHMARKS
from turbofit_runtime.intelligence_campaign import LEVEL_PARAMETERS
from turbofit_runtime.intelligence_readme import build_rows, load_scores, render_section

ROOT = Path(__file__).parents[1]
CAMPAIGN = ROOT / "scripts/turbofit-intelligence-campaign"
SCORES = ROOT / "references/intelligence-scores.json"
README = ROOT / "README.md"


def test_the_index_is_exactly_the_four_hermes_suites() -> None:
    assert HERMES_INDEX_SUITES == (
        "hermes-bench", "terminal-bench-4", "terminal-bench-science", "skills-bench",
    )
    assert RETIRED_BENCHMARKS == ("deep-swe", "agentic-pair")


def test_campaign_runs_hermes_suites_and_has_no_deepswe_scoring_path() -> None:
    source = CAMPAIGN.read_text(encoding="utf-8")

    assert "_run_index_suites" in source
    assert "HERMES_INDEX_SUITES" in source
    # The retired aggregation helpers are gone: no DeepSWE number reaches a score.
    assert "_deepswe_aggregate" not in source
    assert "_agentic_aggregate" not in source
    assert '"deep-swe"' not in source
    assert '"agentic-pair"' not in source


def test_level_parameters_describe_suites_not_the_retired_pair() -> None:
    for level, parameters in LEVEL_PARAMETERS.items():
        assert set(parameters["suite_tasks"]) == {
            "terminal-bench-4", "terminal-bench-science", "skills-bench",
        }, level
        assert "deepswe_tasks" not in parameters
    # Screening is one task per suite: real, small, never a fabricated full run.
    assert LEVEL_PARAMETERS["screening"]["suite_tasks"] == {
        "terminal-bench-4": 1, "terminal-bench-science": 1, "skills-bench": 1,
    }


def test_committed_scores_publish_only_hermes_index_measurements() -> None:
    payload = json.loads(SCORES.read_text(encoding="utf-8"))

    assert payload["index_suites"] == list(HERMES_INDEX_SUITES)
    # A real trial has been recorded, so the file is no longer empty — but what
    # it publishes is still only the Hermes suites.
    assert payload["records"], "a completed trial was recorded; the file must not be empty"
    for record in payload["records"]:
        names = {measurement["name"] for measurement in record["measurements"]}
        assert names, "a record without measurements is not a score"
        assert names <= set(HERMES_INDEX_SUITES), names
        assert not names & set(RETIRED_BENCHMARKS), names
        coverage = record["suite_coverage"]
        assert 1 <= coverage["measured"] <= 4
        assert coverage["measured_suites"] == [
            name for name in HERMES_INDEX_SUITES if name in names
        ]
        # The index is the mean over the suites that ran, never over all four.
        mean = 100.0 * sum(m["score"] for m in record["measurements"]) / len(names)
        assert record["intelligence_score"] == pytest.approx(mean)
        for measurement in record["measurements"]:
            assert measurement["tasks_total"] >= 1
            assert measurement["raw_result_sha256"].startswith("sha256:")

    archived = payload["archived"]
    assert len(archived) == 1
    record = archived[0]
    assert record["configuration_id"] == "qwen3-8-27b-gsq-rco-iq3s--auto--262k"
    assert record["status"] == "superseded"
    assert "retired" in record["superseded_because"]
    # Provenance is kept: the old numbers are still readable, and still label
    # themselves as the retired definition.
    assert record["previous_intelligence_score"] == 46.875
    assert {m["name"] for m in record["previous_measurements"]} == {"deep-swe", "agentic-pair"}


def test_no_committed_row_can_render_a_deepswe_index() -> None:
    scores = load_scores(SCORES)

    rows = build_rows(scores)
    assert rows, "the recorded trial must render, or the table hides a real measurement"
    for row in rows:
        names = {name.strip() for name in str(row["suite_names"]).split(",") if name.strip()}
        assert names and names != {"none"}, row
        assert names <= set(HERMES_INDEX_SUITES), row
        assert not row["retired_suites"], row
        # Partial coverage is published as such, never rounded up to 4/4.
        assert row["suites"] == f"{row['suites_measured']}/{len(HERMES_INDEX_SUITES)}"
        assert 1 <= row["suites_measured"] < len(HERMES_INDEX_SUITES)

    section = render_section(scores, [48])
    assert "1/4" in section
    assert "46.9" not in section
    assert "Superseded and not published" in section


def test_committed_readme_shows_the_superseded_record_without_a_number() -> None:
    text = README.read_text(encoding="utf-8")

    start = text.index("<!-- turbofit:generated:hermes-index:begin -->")
    end = text.index("<!-- turbofit:generated:hermes-index:end -->")
    section = text[start:end]
    assert "no measurements yet" in section
    assert "qwen3-8-27b-gsq-rco-iq3s--auto--262k" in section
    assert "46.9" not in section
    assert "| 46" not in section
