"""The index is the four Hermes suites — provenance tests for that switch.

DeepSWE and the agentic pair remain usable harnesses, but nothing in the shipped
repository may turn their numbers into an index. These tests pin the wiring:
the campaign runs the Hermes suites, the level parameters describe suites, and
the committed score file archives the retired record instead of publishing it.
"""
from __future__ import annotations

import json
from pathlib import Path

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


def test_committed_scores_publish_nothing_and_archive_the_retired_record() -> None:
    payload = json.loads(SCORES.read_text(encoding="utf-8"))

    assert payload["index_suites"] == list(HERMES_INDEX_SUITES)
    assert payload["records"] == []
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

    assert build_rows(scores) == []
    section = render_section(scores, [48])
    assert "no measurements yet" in section
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
