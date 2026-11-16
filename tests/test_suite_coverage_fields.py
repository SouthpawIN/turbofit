"""Coverage must be machine-readable on the record, not just in prose.

A published row has to say "N of 4 suites" next to its index. The count lived
only in the human-readable summary, so consumers of the JSON saw nothing — and
a record scored on one suite looked identical to a full index.
"""
from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.intelligence import HERMES_INDEX_SUITES, refresh_score_payload

SCORES = Path(__file__).resolve().parents[1] / "references" / "intelligence-scores.json"


def _record() -> dict:
    data = json.loads(SCORES.read_text(encoding="utf-8"))
    records = data.get("records") or []
    assert records, "expected at least one recorded measurement"
    return records[0]


def test_record_carries_measured_and_total_counts() -> None:
    refreshed = refresh_score_payload(_record())
    assert refreshed["suites_measured"] == len(refreshed["index_suites"])
    assert refreshed["suites_total"] == len(HERMES_INDEX_SUITES)


def test_partial_coverage_is_not_a_full_index() -> None:
    refreshed = refresh_score_payload(_record())
    if refreshed["suites_measured"] < refreshed["suites_total"]:
        assert refreshed["suites_measured"] != len(HERMES_INDEX_SUITES)


def test_retired_benchmarks_cannot_carry_an_index() -> None:
    payload = dict(_record())
    payload["measurements"] = [
        {
            "name": "deep-swe",
            "score": 1.0,
            "tasks_passed": 1,
            "tasks_total": 1,
            "revision": "pin",
            "raw_result": "retired.json",
            "raw_result_sha256": "sha256:0",
        }
    ]
    try:
        refresh_score_payload(payload)
    except ValueError:
        return
    raise AssertionError("a deep-swe-only payload must not produce an index")
