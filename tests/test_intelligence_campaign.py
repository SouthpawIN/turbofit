from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.intelligence_campaign import (
    LEVELS,
    apply_deferred_configurations,
    build_state,
    complete_run,
    mark_running,
    recover_interrupted,
    reconcile_state,
    refresh_runtime_status,
    next_ready_run,
)


def matrix() -> dict:
    return {
        "rows": [
            {"id": "a--auto--64k", "main": "a", "auxiliary": "auto", "context": 65536},
            {"id": "b--aux--128k", "main": "b", "auxiliary": "aux", "context": 131072},
        ]
    }


def test_intelligence_campaign_expands_every_configuration_into_reproducible_levels() -> None:
    state = build_state(matrix())

    assert len(state["runs"]) == 2 * len(LEVELS)
    assert {run["level"] for run in state["runs"].values()} == set(LEVELS)
    assert all(
        run["status"] == ("waiting-runtime" if run["level"] == "screening" else f"waiting-{LEVELS[LEVELS.index(run['level']) - 1]}")
        for run in state["runs"].values()
    )


def test_only_physically_successful_runtime_rows_become_benchmark_ready() -> None:
    state = build_state(matrix())
    runtime = {
        "rows": {
            "a-auto-64k": {"status": "success"},
            "b-aux-128k": {"status": "failed"},
        }
    }
    display_ids = {"a--auto--64k": "a-auto-64k", "b--aux--128k": "b-aux-128k"}

    refresh_runtime_status(state, runtime, display_ids)

    assert state["runs"]["a--auto--64k::screening"]["status"] == "ready"
    assert state["runs"]["b--aux--128k::screening"]["status"] == "blocked-runtime"
    assert state["runs"]["b--aux--128k::promotion"]["status"] == "blocked-screening"
    assert state["runs"]["b--aux--128k::release"]["status"] == "blocked-promotion"
    assert state["runs"]["a--auto--64k::promotion"]["status"] == "waiting-screening"


def test_failed_screening_terminates_dependent_levels():
    state = build_state({
        "rows": [{"id": "cfg-a", "main": "a", "auxiliary": "auto", "context": 65536}]
    })
    state["runs"]["cfg-a::screening"]["status"] = "ready"
    mark_running(state, "cfg-a::screening")
    complete_run(state, "cfg-a::screening", result=None, error="benchmark crashed")

    assert state["runs"]["cfg-a::screening"]["status"] == "failed"
    assert state["runs"]["cfg-a::promotion"]["status"] == "blocked-screening"
    assert state["runs"]["cfg-a::release"]["status"] == "blocked-promotion"


def test_recipe_change_invalidates_old_intelligence_and_reopens_dependencies() -> None:
    state = build_state({
        "rows": [{"id": "cfg-a", "main": "a", "auxiliary": "auto", "context": 65536}]
    })
    state["runs"]["cfg-a::screening"].update(status="success", result="old-screening.json")
    state["runs"]["cfg-a::promotion"].update(status="ready")
    runtime = {"rows": {"cfg-a-display": {
        "status": "success", "recipe_sha256": "sha256:old",
    }}}

    refresh_runtime_status(
        state, runtime, {"cfg-a": "cfg-a-display"}, {"cfg-a": "sha256:new"},
    )

    assert state["runs"]["cfg-a::screening"]["status"] == "waiting-runtime"
    assert state["runs"]["cfg-a::screening"]["result"] is None
    assert state["runs"]["cfg-a::promotion"]["status"] == "waiting-screening"
    assert state["runs"]["cfg-a::release"]["status"] == "waiting-promotion"


def test_interrupted_worker_is_requeued_without_losing_attempt_history():
    state = build_state({
        "rows": [{"id": "cfg-a", "main": "a", "auxiliary": "auto", "context": 65536}]
    })
    state["runs"]["cfg-a::screening"]["status"] = "ready"
    run = mark_running(state, "cfg-a::screening")
    run["worker_pid"] = 999999

    recovered = recover_interrupted(state, active_pids=set())

    assert recovered == ("cfg-a::screening",)
    assert run["status"] == "ready"
    assert run["attempts"] == 1
    assert "interrupted" in run["error"]


def test_campaign_finishes_screening_before_promotion_and_release() -> None:
    state = build_state(matrix())
    first = state["runs"]["a--auto--64k::screening"]
    first["status"] = "success"
    second = state["runs"]["a--auto--64k::promotion"]
    second["status"] = "ready"
    state["runs"]["b--aux--128k::screening"]["status"] = "ready"

    selected = next_ready_run(state)

    assert selected["level"] == "screening"
    assert selected["configuration_id"] == "b--aux--128k"


def test_deferred_configuration_remains_accounted_and_can_be_reopened() -> None:
    state = build_state(matrix())

    apply_deferred_configurations(
        state, {"b--aux--128k"}, reason="replacement arrives in two days",
    )

    assert state["runs"]["b--aux--128k::screening"]["status"] == "deferred-runtime"
    assert state["runs"]["b--aux--128k::promotion"]["status"] == "deferred-screening"
    assert state["runs"]["b--aux--128k::release"]["status"] == "deferred-promotion"
    assert next_ready_run(state) is None

    apply_deferred_configurations(state, set(), reason="unused")

    assert state["runs"]["b--aux--128k::screening"]["status"] == "waiting-runtime"
    assert state["runs"]["b--aux--128k::promotion"]["status"] == "waiting-screening"
    assert state["runs"]["b--aux--128k::release"]["status"] == "waiting-promotion"


def test_reconcile_state_preserves_retained_runs_and_adds_replacement_rows() -> None:
    state = build_state(matrix())
    state["runs"]["a--auto--64k::screening"]["status"] = "success"
    replacement = {
        "rows": [
            {"id": "a--auto--64k", "main": "a", "auxiliary": "auto", "context": 65536},
            {"id": "qwen--auto--64k", "main": "qwen", "auxiliary": "auto", "context": 65536},
        ]
    }

    reconciled = reconcile_state(state, replacement)

    assert reconciled["runs"]["a--auto--64k::screening"]["status"] == "success"
    assert reconciled["runs"]["qwen--auto--64k::screening"]["status"] == "waiting-runtime"
    assert not any(key.startswith("b--aux--128k::") for key in reconciled["runs"])
