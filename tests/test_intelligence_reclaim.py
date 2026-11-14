"""An environment failure must not poison a benchmark run forever."""

from turbofit_runtime.intelligence_campaign import reclaim_environmental_failures


def _state(*runs):
    return {"runs": {run["id"]: run for run in runs}, "updated_at": None}


def _run(run_id, *, status="failed", attempts=1, error=""):
    return {"id": run_id, "status": status, "attempts": attempts, "error": error}


def test_missing_installer_is_requeued():
    state = _state(_run("a::screening", error="FileNotFoundError: [Errno 2] No such file or directory: 'uv'"))
    assert reclaim_environmental_failures(state) == ("a::screening",)
    assert state["runs"]["a::screening"]["status"] == "ready"
    assert state["runs"]["a::screening"]["error"] is None


def test_docker_and_prepare_failures_are_requeued():
    state = _state(
        _run("b::screening", error="prepare returned non-zero exit status 1"),
        _run("c::screening", error="Cannot connect to the Docker daemon"),
    )
    assert set(reclaim_environmental_failures(state)) == {"b::screening", "c::screening"}


def test_model_failure_is_not_requeued():
    state = _state(_run("d::screening", error="score 0.41 below the promotion threshold"))
    assert reclaim_environmental_failures(state) == ()
    assert state["runs"]["d::screening"]["status"] == "failed"


def test_exhausted_attempts_stay_failed():
    state = _state(_run("e::screening", attempts=3, error="FileNotFoundError: 'uv'"))
    assert reclaim_environmental_failures(state) == ()
    assert state["runs"]["e::screening"]["status"] == "failed"


def test_non_failed_runs_are_untouched():
    state = _state(_run("f::screening", status="success", attempts=1, error="No such file"))
    assert reclaim_environmental_failures(state) == ()
    assert state["runs"]["f::screening"]["status"] == "success"
