"""The doctor's judgements: what counts as broken, and what it may repair."""

from __future__ import annotations

from turbofit_runtime import doctor


def test_health_that_promises_local_while_an_api_answers_is_a_failure():
    finding = doctor.classify_health(
        {"ok": True, "main": "ready", "local": False, "cloud_fallback": True}
    )

    assert finding.status == doctor.FAIL
    assert "an API is answering" in finding.detail


def test_a_yield_is_not_a_failure_it_is_the_feature_working():
    finding = doctor.classify_health(
        {"ok": True, "main": "yielded", "local": False, "cloud_fallback": True}
    )

    assert finding.status == doctor.OK
    assert "yielded" in finding.detail


def test_a_real_local_backend_is_ok():
    finding = doctor.classify_health(
        {"ok": True, "main": "ready", "local": True, "cloud_fallback": False,
         "backend": "local"}
    )

    assert finding.status == doctor.OK


def test_a_dead_gateway_is_a_failure_with_a_repair():
    finding = doctor.classify_health(None)

    assert finding.status == doctor.FAIL
    assert finding.repair == "restart"


def test_a_unit_that_launches_the_script_directly_needs_the_runner():
    unit = """[Service]
ExecStart=/home/u/turbofit/scripts/turbofit-gateway.py
"""
    assert doctor.unit_needs_runner(unit) is True

    rewritten = doctor.rewrite_unit(unit, "/home/u/turbofit/scripts/turbofit-run")
    assert "ExecStart=/home/u/turbofit/scripts/turbofit-run /home/u/turbofit/scripts/turbofit-gateway.py" in rewritten
    assert doctor.unit_needs_runner(rewritten) is False


def test_rewriting_is_idempotent_and_keeps_arguments():
    unit = "ExecStart=/x/turbofit-run /x/scripts/turbofit-controller --interval 2"
    assert doctor.rewrite_unit(unit, "/x/turbofit-run") == unit


def test_a_declared_rung_that_is_not_listening_is_dead():
    finding = doctor.diagnose_runtime_state(
        declared={"rung_id": "iq3s-262k", "port": 8082}, listening=False, released=False
    )

    assert finding.status == doctor.FAIL
    assert "8082" in finding.detail


def test_a_released_rung_is_expected_to_be_down():
    finding = doctor.diagnose_runtime_state(
        declared={"rung_id": "iq3s-262k", "port": 8082}, listening=False, released=True
    )

    assert finding.status == doctor.OK
    assert "released" in finding.detail


def test_a_serving_rung_is_ok():
    finding = doctor.diagnose_runtime_state(
        declared={"rung_id": "iq3s-262k", "port": 8082}, listening=True, released=False
    )

    assert finding.status == doctor.OK


def test_no_selected_rung_yet_is_a_warning_not_a_failure():
    assert doctor.diagnose_runtime_state(declared=None, listening=False,
                                         released=False).status == doctor.WARN


def test_an_interpreter_without_psutil_loses_the_supervisor():
    finding = doctor.diagnose_interpreter(
        python="/usr/bin/python3", has_psutil=False, has_hermes=True, adopted=8, total=10
    )

    assert finding.status == doctor.WARN
    assert finding.repair == "python"
    assert "psutil" in finding.detail


def test_full_adoption_is_ok():
    finding = doctor.diagnose_interpreter(
        python="/h/.venv/bin/python", has_psutil=True, has_hermes=True, adopted=10, total=10
    )

    assert finding.status == doctor.OK
    assert "10/10" in finding.detail


def test_a_mixed_strata_install_is_a_warning():
    finding = doctor.diagnose_strata(installed="0.2.10", problems=["engine is 0.1.55"])
    assert finding.status == doctor.WARN
    assert "0.1.55" in finding.detail


def test_strata_absent_is_not_a_problem():
    assert doctor.diagnose_strata(installed=None, problems=[]).status == doctor.OK


def test_a_missing_engine_pin_is_reported():
    finding = doctor.diagnose_engine_pins(expected="mainline-llama.cpp", present=["prism-llama.cpp"])
    assert finding.status == doctor.WARN
    assert "mainline-llama.cpp" in finding.detail


def test_the_worst_finding_wins_and_repairs_are_listed():
    diagnosis = doctor.Diagnosis(findings=[
        doctor.Finding("a", doctor.OK, "fine"),
        doctor.Finding("b", doctor.WARN, "meh", repair="unit"),
        doctor.Finding("c", doctor.FAIL, "broken"),
    ])

    assert diagnosis.worst == doctor.FAIL
    assert [f.name for f in diagnosis.repairs()] == ["b"]
    assert "1 failing" in doctor.summarise(diagnosis)
    assert diagnosis.to_mapping()["repairs_available"] == ["b"]
