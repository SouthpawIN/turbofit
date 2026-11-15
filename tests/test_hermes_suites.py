"""Contract tests for the Hermes Index suite runners.

The runners exist so a TurboFit-served local model can be scored on the same
four suites Nous Research averages. These tests pin the three properties that
make that honest:

* every harness and dataset is pinned by exact version/revision,
* a suite that cannot run is ``unavailable`` with a reason — never a zero, and
* evidence is read back from the harness's own result files and SHA-256 bound,
  so a published pass@1 can be re-verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from turbofit_runtime.hermes_suites import (
    BENCHFLOW_VERSION,
    HARBOR_VERSION,
    SKILLSBENCH_DATASET,
    SKILLSBENCH_REVISION,
    TERMINAL_BENCH_DATASET,
    TERMINAL_BENCH_SCIENCE_DATASET,
    availability,
    harbor_command,
    index_block,
    measurement_from_evidence,
    probe_throughput,
    render_index_block,
    skillsbench_command,
    summarize_job,
)

SHA = "sha256:" + "a" * 64


def test_every_runnable_suite_is_pinned_by_harness_and_dataset() -> None:
    assert HARBOR_VERSION == "0.26.0"
    assert BENCHFLOW_VERSION == "0.6.3"
    assert TERMINAL_BENCH_DATASET == "terminal-bench/terminal-bench@4.1.0"
    assert TERMINAL_BENCH_SCIENCE_DATASET == "terminal-bench-science/terminal-bench-science@0.1.0"
    assert SKILLSBENCH_DATASET == "skillsbench@1.1"
    assert SKILLSBENCH_REVISION == "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"


def test_hermes_bench_is_unavailable_with_a_reason_not_a_zero() -> None:
    report = availability()

    hermes = report["suites"]["hermes-bench"]
    assert hermes["status"] == "unavailable"
    assert "nous-internal" in hermes["reason"]
    assert hermes["fix"]
    assert report["max_local_coverage"] == {"measured": 3, "total": 4}


def test_harbor_command_points_a_local_endpoint_at_the_host_route(monkeypatch) -> None:
    import turbofit_runtime.hermes_suites as module

    monkeypatch.setattr(module, "host_route_address", lambda: "192.168.1.172")
    command, environment = harbor_command(
        suite="terminal-bench-4",
        model="qwen3-8-27b",
        base_url="http://127.0.0.1:8092/v1",
        jobs_dir=Path("/tmp/jobs"),
        n_tasks=3,
        include=("terminal-bench/bun-sourcemap-leak",),
    )

    assert command[:2] == ["harbor", "run"]
    assert command[command.index("--dataset") + 1] == "terminal-bench/terminal-bench@4.1.0"
    assert command[command.index("--env") + 1] == "docker"
    assert command[command.index("--agent") + 1] == "mini-swe-agent"
    assert command[command.index("--model") + 1] == "openai/qwen3-8-27b"
    assert command[command.index("--base-url") + 1] == "http://192.168.1.172:8092/v1"
    assert command[command.index("--api-format") + 1] == "openai_chat_completions"
    assert command[command.index("--allow-agent-host") + 1] == "192.168.1.172"
    assert command[command.index("--include-task-name") + 1] == "terminal-bench/bun-sourcemap-leak"
    assert command[command.index("--n-tasks") + 1] == "3"
    assert environment["OPENAI_BASE_URL"] == "http://192.168.1.172:8092/v1"
    assert environment["OPENAI_API_KEY"] == "turbofit-local"


def test_harbor_command_refuses_the_unavailable_suite() -> None:
    with pytest.raises(RuntimeError, match="nous-internal"):
        harbor_command(
            suite="hermes-bench", model="m", base_url="http://127.0.0.1:1/v1",
            jobs_dir=Path("/tmp/j"), n_tasks=1,
        )


def test_skillsbench_command_runs_the_pinned_registry_dataset(monkeypatch) -> None:
    import turbofit_runtime.hermes_suites as module

    monkeypatch.setattr(module, "host_route_address", lambda: "10.0.0.5")
    command, environment = skillsbench_command(
        model="qwen3-8-27b", base_url="http://127.0.0.1:8092/v1",
        jobs_dir=Path("/tmp/jobs"), include=("offer-letter-generator",),
    )

    assert command[:4] == ["bench", "eval", "create", "--sandbox"]
    assert command[command.index("--dataset") + 1] == "skillsbench@1.1"
    assert command[command.index("--include") + 1] == "offer-letter-generator"
    assert "OPENAI_BASE_URL=http://10.0.0.5:8092/v1" in command
    assert environment["OPENAI_API_BASE"] == "http://10.0.0.5:8092/v1"


def _trial(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _harbor_trial(task: str, reward: float, *, exception=None, out_tokens=1000) -> dict:
    return {
        "task_name": f"terminal-bench/{task}",
        "trial_name": f"{task}__x",
        "started_at": "2026-11-15T08:38:44.403916Z",
        "finished_at": "2026-11-15T08:42:50.020385Z",
        "verifier_result": {"rewards": {"reward": reward}},
        "agent_result": {"n_input_tokens": 60002, "n_output_tokens": out_tokens},
        "exception_info": exception,
        "agent_info": {"name": "mini-swe-agent"},
    }


def _benchflow_trial(task: str, reward: float) -> dict:
    return {
        "task_name": task,
        "started_at": "2026-11-15 08:43:35.179625",
        "finished_at": "2026-11-15 08:47:44.708261",
        "rewards": {"reward": reward},
        "agent_result": {"n_input_tokens": 1200, "n_output_tokens": 400},
        "error": None,
    }


def test_harbor_evidence_scores_pass_at_1_and_binds_a_hash(tmp_path: Path) -> None:
    _trial(tmp_path, "a", _harbor_trial("bun-sourcemap-leak", 1.0))
    _trial(tmp_path, "b", _harbor_trial("batched-eval-parity", 0.0))
    _trial(tmp_path, "c", _harbor_trial("vllm-deepseek-streaming", 0.0))

    evidence = summarize_job(
        suite="terminal-bench-4", jobs_dir=tmp_path,
        configuration_id="qwen--none--64k", production_recipe_sha256=SHA,
    )

    assert evidence["status"] == "measured"
    assert evidence["summary"]["tasks_total"] == 3
    assert evidence["summary"]["tasks_passed"] == 1
    assert float(evidence["summary"]["score"]) == pytest.approx(1 / 3)
    assert evidence["summary"]["seconds_per_task"] == pytest.approx(245.6, abs=0.2)
    assert evidence["summary"]["tokens_per_second"] == pytest.approx(3000 / 736.848, rel=0.01)
    assert evidence["suite"]["revision"] == TERMINAL_BENCH_DATASET
    assert evidence["evidence_sha256"].startswith("sha256:")
    measurement = measurement_from_evidence(evidence)
    assert measurement.name == "terminal-bench-4"
    assert measurement.tasks_passed == 1 and measurement.tasks_total == 3


def test_benchflow_evidence_uses_the_same_trial_layout(tmp_path: Path) -> None:
    _trial(tmp_path, "a", _benchflow_trial("offer-letter-generator", 1.0))
    _trial(tmp_path, "b", _benchflow_trial("data-to-d3", 0.0))

    evidence = summarize_job(
        suite="skills-bench", jobs_dir=tmp_path,
        configuration_id="qwen--none--64k", production_recipe_sha256=SHA,
    )

    assert evidence["summary"]["tasks_total"] == 2
    assert evidence["summary"]["tasks_passed"] == 1
    assert evidence["suite"]["revision"] == SKILLSBENCH_REVISION


def test_a_trial_without_a_verifier_reward_invalidates_the_run(tmp_path: Path) -> None:
    _trial(tmp_path, "a", _harbor_trial("bun-sourcemap-leak", 1.0))
    broken = _harbor_trial("batched-eval-parity", 0.0)
    broken["verifier_result"] = None
    _trial(tmp_path, "b", broken)

    output = tmp_path / "evidence.json"
    with pytest.raises(RuntimeError, match="without a verifier reward"):
        summarize_job(
            suite="terminal-bench-4", jobs_dir=tmp_path,
            configuration_id="qwen--none--64k", production_recipe_sha256=SHA,
            output=output,
        )
    assert output.is_file()
    assert json.loads(output.read_text())["summary"]["infrastructure_failures"] == 1


def test_evidence_hash_is_stable_across_rereads(tmp_path: Path) -> None:
    _trial(tmp_path, "a", _harbor_trial("bun-sourcemap-leak", 1.0))
    first = summarize_job(
        suite="terminal-bench-4", jobs_dir=tmp_path,
        configuration_id="q", production_recipe_sha256=SHA,
    )
    second = summarize_job(
        suite="terminal-bench-4", jobs_dir=tmp_path,
        configuration_id="q", production_recipe_sha256=SHA,
    )
    canonical = {k: v for k, v in first.items() if k != "evidence_sha256"}
    digest = "sha256:" + hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["evidence_sha256"] == digest
    assert second["trials"][0]["sha256"] == first["trials"][0]["sha256"]


def test_index_block_averages_only_measured_hermes_suites_and_reports_n_of_4() -> None:
    measurements = [
        {"name": "terminal-bench-4", "score": 0.5, "tasks_total": 2, "tasks_passed": 1, "revision": "tb@4.1.0"},
        {"name": "skills-bench", "score": 1.0, "tasks_total": 4, "tasks_passed": 4, "revision": SKILLSBENCH_REVISION},
    ]

    block = index_block(
        model="qwen3-8-27b", configuration="qwen3-8-27b--none--64k",
        hardware_tier_gb=24, context=65_536, measurements=measurements,
        tokens_per_second=25.0,
    )

    assert block["index"] == pytest.approx(75.0)
    assert block["suites_measured"] == 2 and block["suites_total"] == 4
    assert block["seconds_per_task_1k"] == 40.0
    by_name = {row["suite"]: row for row in block["suites"]}
    assert by_name["hermes-bench"]["status"] == "unavailable"
    assert by_name["hermes-bench"]["pass_at_1"] is None
    assert by_name["hermes-bench"]["reason"].startswith("nous-internal")
    assert by_name["terminal-bench-science"]["status"] == "runnable"

    rendered = render_index_block(block)
    assert "2/4 suites" in rendered
    assert "| terminal-bench-4 | 50.00 | 1/2 | measured |" in rendered
    assert "unavailable (nous-internal)" in rendered


def test_index_block_has_no_index_without_a_hermes_suite() -> None:
    block = index_block(
        model="m", configuration="m--none--64k", hardware_tier_gb=24,
        context=65_536, measurements=[], tokens_per_second=None,
    )
    assert block["index"] is None
    assert block["suites_measured"] == 0
    assert "—" in render_index_block(block)


def test_probe_throughput_prefers_the_server_reported_decode_rate(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "ready"}}],
                "usage": {"completion_tokens": 2},
                "timings": {"predicted_per_second": 41.5},
            }).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    assert probe_throughput(base_url="http://127.0.0.1:8092/v1", model="m") == 41.5


def test_probe_throughput_refuses_an_unmeasurable_endpoint(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": []}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="no measurable decode throughput"):
        probe_throughput(base_url="http://127.0.0.1:8092/v1", model="m")
