from __future__ import annotations

import importlib.util
import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "turbofit-deepswe"


def _module():
    spec = importlib.util.spec_from_file_location(
        "turbofit_deepswe",
        SCRIPT,
        loader=SourceFileLoader("turbofit_deepswe", str(SCRIPT)),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deepswe_harness_is_reproducibly_pinned() -> None:
    module = _module()

    assert module.DEEPSWE_REVISION == "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
    assert module.PIER_VERSION == "0.3.0"


def test_local_endpoint_command_uses_mini_swe_agent_and_deterministic_subset(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "host_route_address", lambda: "192.168.1.172")

    command, environment = module.pier_command(
        dataset=tmp_path / "deep-swe" / "tasks",
        model="deepseek-v4-flash-0731",
        base_url="http://127.0.0.1:11608/v1",
        jobs_dir=tmp_path / "jobs",
        n_tasks=10,
        sample_seed=17,
        agent_step_limit=60,
        max_output_tokens=1024,
    )

    assert command == [
        "pier", "run", "-p", str(tmp_path / "deep-swe" / "tasks"),
        "--agent-import-path", "turbofit_runtime.pier_agent:TurbofitMiniSWEAgent",
        "--model", "openai/deepseek-v4-flash-0731",
        "--agent-kwarg", 'model_kwargs={"num_retries":0,"timeout":300,"max_tokens":1024}',
        "--agent-kwarg", "step_limit=60",
        "--agent-env", "OPENAI_BASE_URL=http://192.168.1.172:11608/v1",
        "--agent-env", "OPENAI_API_BASE=http://192.168.1.172:11608/v1",
        "--agent-env", "OPENAI_API_KEY=turbofit-local",
        "--n-tasks", "10",
        "--sample-seed", "17",
        "--n-concurrent", "1",
        "--jobs-dir", str(tmp_path / "jobs"),
        "--yes",
    ]
    assert environment["OPENAI_BASE_URL"] == "http://192.168.1.172:11608/v1"
    assert environment["OPENAI_API_BASE"] == "http://192.168.1.172:11608/v1"
    assert environment["OPENAI_API_KEY"] == "turbofit-local"
    assert str(module.PROJECT_SRC) in environment["PYTHONPATH"].split(os.pathsep)


def test_deepswe_preserves_non_loopback_endpoint_and_supports_gateway_override(monkeypatch) -> None:
    module = _module()
    assert module.container_base_url("http://192.168.1.7:8091/v1/") == "http://192.168.1.7:8091/v1"
    monkeypatch.setenv("TURBOFIT_DOCKER_HOST_GATEWAY", "172.19.0.1")
    assert module.container_base_url("http://localhost:8091/v1") == "http://172.19.0.1:8091/v1"


def test_deepswe_summary_is_hash_bound_and_scores_actual_verifier_rewards(tmp_path: Path) -> None:
    module = _module()
    jobs = tmp_path / "jobs"
    passing = jobs / "job" / "trial-1"
    failing = jobs / "job" / "trial-2"
    passing.mkdir(parents=True)
    failing.mkdir(parents=True)
    route = '{"schema":"turbofit.pier-container-route/v1","bridge_gateway":"172.20.0.1","provider_base_url":"http://172.20.0.1:18092/main/v1"}'
    for trial in (passing, failing):
        (trial / "agent").mkdir()
        (trial / "agent" / "turbofit-route.json").write_text(route)
        (trial / "agent" / "trajectory.json").write_text(
            '{"steps":[{"llm_call_count":2},{"llm_call_count":1}]}'
        )
    (jobs / "job" / "result.json").write_text('{"n_total_trials":2,"stats":{"n_completed_trials":2}}')
    agent = '"agent_result":{"n_agent_steps":2,"n_input_tokens":100,"n_output_tokens":20}'
    (passing / "result.json").write_text('{"verifier_result":{"rewards":{"reward":1.0}},"exception_info":null,' + agent + '}')
    (failing / "result.json").write_text('{"verifier_result":{"rewards":{"reward":0.0}},"exception_info":null,' + agent + '}')

    result = module.summarize_job(
        jobs_dir=jobs,
        configuration_id="main--aux--128k",
        production_recipe_sha256="sha256:" + "a" * 64,
        output=tmp_path / "summary.json",
    )

    assert result["status"] == "measured"
    assert result["summary"]["tasks_total"] == 2
    assert result["summary"]["tasks_passed"] == 1
    assert result["summary"]["score"] == 0.5
    assert result["summary"]["infrastructure_failures"] == 0
    assert result["evidence_sha256"].startswith("sha256:")
    assert len(result["trials"]) == 2
    assert all(item["model_calls"] == 3 for item in result["trials"])
    assert result["summary"]["model_calls"] == 6
    assert result["summary"]["input_tokens"] == 200
    assert result["summary"]["output_tokens"] == 40
    assert all(item["route_evidence"]["sha256"].startswith("sha256:") for item in result["trials"])


def test_deepswe_rejects_model_tokens_without_container_route_proof(tmp_path: Path) -> None:
    module = _module()
    trial = tmp_path / "jobs" / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        '{"verifier_result":{"rewards":{"reward":1}},"exception_info":null,'
        '"agent_result":{"n_agent_steps":2,"n_input_tokens":100,"n_output_tokens":20}}'
    )
    with pytest.raises(RuntimeError, match="without model execution"):
        module.summarize_job(
            jobs_dir=tmp_path / "jobs",
            configuration_id="main--aux--64k",
            production_recipe_sha256="sha256:" + "c" * 64,
            output=tmp_path / "invalid-route.json",
        )


def test_deepswe_rejects_tokens_without_proven_model_calls(tmp_path: Path) -> None:
    module = _module()
    trial = tmp_path / "jobs" / "job" / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "turbofit-route.json").write_text(
        '{"schema":"turbofit.pier-container-route/v1","bridge_gateway":"172.20.0.1",'
        '"provider_base_url":"http://172.20.0.1:18092/main/v1"}'
    )
    (trial / "agent" / "trajectory.json").write_text('{"steps":[]}')
    (trial / "result.json").write_text(
        '{"verifier_result":{"rewards":{"reward":0}},"exception_info":null,'
        '"agent_result":{"n_agent_steps":2,"n_input_tokens":100,"n_output_tokens":20}}'
    )

    with pytest.raises(RuntimeError, match="without model execution"):
        module.summarize_job(
            jobs_dir=tmp_path / "jobs",
            configuration_id="main--aux--64k",
            production_recipe_sha256="sha256:" + "d" * 64,
            output=tmp_path / "invalid-calls.json",
        )


def test_deepswe_rejects_zero_token_infrastructure_failures(tmp_path: Path) -> None:
    module = _module()
    trial = tmp_path / "jobs" / "job" / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        '{"verifier_result":{"rewards":{"reward":0}},'
        '"agent_result":{"n_agent_steps":0,"n_input_tokens":0,"n_output_tokens":0},'
        '"exception_info":{"exception_type":"NonZeroAgentExitCodeError"}}'
    )
    output = tmp_path / "invalid.json"

    with pytest.raises(RuntimeError, match="without model execution"):
        module.summarize_job(
            jobs_dir=tmp_path / "jobs",
            configuration_id="main--aux--64k",
            production_recipe_sha256="sha256:" + "b" * 64,
            output=output,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "invalid"
    assert payload["summary"]["infrastructure_failures"] == 1
