from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_rejects_non_finite_timeout() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "dashboard"))
    from smoke_ops import smoke_local_runtime

    with pytest.raises(ValueError, match="finite"):
        smoke_local_runtime(timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="timeout_seconds"):
        smoke_local_runtime(timeout_seconds=0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        smoke_local_runtime(timeout_seconds=901)


def test_smoke_persists_evidence_and_never_promotes(tmp_path, monkeypatch) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "dashboard"))
    import smoke_ops

    monkeypatch.setattr(smoke_ops, "_evidence_dir", lambda: tmp_path)

    def fake_run(**kwargs):
        assert kwargs["base_url"] == "http://127.0.0.1:8091/v1"
        assert kwargs["model"] == "auto"
        assert kwargs["require_gpu_samples"] is False
        Path(kwargs["output_path"]).write_text("{}\n", encoding="utf-8")
        return {
            "status": "pass",
            "evidence_sha256": "sha256:abc",
            "summary": {"effective_output_tokens_per_second": 12.0},
            "request_failures": [],
            "validator_failures": [],
            "resource_failures": [],
            "resource_warnings": ["gpu-memory-sampling-unavailable"],
        }

    monkeypatch.setattr(smoke_ops, "run_benchmark", fake_run)
    result = smoke_ops.smoke_local_runtime()

    assert result["ok"] is True
    assert result["promoted"] is False
    assert result["endpoint"] == "http://127.0.0.1:8091/v1"
    assert result["suite"] == "local-runtime-smoke-v1"
    assert result["resource_warnings"] == ["gpu-memory-sampling-unavailable"]
    assert Path(result["evidence_path"]).is_file()
    assert "shift" not in json.dumps(result)


def test_run_benchmark_warns_when_gpu_samples_are_optional(tmp_path, monkeypatch) -> None:
    from turbofit_runtime import benchmark_stage as stage
    from turbofit_runtime.benchmark_stage import BenchmarkSuite, run_benchmark

    monkeypatch.setattr(
        stage,
        "_chat_completion",
        lambda **_: {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        },
    )
    monkeypatch.setattr(stage, "evaluate", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(stage, "_gpu_memory_used_mib", lambda: ())
    monkeypatch.setattr(stage, "hardware_snapshot", lambda: {"os": "windows"})
    suite = BenchmarkSuite.load(ROOT / "benchmarks" / "stage-v1.json")

    warned = run_benchmark(
        suite=suite,
        base_url="http://127.0.0.1:8091/v1",
        model="auto",
        candidate="active:main",
        configuration="test",
        output_path=tmp_path / "warn.json",
        require_gpu_samples=False,
    )
    failed = run_benchmark(
        suite=suite,
        base_url="http://127.0.0.1:8091/v1",
        model="auto",
        candidate="active:main",
        configuration="test",
        output_path=tmp_path / "fail.json",
        require_gpu_samples=True,
    )

    assert warned["status"] == "pass"
    assert warned["resource_warnings"] == ["gpu-memory-sampling-unavailable"]
    assert warned["resource_failures"] == []
    assert failed["status"] == "fail"
    assert failed["resource_failures"] == ["gpu-memory-sampling-unavailable"]
