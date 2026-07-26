from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.campaign import CampaignRunner, RawBenchmark, campaign_order
from turbofit_runtime.evidence import BenchmarkResult
from turbofit_runtime.gpu import GPUClearEvent, GPUSample
from turbofit_runtime.schema import Matrix, MatrixRow


def row(main: str, aux: str, context: int) -> MatrixRow:
    return MatrixRow(
        id=MatrixRow.make_id(main, aux, context),
        main=main,
        aux=aux,
        context=context,
        status="pending",
        method_priority=("dspark", "mtp", "nextn"),
    )


def clear_event(label: str) -> GPUClearEvent:
    return GPUClearEvent(
        timestamp="2026-07-23T00:00:00+00:00",
        label=label,
        passed=True,
        ceilings_mb={0: 1024, 1: 1024},
        snapshot=(GPUSample(gpu=0, total_mb=24576, used_mb=500, free_mb=24076, utilization_pct=0),),
        samples_observed=3,
    )


class FakeClearGate:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def wait(self, **kwargs) -> GPUClearEvent:
        self.labels.append(kwargs["label"])
        return clear_event(kwargs["label"])


class FakeExecutor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[str] = []

    def execute(self, item: MatrixRow) -> RawBenchmark:
        self.rows.append(item.id)
        if self.fail:
            raise RuntimeError("launch failed")
        return RawBenchmark(
            method="mtp",
            exact_context=True,
            main_health=True,
            aux_health=True,
            main_output="main",
            aux_output="aux",
            main_tps=40.0,
            aux_tps=80.0,
            gpu_peak_mb={0: 12000},
            runtime_string=f"turbofit-runtime use {item.id}",
            raw_result_path=f"references/results/{item.id}.json",
        )


class FakePublisher:
    def __init__(self) -> None:
        self.results: list[BenchmarkResult] = []

    def publish_success(self, item: MatrixRow, result: BenchmarkResult) -> Path:
        self.results.append(result)
        return Path(f"evidence/{item.id}.md")


class FakeRegistry:
    def __init__(self) -> None:
        self.rows: list[str] = []

    def register(self, item: MatrixRow, result: BenchmarkResult, evidence_path: Path) -> None:
        self.rows.append(item.id)


def write_matrix(path: Path, rows: tuple[MatrixRow, ...]) -> None:
    path.write_text(json.dumps(Matrix(rows).to_dict(), indent=2))


def test_campaign_order_is_bottom_up_then_context_ascending() -> None:
    rows = [
        row("GLM 5.2", "auto", 65_536),
        row("Carwin Nano", "auto", 262_144),
        row("Ternary Bonsai", "auto", 131_072),
        row("Carwin Nano", "auto", 65_536),
        row("1 Bit Bonsai", "auto", 65_536),
    ]

    ordered = sorted(rows, key=campaign_order)

    assert [item.main for item in ordered] == [
        "1 Bit Bonsai", "Ternary Bonsai", "Carwin Nano", "Carwin Nano", "GLM 5.2"
    ]
    assert [item.context for item in ordered[2:4]] == [65_536, 262_144]


def test_success_clears_before_and_after_publishes_and_registers(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    clear = FakeClearGate(); executor = FakeExecutor(); publisher = FakePublisher(); registry = FakeRegistry()
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=executor, clear_gate=clear, publisher=publisher, registry=registry,
    )

    outcome = runner.run_one()

    assert outcome.status == "success"
    assert clear.labels == [f"before-{item.id}", f"after-{item.id}"]
    assert executor.rows == [item.id]
    assert publisher.results[0].gpu_clear_after.label == f"after-{item.id}"
    assert registry.rows == [item.id]
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["rows"][item.id]["status"] == "success"
    assert state["rows"][item.id]["attempts"] == 1


def test_failure_still_clears_gpu_and_remains_retryable(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    clear = FakeClearGate(); executor = FakeExecutor(fail=True); publisher = FakePublisher(); registry = FakeRegistry()
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=executor, clear_gate=clear, publisher=publisher, registry=registry,
    )

    outcome = runner.run_one()

    assert outcome.status == "failed"
    assert clear.labels == [f"before-{item.id}", f"after-{item.id}"]
    assert publisher.results == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["rows"][item.id]["status"] == "failed"
    assert "launch failed" in state["rows"][item.id]["error"]
    assert runner.pending_rows()[0].id == item.id


def test_resume_skips_rows_already_successful_in_state(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    first = row("Carwin Nano", "auto", 65_536)
    second = row("Carwin Nano", "auto", 131_072)
    write_matrix(matrix_path, (first, second))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {first.id: {"status": "success", "attempts": 1}}}))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=FakeExecutor(), clear_gate=FakeClearGate(), publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert [item.id for item in runner.pending_rows()] == [second.id]


def test_run_one_returns_complete_when_nothing_is_pending(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    successful = MatrixRow(
        id="carwin-nano-auto-64k", main="Carwin Nano", aux="auto", context=65_536,
        status="success", method_priority=("dspark", "mtp", "nextn"),
    )
    write_matrix(matrix_path, (successful,))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=FakeExecutor(), clear_gate=FakeClearGate(), publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert runner.run_one().status == "complete"
