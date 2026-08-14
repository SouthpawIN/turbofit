from __future__ import annotations

import hashlib
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
    def __init__(self, fail: bool = False, recipe: str = "sha256:recipe-a") -> None:
        self.fail = fail
        self.recipe = recipe
        self.rows: list[str] = []

    def recipe_sha256(self, item: MatrixRow) -> str:
        return self.recipe

    def evidence_is_current(self, record: dict) -> bool:
        return True

    def current_physical_fingerprint(self) -> str:
        return "sha256:" + "c" * 64

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
            physical_fingerprint="sha256:" + "a" * 64,
            raw_result_sha256="sha256:" + "b" * 64,
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
    assert state["rows"][item.id]["raw_result_sha256"] == "sha256:" + "b" * 64
    assert state["rows"][item.id]["physical_fingerprint"] == "sha256:" + "a" * 64


def test_campaign_lease_surrounds_both_gpu_clear_gates(tmp_path: Path) -> None:
    events = []

    class LeaseExecutor(FakeExecutor):
        def prepare(self, item):
            events.append("prepare")

        def execute(self, item):
            events.append("execute")
            return super().execute(item)

        def finish(self, item):
            events.append("finish")

    class OrderedClear(FakeClearGate):
        def wait(self, **kwargs):
            events.append(kwargs["label"].split("-", 1)[0])
            return super().wait(**kwargs)

    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=LeaseExecutor(), clear_gate=OrderedClear(),
        publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert runner.run_one().status == "success"
    assert events == ["prepare", "before", "execute", "after", "finish"]


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


def test_repeated_failure_backs_off_without_becoming_resolved(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=FakeExecutor(fail=True), clear_gate=FakeClearGate(),
        publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert runner.run_one().status == "failed"
    assert runner.run_one().status == "failed"
    state = json.loads((tmp_path / "state.json").read_text())["rows"][item.id]
    assert state["failure_class"] == "retryable-runtime"
    assert state["next_retry_at"]
    assert runner.pending_rows() == []
    assert [row.id for row in runner.unresolved_rows()] == [item.id]


def test_resume_skips_rows_already_successful_in_state(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    first = row("Carwin Nano", "auto", 65_536)
    second = row("Carwin Nano", "auto", 131_072)
    write_matrix(matrix_path, (first, second))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {first.id: {
        "status": "success", "attempts": 1, "recipe_sha256": "sha256:recipe-a",
    }}}))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=FakeExecutor(), clear_gate=FakeClearGate(), publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert [item.id for item in runner.pending_rows()] == [second.id]


def test_deferred_rows_are_not_scheduled_but_remain_in_the_matrix(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    grm = row("GRM 2.6 Plus", "auto", 65_536)
    qwen = row("Qwen3.8 27B", "auto", 65_536)
    write_matrix(matrix_path, (grm, qwen))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=tmp_path / "state.json",
        executor=FakeExecutor(), clear_gate=FakeClearGate(),
        publisher=FakePublisher(), registry=FakeRegistry(),
        deferred_row_ids=frozenset({grm.id}),
    )

    assert [item.id for item in runner.pending_rows()] == [qwen.id]
    assert len(json.loads(matrix_path.read_text())["rows"]) == 2


def test_recipe_change_requeues_previous_success_and_resets_attempts(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {item.id: {
        "status": "success", "attempts": 7, "recipe_sha256": "sha256:old",
    }}}))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=FakeExecutor(recipe="sha256:new"), clear_gate=FakeClearGate(),
        publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert [pending.id for pending in runner.pending_rows()] == [item.id]
    assert runner.run_one().status == "success"
    state = json.loads(state_path.read_text())
    assert state["rows"][item.id]["attempts"] == 1
    assert state["rows"][item.id]["recipe_sha256"] == "sha256:new"


def test_missing_or_mutated_success_evidence_is_requeued(tmp_path: Path) -> None:
    class EvidenceCheckingExecutor(FakeExecutor):
        def evidence_is_current(self, record: dict) -> bool:
            return False

    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {item.id: {
        "status": "success", "attempts": 1, "recipe_sha256": "sha256:recipe-a",
        "raw_result_path": "/missing/result.json",
    }}}))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=EvidenceCheckingExecutor(), clear_gate=FakeClearGate(),
        publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert [pending.id for pending in runner.pending_rows()] == [item.id]


def test_hardware_incompatibility_requires_hash_bound_physical_proof(tmp_path: Path) -> None:
    fingerprint = "sha256:" + "c" * 64

    class PhysicalExecutor(FakeExecutor):
        def current_physical_fingerprint(self) -> str:
            return fingerprint

    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {item.id: {
        "status": "failed", "attempts": 2, "recipe_sha256": "sha256:recipe-a",
        "error": "allocation failed",
    }}}))
    failure = tmp_path / "failure.json"
    failure.write_text("physical allocation failure\n")
    failure_sha = "sha256:" + hashlib.sha256(failure.read_bytes()).hexdigest()
    evidence = tmp_path / "incompatibility.json"
    evidence.write_text(json.dumps({
        "schema": "turbofit.hardware-incompatibility/v1",
        "row_id": item.id,
        "production_recipe_sha256": "sha256:recipe-a",
        "physical_fingerprint": fingerprint,
        "failure_evidence": str(failure),
        "failure_evidence_sha256": failure_sha,
        "required_memory_mb": {"0": 26000},
        "available_memory_mb": {"0": 24576},
        "reason": "pinned recipe exceeds physical device memory",
    }))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=PhysicalExecutor(), clear_gate=FakeClearGate(),
        publisher=FakePublisher(), registry=FakeRegistry(),
    )

    runner.mark_hardware_incompatible(item.id, evidence)
    record = json.loads(state_path.read_text())["rows"][item.id]
    assert record["status"] == "hardware-incompatible"
    assert runner.unresolved_rows() == []


def test_retry_failed_preserves_failure_history_and_requeues_row(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    item = row("Carwin Nano", "auto", 65_536)
    write_matrix(matrix_path, (item,))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "schema_version": 1,
        "rows": {item.id: {
            "status": "failed", "attempts": 2, "error": "old OOM",
            "recipe_sha256": "sha256:recipe-a",
        }},
    }))
    runner = CampaignRunner(
        matrix_path=matrix_path,
        state_path=state_path,
        executor=FakeExecutor(),
        clear_gate=FakeClearGate(),
        publisher=FakePublisher(),
        registry=FakeRegistry(),
    )

    assert runner.retry_failed(limit=1) == (item.id,)
    state = json.loads(state_path.read_text())
    assert state["rows"][item.id]["status"] == "pending"
    assert state["rows"][item.id]["attempts"] == 0
    assert state["rows"][item.id]["history"] == [
        {"status": "failed", "attempts": 2, "error": "old OOM"}
    ]
    assert [pending.id for pending in runner.pending_rows()] == [item.id]


def test_run_one_returns_complete_when_nothing_is_pending(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.json"
    successful = MatrixRow(
        id="carwin-nano-auto-64k", main="Carwin Nano", aux="auto", context=65_536,
        status="success", method_priority=("dspark", "mtp", "nextn"),
    )
    write_matrix(matrix_path, (successful,))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"schema_version": 1, "rows": {successful.id: {
        "status": "success", "attempts": 1, "recipe_sha256": "sha256:recipe-a",
    }}}))
    runner = CampaignRunner(
        matrix_path=matrix_path, state_path=state_path,
        executor=FakeExecutor(), clear_gate=FakeClearGate(), publisher=FakePublisher(), registry=FakeRegistry(),
    )

    assert runner.run_one().status == "complete"
