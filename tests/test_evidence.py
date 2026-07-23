from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.evidence import BenchmarkResult, EvidencePublisher, IncompleteBenchmark
from turbofit_runtime.gpu import GPUClearEvent, GPUSample
from turbofit_runtime.schema import Matrix, MatrixRow


def make_row() -> MatrixRow:
    return MatrixRow(
        id="a-b-64k",
        main="A",
        aux="B",
        context=65_536,
        status="pending",
        method_priority=("dspark", "mtp", "nextn"),
    )


def make_result() -> BenchmarkResult:
    clear = GPUClearEvent(
        timestamp="2026-07-23T00:00:00+00:00",
        label="after-a-b-64k",
        passed=True,
        ceilings_mb={0: 1024, 1: 1024},
        snapshot=(
            GPUSample(gpu=0, total_mb=24576, used_mb=500, free_mb=24076, utilization_pct=0),
            GPUSample(gpu=1, total_mb=24576, used_mb=60, free_mb=24516, utilization_pct=0),
        ),
        samples_observed=3,
    )
    return BenchmarkResult(
        row_id="a-b-64k",
        method="mtp",
        exact_context=True,
        main_health=True,
        aux_health=True,
        main_output="main ok",
        aux_output="aux ok",
        main_tps=40.5,
        aux_tps=80.25,
        gpu_peak_mb={0: 14000, 1: 18000},
        runtime_string="turbofit-runtime use a-b-64k",
        gpu_clear_after=clear,
        raw_result_path="references/results/a-b-64k.json",
    )


def test_publish_success_updates_matrix_checklist_and_evidence(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(Matrix((row,)).to_dict(), indent=2))
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "# Matrix\n\n### Success index\n\n"
        '<a id="a-b-64k"></a>\n- [ ] **A:B @ 64K context**\n'
    )
    evidence_dir = tmp_path / "evidence"

    evidence_path = EvidencePublisher(
        matrix_path=matrix_path,
        checklist_path=checklist,
        evidence_dir=evidence_dir,
    ).publish_success(row, make_result())

    matrix = json.loads(matrix_path.read_text())
    assert matrix["rows"][0]["status"] == "success"
    text = checklist.read_text()
    assert "- [x] **A:B @ 64K context**" in text
    assert "[evidence](evidence/a-b-64k.md)" in text
    assert text.count("a-b-64k.md") == 2
    evidence = evidence_path.read_text()
    assert "40.50 tok/s" in evidence
    assert "80.25 tok/s" in evidence
    assert "GPU-clear gate: `PASS`" in evidence
    assert "turbofit-runtime use a-b-64k" in evidence


def test_publish_success_is_idempotent(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(Matrix((row,)).to_dict(), indent=2))
    checklist = tmp_path / "checklist.md"
    checklist.write_text("### Success index\n\n<a id=\"a-b-64k\"></a>\n- [ ] **A:B @ 64K context**\n")
    publisher = EvidencePublisher(matrix_path=matrix_path, checklist_path=checklist, evidence_dir=tmp_path / "evidence")

    publisher.publish_success(row, make_result())
    first = checklist.read_text()
    publisher.publish_success(row, make_result())

    assert checklist.read_text() == first


def test_incomplete_result_does_not_promote_row(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    original = json.dumps(Matrix((row,)).to_dict(), indent=2)
    matrix_path.write_text(original)
    checklist = tmp_path / "checklist.md"
    checklist.write_text('<a id="a-b-64k"></a>\n- [ ] **A:B @ 64K context**\n')
    bad = make_result()
    bad = BenchmarkResult(**{**bad.__dict__, "aux_output": ""})

    with pytest.raises(IncompleteBenchmark, match="aux output"):
        EvidencePublisher(matrix_path=matrix_path, checklist_path=checklist, evidence_dir=tmp_path / "evidence").publish_success(row, bad)

    assert matrix_path.read_text() == original
    assert "[ ]" in checklist.read_text()
    assert not (tmp_path / "evidence").exists()
