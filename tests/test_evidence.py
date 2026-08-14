from __future__ import annotations

import hashlib
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


def make_result(tmp_path: Path) -> BenchmarkResult:
    fingerprint = "sha256:" + "a" * 64
    raw_path = tmp_path / "raw-result.json"
    raw_path.write_text(json.dumps({
        "physical_hardware": {"fingerprint_sha256": fingerprint},
    }))
    raw_sha = "sha256:" + hashlib.sha256(raw_path.read_bytes()).hexdigest()
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
        physical_fingerprint=fingerprint,
        raw_result_sha256=raw_sha,
        runtime_string="turbofit-runtime use a-b-64k",
        gpu_clear_after=clear,
        raw_result_path=str(raw_path),
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
    ).publish_success(row, make_result(tmp_path))

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

    publisher.publish_success(row, make_result(tmp_path))
    first = checklist.read_text()
    publisher.publish_success(row, make_result(tmp_path))

    assert checklist.read_text() == first


def test_publish_success_bootstraps_missing_checklist(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(Matrix((row,)).to_dict(), indent=2))
    checklist = tmp_path / "nested" / "checklist.md"

    EvidencePublisher(
        matrix_path=matrix_path,
        checklist_path=checklist,
        evidence_dir=tmp_path / "evidence",
    ).publish_success(row, make_result(tmp_path))

    assert checklist.is_file()
    assert "### Success index" in checklist.read_text()
    assert "- [x] **A:B @ 64K context**" in checklist.read_text()


def test_ensure_checklist_adds_new_matrix_rows_and_preserves_successes(tmp_path: Path) -> None:
    first = make_row()
    second = MatrixRow(
        id="c-d-128k",
        main="C",
        aux="D",
        context=131_072,
        status="pending",
        method_priority=("baseline",),
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(Matrix((first, second)).to_dict(), indent=2))
    checklist = tmp_path / "checklist.md"
    checklist.write_text(
        "### Success index\n\n"
        "- [A:B @ 64K](#a-b-64k) — evidence\n"
        '<a id="a-b-64k"></a>\n'
        "- [x] **A:B @ 64K context** — [evidence](evidence/a-b-64k.md)\n"
    )

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "a-b-64k.md").write_text("verified evidence\n")

    EvidencePublisher(
        matrix_path=matrix_path,
        checklist_path=checklist,
        evidence_dir=evidence_dir,
    ).ensure_checklist()

    text = checklist.read_text()
    assert "- [x] **A:B @ 64K context**" in text
    assert '<a id="c-d-128k"></a>' in text
    assert "- [ ] **C:D @ 128K context**" in text
    assert text.count("A:B @ 64K](#a-b-64k)") == 1


def test_incomplete_result_does_not_promote_row(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    original = json.dumps(Matrix((row,)).to_dict(), indent=2)
    matrix_path.write_text(original)
    checklist = tmp_path / "checklist.md"
    checklist.write_text('<a id="a-b-64k"></a>\n- [ ] **A:B @ 64K context**\n')
    bad = make_result(tmp_path)
    bad = BenchmarkResult(**{**bad.__dict__, "aux_output": ""})

    with pytest.raises(IncompleteBenchmark, match="aux output"):
        EvidencePublisher(matrix_path=matrix_path, checklist_path=checklist, evidence_dir=tmp_path / "evidence").publish_success(row, bad)

    assert matrix_path.read_text() == original
    assert "[ ]" in checklist.read_text()
    assert not (tmp_path / "evidence").exists()


def test_mutated_raw_result_does_not_promote_row(tmp_path: Path) -> None:
    row = make_row()
    matrix_path = tmp_path / "matrix.json"
    original = json.dumps(Matrix((row,)).to_dict(), indent=2)
    matrix_path.write_text(original)
    checklist = tmp_path / "checklist.md"
    checklist.write_text('<a id="a-b-64k"></a>\n- [ ] **A:B @ 64K context**\n')
    result = make_result(tmp_path)
    Path(result.raw_result_path).write_text("{}")

    with pytest.raises(IncompleteBenchmark, match="raw-result checksum mismatch"):
        EvidencePublisher(
            matrix_path=matrix_path,
            checklist_path=checklist,
            evidence_dir=tmp_path / "evidence",
        ).publish_success(row, result)

    assert matrix_path.read_text() == original
