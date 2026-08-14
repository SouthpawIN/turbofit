"""Evidence gate and atomic publication for Main:Aux matrix results."""
from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .gpu import GPUClearEvent
from .schema import Matrix, MatrixRow, load_matrix


class IncompleteBenchmark(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkResult:
    row_id: str
    method: str
    exact_context: bool
    main_health: bool
    aux_health: bool
    main_output: str
    aux_output: str
    main_tps: float
    aux_tps: float
    gpu_peak_mb: dict[int, int]
    physical_fingerprint: str
    raw_result_sha256: str
    runtime_string: str
    gpu_clear_after: GPUClearEvent
    raw_result_path: str

    def validate(self, row: MatrixRow) -> None:
        failures = []
        if self.row_id != row.id:
            failures.append("row id mismatch")
        if not self.exact_context:
            failures.append("exact context")
        if not self.main_health:
            failures.append("main health")
        if not self.aux_health:
            failures.append("aux health")
        if not self.main_output.strip():
            failures.append("main output")
        if not self.aux_output.strip():
            failures.append("aux output")
        if self.main_tps <= 0:
            failures.append("main throughput")
        if self.aux_tps <= 0:
            failures.append("aux throughput")
        if not self.gpu_peak_mb:
            failures.append("GPU peak memory")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.physical_fingerprint):
            failures.append("physical hardware fingerprint")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.raw_result_sha256):
            failures.append("raw-result checksum")
        else:
            raw_path = Path(self.raw_result_path)
            if not raw_path.is_file():
                failures.append("raw-result file")
            else:
                actual = "sha256:" + hashlib.sha256(raw_path.read_bytes()).hexdigest()
                if actual != self.raw_result_sha256:
                    failures.append("raw-result checksum mismatch")
                try:
                    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    bound = str((raw_payload.get("physical_hardware") or {}).get("fingerprint_sha256", ""))
                    if bound != self.physical_fingerprint:
                        failures.append("raw-result physical fingerprint mismatch")
                except (OSError, json.JSONDecodeError):
                    failures.append("raw-result JSON")
        if not self.runtime_string.strip():
            failures.append("runtime string")
        if not self.gpu_clear_after.passed:
            failures.append("GPU clear after")
        if failures:
            raise IncompleteBenchmark("incomplete benchmark: " + ", ".join(failures))


class EvidencePublisher:
    def __init__(self, *, matrix_path: Path, checklist_path: Path, evidence_dir: Path) -> None:
        self.matrix_path = matrix_path
        self.checklist_path = checklist_path
        self.evidence_dir = evidence_dir

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _context_label(context: int) -> str:
        return {65_536: "64K", 131_072: "128K", 262_144: "262K", 1_048_576: "1M"}[context]

    def _initial_checklist(self, matrix: Matrix) -> str:
        lines = [
            "---",
            "title: Turbofit benchmark campaign",
            "type: benchmark",
            "tags: [turbofit, native-runtime, benchmark]",
            "---",
            "",
            "# Turbofit benchmark campaign",
            "",
            "### Success index",
            "",
        ]
        for item in matrix.rows:
            label = self._context_label(item.context)
            lines.append(f'<a id="{item.id}"></a>')
            lines.append(f'- [ ] **{item.main}:{item.aux} @ {label} context**')
        return "\n".join(lines) + "\n"

    def ensure_checklist(self) -> None:
        """Synchronize a stale checklist to the current matrix without losing successes."""
        matrix = load_matrix(self.matrix_path)
        fresh = self._initial_checklist(matrix)
        try:
            existing = self.checklist_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = ""

        success_index: list[str] = []
        in_index = False
        for line in existing.splitlines():
            if line == "### Success index":
                in_index = True
                continue
            if in_index and line.startswith("- [") and not line.startswith(("- [ ]", "- [x]")):
                success_index.append(line)
            elif in_index and line.startswith("<a id="):
                break

        for row in matrix.rows:
            label = self._context_label(row.context)
            pending = f'- [ ] **{row.main}:{row.aux} @ {label} context**'
            passed_base = f'- [x] **{row.main}:{row.aux} @ {label} context**'
            old_passed = next((line for line in existing.splitlines() if line.startswith(passed_base)), None)
            evidence_path = self.evidence_dir / f"{row.id}.md"
            if old_passed and evidence_path.is_file():
                fresh = fresh.replace(pending, old_passed, 1)
            elif evidence_path.is_file():
                fresh = fresh.replace(
                    pending,
                    f'{passed_base} — [evidence](evidence/{evidence_path.name})',
                    1,
                )

        if success_index:
            marker = "### Success index\n\n"
            fresh = fresh.replace(marker, marker + "\n".join(dict.fromkeys(success_index)) + "\n", 1)
        if fresh != existing:
            self._atomic_write(self.checklist_path, fresh)

    def publish_success(self, row: MatrixRow, result: BenchmarkResult) -> Path:
        result.validate(row)
        matrix = load_matrix(self.matrix_path)
        matches = [item for item in matrix.rows if item.id == row.id]
        if len(matches) != 1:
            raise ValueError(f"matrix row not found exactly once: {row.id}")
        stored = matches[0]
        if (stored.main, stored.aux, stored.context) != (row.main, row.aux, row.context):
            raise ValueError(f"matrix row differs from publication request: {row.id}")

        label = self._context_label(row.context)
        evidence_path = self.evidence_dir / f"{row.id}.md"
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_rows = "\n".join(
            f"| GPU {gpu} | {peak:,} MiB |" for gpu, peak in sorted(result.gpu_peak_mb.items())
        )
        evidence = f"""---
title: Matrix evidence - {row.main} with {row.aux} at {label}
created: {timestamp[:10]}
updated: {timestamp[:10]}
type: benchmark
tags: [turbofit, native-runtime, benchmark, runtime]
---

# Matrix evidence: {row.main}:{row.aux} @ {label}

- Checklist row: [{row.main}:{row.aux} @ {label}](../main-aux-inference-checklist.md#{row.id})
- Method: `{result.method}`
- Exact context: `{row.context}`
- Main health: `PASS`
- Auxiliary health: `PASS`
- Main decode: `{result.main_tps:.2f} tok/s`
- Auxiliary decode: `{result.aux_tps:.2f} tok/s`
- Runtime string: `{result.runtime_string}`
- Raw result: `{result.raw_result_path}`
- Raw result SHA-256: `{result.raw_result_sha256}`
- Physical hardware fingerprint: `{result.physical_fingerprint}`
- GPU-clear gate: `PASS`
- GPU-clear event: `{result.gpu_clear_after.timestamp}`

| Device | Peak memory |
|---|---:|
{memory_rows}

## Gate

**PASS.** Exact context, both health checks, both inference routes, measured throughput, per-card peak memory, a deterministic runtime string, and post-run GPU clearing were all recorded.
"""

        try:
            checklist = self.checklist_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            checklist = self._initial_checklist(matrix)
        pending = f'- [ ] **{row.main}:{row.aux} @ {label} context**'
        passed_base = f'- [x] **{row.main}:{row.aux} @ {label} context**'
        passed = f'{passed_base} — [evidence](evidence/{evidence_path.name})'
        if pending in checklist:
            checklist = checklist.replace(pending, passed, 1)
        elif passed_base not in checklist:
            raise ValueError(f"checklist row not found: {row.id}")
        index_line = (
            f"- [{row.main}:{row.aux} @ {label}](#{row.id}) — "
            f"`{result.runtime_string}`; [evidence](evidence/{evidence_path.name})."
        )
        if index_line not in checklist:
            marker = "### Success index\n\n"
            if marker not in checklist:
                raise ValueError("success index marker not found")
            checklist = checklist.replace(marker, marker + index_line + "\n", 1)

        updated_rows = tuple(replace(item, status="success") if item.id == row.id else item for item in matrix.rows)
        updated_matrix = Matrix(updated_rows)
        self._atomic_write(evidence_path, evidence)
        self._atomic_write(self.matrix_path, json.dumps(updated_matrix.to_dict(), indent=2) + "\n")
        self._atomic_write(self.checklist_path, checklist)
        return evidence_path
