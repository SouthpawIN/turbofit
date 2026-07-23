"""Resumable bottom-up campaign state machine."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .evidence import BenchmarkResult, EvidencePublisher
from .gpu import GPUClearEvent, GPUClearGate
from .schema import MatrixRow, load_matrix


FAMILY_RANK = {
    "1 Bit Bonsai": 0,
    "Ternary Bonsai": 1,
    "Carwin Nano": 2,
    "GRM 2.6 Plus": 3,
    "Laguna S2.1": 4,
    "MiniMax M3": 5,
    "GLM 5.2": 6,
}


def campaign_order(row: MatrixRow) -> tuple[int, int, str, str]:
    return (FAMILY_RANK.get(row.main, 99), row.context, row.main, row.aux)


@dataclass(frozen=True)
class RawBenchmark:
    method: str
    exact_context: bool
    main_health: bool
    aux_health: bool
    main_output: str
    aux_output: str
    main_tps: float
    aux_tps: float
    gpu_peak_mb: dict[int, int]
    runtime_string: str
    raw_result_path: str


@dataclass(frozen=True)
class CampaignOutcome:
    row_id: str | None
    status: str
    error: str = ""


class Executor(Protocol):
    def execute(self, item: MatrixRow) -> RawBenchmark: ...


class Registry(Protocol):
    def register(self, item: MatrixRow, result: BenchmarkResult, evidence_path: Path) -> None: ...


class CampaignRunner:
    def __init__(
        self,
        *,
        matrix_path: Path,
        state_path: Path,
        executor: Executor,
        clear_gate: GPUClearGate,
        publisher: EvidencePublisher,
        registry: Registry,
        clear_ceilings_mb: dict[int, int] | None = None,
    ) -> None:
        self.matrix_path = matrix_path
        self.state_path = state_path
        self.executor = executor
        self.clear_gate = clear_gate
        self.publisher = publisher
        self.registry = registry
        self.clear_ceilings_mb = clear_ceilings_mb or {0: 1024, 1: 1024}

    def _load_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            state = {"schema_version": 1, "rows": {}}
        if state.get("schema_version") != 1:
            raise ValueError(f"unsupported campaign state: {state.get('schema_version')}")
        state.setdefault("rows", {})
        return state

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.state_path.name}.", dir=self.state_path.parent)
        try:
            with os.fdopen(descriptor, "w") as handle:
                json.dump(state, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def pending_rows(self) -> list[MatrixRow]:
        matrix = load_matrix(self.matrix_path)
        state = self._load_state()
        pending = []
        for row in matrix.rows:
            campaign_status = (state["rows"].get(row.id) or {}).get("status")
            attempts = int((state["rows"].get(row.id) or {}).get("attempts", 0))
            if row.status in {"success", "blocked"}:
                continue
            if campaign_status in {"success", "blocked"}:
                continue
            if campaign_status == "failed" and attempts >= 2:
                continue
            pending.append(row)
        return sorted(pending, key=campaign_order)

    def _record(self, row: MatrixRow, *, status: str, error: str = "") -> None:
        state = self._load_state()
        previous = state["rows"].get(row.id) or {}
        state["rows"][row.id] = {
            **previous,
            "status": status,
            "attempts": int(previous.get("attempts", 0)) + 1,
            "error": error,
        }
        self._save_state(state)

    def run_one(self) -> CampaignOutcome:
        pending = self.pending_rows()
        if not pending:
            return CampaignOutcome(row_id=None, status="complete")
        row = pending[0]
        before: GPUClearEvent | None = None
        after: GPUClearEvent | None = None
        raw: RawBenchmark | None = None
        error = ""
        try:
            before = self.clear_gate.wait(
                ceilings_mb=self.clear_ceilings_mb,
                settle_samples=3,
                timeout_s=180,
                label=f"before-{row.id}",
            )
            raw = self.executor.execute(row)
        except Exception as exc:
            error = repr(exc)
        finally:
            try:
                after = self.clear_gate.wait(
                    ceilings_mb=self.clear_ceilings_mb,
                    settle_samples=3,
                    timeout_s=180,
                    label=f"after-{row.id}",
                )
            except Exception as exc:
                error = f"{error}; GPU clear failure: {exc!r}" if error else f"GPU clear failure: {exc!r}"

        if error or raw is None or before is None or after is None:
            self._record(row, status="failed", error=error or "benchmark did not return a result")
            return CampaignOutcome(row_id=row.id, status="failed", error=error)

        result = BenchmarkResult(
            row_id=row.id,
            method=raw.method,
            exact_context=raw.exact_context,
            main_health=raw.main_health,
            aux_health=raw.aux_health,
            main_output=raw.main_output,
            aux_output=raw.aux_output,
            main_tps=raw.main_tps,
            aux_tps=raw.aux_tps,
            gpu_peak_mb=raw.gpu_peak_mb,
            runtime_string=raw.runtime_string,
            gpu_clear_after=after,
            raw_result_path=raw.raw_result_path,
        )
        try:
            evidence_path = self.publisher.publish_success(row, result)
            self.registry.register(row, result, evidence_path)
        except Exception as exc:
            error = repr(exc)
            self._record(row, status="failed", error=error)
            return CampaignOutcome(row_id=row.id, status="failed", error=error)
        self._record(row, status="success")
        return CampaignOutcome(row_id=row.id, status="success")
