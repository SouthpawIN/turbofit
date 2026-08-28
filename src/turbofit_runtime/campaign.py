"""Resumable bottom-up campaign state machine."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    physical_fingerprint: str
    raw_result_sha256: str
    runtime_string: str
    raw_result_path: str


@dataclass(frozen=True)
class CampaignOutcome:
    row_id: str | None
    status: str
    error: str = ""


class Executor(Protocol):
    def execute(self, item: MatrixRow) -> RawBenchmark: ...
    def recipe_sha256(self, item: MatrixRow) -> str: ...
    def evidence_is_current(self, record: dict) -> bool: ...
    def current_physical_fingerprint(self) -> str: ...


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
        deferred_row_ids: frozenset[str] | None = None,
    ) -> None:
        self.matrix_path = matrix_path
        self.state_path = state_path
        self.executor = executor
        self.clear_gate = clear_gate
        self.publisher = publisher
        self.registry = registry
        self.clear_ceilings_mb = clear_ceilings_mb or {0: 1024, 1: 1024}
        self.deferred_row_ids = deferred_row_ids or frozenset()

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
            if row.id in self.deferred_row_ids:
                continue
            record = state["rows"].get(row.id) or {}
            campaign_status = record.get("status")
            attempts = int(record.get("attempts", 0))
            current_recipe = self.executor.recipe_sha256(row)
            recipe_changed = record.get("recipe_sha256") != current_recipe
            evidence_check = getattr(self.executor, "evidence_is_current", None)
            evidence_stale = (
                campaign_status == "success"
                and evidence_check is not None
                and not evidence_check(record)
            )
            recipe_changed = recipe_changed or evidence_stale
            if row.status in {"success", "blocked"}:
                if not recipe_changed:
                    continue
            if campaign_status in {"success", "blocked", "hardware-incompatible"} and not recipe_changed:
                continue
            if campaign_status == "failed" and not recipe_changed:
                retry_at = str(record.get("next_retry_at") or "")
                if retry_at:
                    try:
                        if datetime.fromisoformat(retry_at) > datetime.now(timezone.utc):
                            continue
                    except ValueError:
                        pass
            pending.append(row)
        return sorted(pending, key=campaign_order)

    def unresolved_rows(self) -> list[MatrixRow]:
        runnable = {row.id for row in self.pending_rows()}
        matrix = load_matrix(self.matrix_path)
        state = self._load_state()
        unresolved = []
        for row in matrix.rows:
            if row.id in self.deferred_row_ids:
                continue
            record = state["rows"].get(row.id) or {}
            if row.id in runnable:
                unresolved.append(row)
                continue
            if (
                record.get("status") == "failed"
                and record.get("recipe_sha256") == self.executor.recipe_sha256(row)
            ):
                unresolved.append(row)
        return sorted(unresolved, key=campaign_order)

    def _record(
        self, row: MatrixRow, *, status: str, error: str = "",
        raw: RawBenchmark | None = None,
    ) -> None:
        state = self._load_state()
        previous = state["rows"].get(row.id) or {}
        recipe_sha256 = self.executor.recipe_sha256(row)
        attempts = int(previous.get("attempts", 0)) if previous.get("recipe_sha256") == recipe_sha256 else 0
        state["rows"][row.id] = {
            **previous,
            "status": status,
            "attempts": attempts + 1,
            "error": error,
            "recipe_sha256": recipe_sha256,
        }
        if status == "failed":
            delay_seconds = 0 if attempts == 0 else min(
                21_600, 30 * (2 ** min(attempts - 1, 10))
            )
            state["rows"][row.id].update({
                "failure_class": "retryable-runtime",
                "next_retry_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                ).isoformat(),
            })
        else:
            state["rows"][row.id].pop("failure_class", None)
            state["rows"][row.id].pop("next_retry_at", None)
        if raw is not None:
            state["rows"][row.id].update({
                "raw_result_path": raw.raw_result_path,
                "raw_result_sha256": raw.raw_result_sha256,
                "physical_fingerprint": raw.physical_fingerprint,
            })
        self._save_state(state)

    def retry_failed(self, *, limit: int | None = None) -> tuple[str, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("retry limit must be positive")
        matrix = load_matrix(self.matrix_path)
        by_id = {row.id: row for row in matrix.rows}
        state = self._load_state()
        failed = [
            row_id for row_id, value in state["rows"].items()
            if row_id in by_id and value.get("status") == "failed"
        ]
        failed.sort(key=lambda row_id: campaign_order(by_id[row_id]))
        selected = failed if limit is None else failed[:limit]
        for row_id in selected:
            previous = state["rows"][row_id]
            history = list(previous.get("history") or [])
            history.append({
                "status": "failed",
                "attempts": int(previous.get("attempts", 0)),
                "error": str(previous.get("error", "")),
            })
            state["rows"][row_id] = {
                **previous,
                "status": "pending",
                "attempts": 0,
                "error": "",
                "history": history,
            }
            state["rows"][row_id].pop("next_retry_at", None)
            state["rows"][row_id].pop("failure_class", None)
        if selected:
            self._save_state(state)
        return tuple(selected)

    def mark_hardware_incompatible(self, row_id: str, evidence_path: Path) -> None:
        matrix = load_matrix(self.matrix_path)
        by_id = {row.id: row for row in matrix.rows}
        row = by_id.get(row_id)
        if row is None:
            raise ValueError(f"unknown campaign row: {row_id}")
        state = self._load_state()
        record = state["rows"].get(row_id) or {}
        recipe_sha = self.executor.recipe_sha256(row)
        if record.get("status") != "failed" or record.get("recipe_sha256") != recipe_sha:
            raise ValueError("only a current-recipe failed row can be classified")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        required = payload.get("required_memory_mb") or {}
        available = payload.get("available_memory_mb") or {}
        if (
            payload.get("schema") != "turbofit.hardware-incompatibility/v1"
            or payload.get("row_id") != row_id
            or payload.get("production_recipe_sha256") != recipe_sha
            or payload.get("physical_fingerprint") != self.executor.current_physical_fingerprint()
            or not isinstance(required, dict)
            or not isinstance(available, dict)
        ):
            raise ValueError("hardware-incompatibility evidence identity mismatch")
        devices = set(required) | set(available)
        if not devices or not any(
            int(required.get(device, 0)) > int(available.get(device, 0))
            for device in devices
        ):
            raise ValueError("evidence does not prove a physical memory incompatibility")
        failure_path = Path(str(payload.get("failure_evidence", "")))
        failure_sha = str(payload.get("failure_evidence_sha256", ""))
        if not failure_path.is_file():
            raise ValueError("failure evidence file is missing")
        actual_failure_sha = "sha256:" + hashlib.sha256(failure_path.read_bytes()).hexdigest()
        if actual_failure_sha != failure_sha:
            raise ValueError("failure evidence checksum mismatch")
        evidence_sha = "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        state["rows"][row_id] = {
            **record,
            "status": "hardware-incompatible",
            "error": str(payload.get("reason") or "physically incompatible"),
            "hardware_incompatibility_evidence": str(evidence_path),
            "hardware_incompatibility_sha256": evidence_sha,
        }
        state["rows"][row_id].pop("next_retry_at", None)
        state["rows"][row_id].pop("failure_class", None)
        self._save_state(state)

    def run_one(self) -> CampaignOutcome:
        pending = self.pending_rows()
        if not pending:
            return CampaignOutcome(row_id=None, status="complete")
        return self.run_row(pending[0])

    def run_row(self, row: MatrixRow) -> CampaignOutcome:
        matches = [item for item in load_matrix(self.matrix_path).rows if item.id == row.id]
        if len(matches) != 1 or matches[0] != row:
            raise ValueError(f"campaign row is not canonical: {row.id}")
        before: GPUClearEvent | None = None
        after: GPUClearEvent | None = None
        raw: RawBenchmark | None = None
        error = ""
        prepared = False
        try:
            prepare = getattr(self.executor, "prepare", None)
            if prepare is not None:
                prepare(row)
                prepared = True
            baseline = self.clear_gate.sample_now()
            before = self.clear_gate.wait(
                ceilings_mb=self.clear_ceilings_mb,
                baseline_mb={sample.gpu: sample.used_mb for sample in baseline},
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
                    baseline_mb=(
                        {sample.gpu: sample.used_mb for sample in before.snapshot}
                        if before is not None else None
                    ),
                    settle_samples=3,
                    timeout_s=180,
                    label=f"after-{row.id}",
                )
            except Exception as exc:
                error = f"{error}; GPU clear failure: {exc!r}" if error else f"GPU clear failure: {exc!r}"
            if prepared:
                try:
                    finish = getattr(self.executor, "finish", None)
                    if finish is not None:
                        finish(row)
                except Exception as exc:
                    error = f"{error}; campaign lease release failure: {exc!r}" if error else f"campaign lease release failure: {exc!r}"

        if error or raw is None or before is None or after is None:
            record_failure = getattr(self.executor, "record_campaign_failure", None)
            if record_failure is not None:
                evidence = record_failure(
                    row, error or "benchmark did not return a result",
                    raw.raw_result_path if raw is not None else None, before, after,
                )
                if "failure_evidence=" not in error:
                    error = f"{error}; failure_evidence={evidence}" if error else f"failure_evidence={evidence}"
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
            physical_fingerprint=raw.physical_fingerprint,
            raw_result_sha256=raw.raw_result_sha256,
            runtime_string=raw.runtime_string,
            gpu_clear_after=after,
            raw_result_path=raw.raw_result_path,
        )
        try:
            evidence_path = self.publisher.publish_success(row, result)
            self.registry.register(row, result, evidence_path)
        except Exception as exc:
            error = repr(exc)
            record_failure = getattr(self.executor, "record_campaign_failure", None)
            if record_failure is not None:
                evidence = record_failure(row, error, raw.raw_result_path, before, after)
                error = f"{error}; failure_evidence={evidence}"
            self._record(row, status="failed", error=error)
            return CampaignOutcome(row_id=row.id, status="failed", error=error)
        self._record(row, status="success", raw=raw)
        return CampaignOutcome(row_id=row.id, status="success")
