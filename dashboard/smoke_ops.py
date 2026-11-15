"""Loopback smoke check of the currently serving Turbofit gateway.

This is not a promotion benchmark. It does not shift production profiles,
does not take a campaign lease, and never writes a promotion record.
"""
from __future__ import annotations

import math
import os
import threading
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from turbofit_runtime.benchmark_stage import BenchmarkSuite, run_benchmark


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "benchmarks" / "stage-v1.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8091/v1"
_SMOKE_LOCK = threading.Lock()


def _evidence_dir() -> Path:
    state_home = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state"))
    return state_home / "turbofit" / "smoke"


def smoke_local_runtime(*, timeout_seconds: float = 300.0) -> dict[str, Any]:
    """Run the local smoke suite against 127.0.0.1:8091. Never promotes."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real) or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite number")
    if timeout_seconds <= 0 or timeout_seconds > 900:
        raise ValueError("timeout_seconds must be between 0 and 900")
    with _SMOKE_LOCK:
        suite = BenchmarkSuite.load(SUITE_PATH)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = _evidence_dir() / f"{stamp}-local-runtime-smoke-v1.json"
        evidence = run_benchmark(
            suite=suite,
            base_url=DEFAULT_BASE_URL,
            model="auto",
            candidate="active:main",
            configuration="active-local-smoke",
            output_path=output_path,
            timeout_seconds=float(timeout_seconds),
            require_gpu_samples=False,
        )
        return {
            "ok": evidence.get("status") == "pass",
            "status": evidence.get("status"),
            "suite": suite.name,
            "promoted": False,
            "endpoint": DEFAULT_BASE_URL,
            "evidence_path": str(output_path),
            "evidence_sha256": evidence.get("evidence_sha256"),
            "summary": evidence.get("summary"),
            "request_failures": evidence.get("request_failures") or [],
            "validator_failures": evidence.get("validator_failures") or [],
            "resource_failures": evidence.get("resource_failures") or [],
            "resource_warnings": evidence.get("resource_warnings") or [],
            "message": (
                "Local smoke completed. This is not a promotion benchmark."
                if evidence.get("status") == "pass"
                else "Local smoke failed. Inspect the evidence file; nothing was promoted."
            ),
        }
