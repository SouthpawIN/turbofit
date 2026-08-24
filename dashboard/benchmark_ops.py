"""Bounded, local candidate benchmarking for the Turbofit Desktop surface."""
from __future__ import annotations

import math
import ipaddress
import tempfile
import threading
from numbers import Real
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from plugin_tools import (
    SELECTION_PATH,
    _load_json,
    combination_snapshot,
    select_profile,
)
from product_ops import shift_configuration
from turbofit_runtime.benchmark_stage import BenchmarkSuite, run_benchmark


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "benchmarks" / "stage-v1.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8091/v1"
_BENCHMARK_LOCK = threading.Lock()
RUNTIME_MUTATION_LOCK = _BENCHMARK_LOCK


def _validated_benchmark_url(value: str) -> str:
    parsed = urlparse(str(value or DEFAULT_BASE_URL).strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("benchmark base_url must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("benchmark base_url must not contain credentials, a query, or a fragment")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost":
        host_ok = True
    else:
        try:
            host_ok = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            host_ok = False
    if not host_ok:
        raise ValueError("benchmark base_url must target this machine's loopback interface")
    return str(value or DEFAULT_BASE_URL).strip().rstrip("/")


def _candidate_profiles(limit: int) -> list[str]:
    rows = combination_snapshot().get("combinations") or []
    profiles: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("fit"):
            continue
        profile = str(row.get("profile") or "").strip()
        if profile and profile not in profiles:
            profiles.append(profile)
        if len(profiles) >= limit:
            break
    return profiles


def benchmark_candidates(
    *,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = 3,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError("limit must be an integer from 1 to 8")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real) or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be a finite number")
    if timeout_seconds <= 0 or timeout_seconds > 900:
        raise ValueError("timeout_seconds must be between 0 and 900")
    validated_url = _validated_benchmark_url(base_url)
    with _BENCHMARK_LOCK:
        return _run_benchmark_candidates(
            base_url=validated_url,
            limit=limit,
            timeout_seconds=float(timeout_seconds),
        )


def _run_benchmark_candidates(
    *,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = 3,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Benchmark fit candidates and restore the prior selection afterward."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError("limit must be an integer from 1 to 8")
    if timeout_seconds <= 0 or timeout_seconds > 900:
        raise ValueError("timeout_seconds must be between 0 and 900")

    candidates = _candidate_profiles(limit)
    if not candidates:
        return {"ok": False, "status": "blocked", "error": "no physically fitting candidates"}

    previous = _load_json(SELECTION_PATH) or {}
    previous_profile = str(previous.get("requested") or previous.get("profile_id") or "auto")
    if previous_profile.startswith("manual-"):
        previous_profile = previous_profile[7:]
    suite = BenchmarkSuite.load(SUITE_PATH)
    results: list[dict[str, Any]] = []
    restore_error: str | None = None
    try:
        for profile in candidates:
            try:
                shift = shift_configuration(profile)
                if not shift.get("shifted"):
                    raise RuntimeError(shift.get("error") or "candidate could not be activated")
                with tempfile.TemporaryDirectory(prefix="turbofit-benchmark-") as directory:
                    evidence = run_benchmark(
                        suite=suite,
                        base_url=base_url,
                        model="auto",
                        candidate=profile,
                        configuration=profile,
                        output_path=Path(directory) / "evidence.json",
                        timeout_seconds=timeout_seconds,
                    )
                results.append({
                    "profile": profile,
                    "status": evidence["status"],
                    "summary": evidence["summary"],
                    "evidence_sha256": evidence["evidence_sha256"],
                    "promoted": False,
                })
            except Exception as exc:
                results.append({
                    "profile": profile,
                    "status": "fail",
                    "error": "candidate benchmark failed; inspect Turbofit logs",
                    "promoted": False,
                })
    finally:
        try:
            select_profile(previous_profile)
        except Exception:
            restore_error = "previous Turbofit profile could not be restored"

    passing = [item for item in results if item.get("status") == "pass"]
    passing.sort(
        key=lambda item: float(
            (item.get("summary") or {}).get("effective_output_tokens_per_second") or 0
        ),
        reverse=True,
    )
    return {
        "ok": restore_error is None,
        "status": "restore_failed" if restore_error else "complete",
        "error": restore_error,
        "candidates": results,
        "best": passing[0] if passing else None,
        "restored_profile": None if restore_error else previous_profile,
        "promoted": False,
    }
