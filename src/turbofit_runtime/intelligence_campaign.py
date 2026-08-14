"""Durable state machine for production configuration intelligence benchmarks."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "turbofit.intelligence-campaign-state/v1"
LEVELS = ("screening", "promotion", "release")
LEVEL_PARAMETERS = {
    "screening": {"deepswe_tasks": 3, "deepswe_repetitions": 1, "agentic_repetitions": 1},
    "promotion": {"deepswe_tasks": 30, "deepswe_repetitions": 3, "agentic_repetitions": 3},
    "release": {"deepswe_tasks": 113, "deepswe_repetitions": 3, "agentic_repetitions": 5},
}


def build_state(configurations: Mapping[str, Any]) -> dict[str, Any]:
    rows = configurations.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("configuration matrix has no rows")
    runs: dict[str, dict[str, Any]] = {}
    for row in rows:
        configuration_id = str(row["id"])
        for level in LEVELS:
            run_id = f"{configuration_id}::{level}"
            status = "waiting-runtime" if level == "screening" else f"waiting-{LEVELS[LEVELS.index(level) - 1]}"
            runs[run_id] = {
                "id": run_id,
                "configuration_id": configuration_id,
                "main": str(row["main"]),
                "auxiliary": str(row["auxiliary"]),
                "context": int(row["context"]),
                "level": level,
                "parameters": dict(LEVEL_PARAMETERS[level]),
                "status": status,
                "attempts": 0,
                "result": None,
                "error": None,
            }
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }


def reconcile_state(state: dict[str, Any], configurations: Mapping[str, Any]) -> dict[str, Any]:
    """Reconcile an immutable campaign history with an expanded/replaced active matrix."""
    fresh = build_state(configurations)
    existing = state.get("runs") or {}
    fresh["created_at"] = state.get("created_at") or fresh["created_at"]
    for key in tuple(fresh["runs"]):
        if key in existing:
            fresh["runs"][key] = existing[key]
    refresh_dependencies(fresh)
    fresh["updated_at"] = datetime.now(timezone.utc).isoformat()
    return fresh


def refresh_runtime_status(
    state: dict[str, Any], runtime_state: Mapping[str, Any], display_ids: Mapping[str, str],
    expected_recipe_sha256: Mapping[str, str] | None = None,
) -> None:
    runtime_rows = runtime_state.get("rows") or {}
    expected_recipe_sha256 = expected_recipe_sha256 or {}
    for run in state["runs"].values():
        if (
            run["level"] != "screening"
            or run["status"] == "running"
            or str(run["status"]).startswith("deferred-")
        ):
            continue
        runtime_id = display_ids.get(run["configuration_id"])
        record = runtime_rows.get(runtime_id) or {} if runtime_id else {}
        status = record.get("status")
        expected = expected_recipe_sha256.get(run["configuration_id"])
        recipe_current = expected is None or record.get("recipe_sha256") == expected
        if not recipe_current or status not in {"success", "failed"}:
            if run["status"] in {"ready", "blocked-runtime"}:
                run["status"] = "waiting-runtime"
                run["error"] = "waiting for current production recipe validation"
            elif run["status"] == "success":
                run["status"] = "waiting-runtime"
                run["result"] = None
                run["error"] = "previous intelligence invalidated by production recipe change"
        elif status == "success":
            if run["status"] in {"waiting-runtime", "blocked-runtime"}:
                run["status"] = "ready"
                run["error"] = None
        elif status == "failed" and run["status"] in {"waiting-runtime", "ready", "blocked-runtime"}:
            run["status"] = "blocked-runtime"
            run["error"] = "current production runtime validation failed"
    refresh_dependencies(state)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()


def apply_deferred_configurations(
    state: dict[str, Any], configuration_ids: set[str], *, reason: str,
) -> None:
    """Defer untouched configuration levels without deleting matrix accounting."""
    statuses = {
        "screening": "deferred-runtime",
        "promotion": "deferred-screening",
        "release": "deferred-promotion",
    }
    for run in state["runs"].values():
        deferred = run["configuration_id"] in configuration_ids
        if deferred:
            if run["status"] not in {"running", "success"}:
                run["status"] = statuses[run["level"]]
                run["result"] = None
                run["error"] = reason
        elif str(run["status"]).startswith("deferred-"):
            index = LEVELS.index(run["level"])
            run["status"] = "waiting-runtime" if index == 0 else f"waiting-{LEVELS[index - 1]}"
            run["error"] = None
    state["updated_at"] = datetime.now(timezone.utc).isoformat()


def refresh_dependencies(state: dict[str, Any]) -> None:
    """Propagate terminal prerequisite failures without hiding them.

    A configuration whose exact production runtime cannot execute must not leave
    promotion/release permanently looking pending.  If that runtime is retried
    successfully later, dependency states reopen automatically.
    """
    runs = state["runs"]
    configuration_ids = {run["configuration_id"] for run in runs.values()}
    for configuration_id in configuration_ids:
        screening = runs[f"{configuration_id}::screening"]
        promotion = runs[f"{configuration_id}::promotion"]
        release = runs[f"{configuration_id}::release"]

        if screening["status"] == "deferred-runtime":
            promotion["status"] = "deferred-screening"
            promotion["result"] = None
            promotion["error"] = screening["error"]
            release["status"] = "deferred-promotion"
            release["result"] = None
            release["error"] = screening["error"]
            continue

        if promotion["status"] != "running":
            if screening["status"] == "success":
                if promotion["status"] in {"waiting-screening", "blocked-screening"}:
                    promotion["status"] = "ready"
                    promotion["error"] = None
            elif screening["status"] in {"failed", "blocked-runtime"}:
                promotion["status"] = "blocked-screening"
                promotion["result"] = None
                promotion["error"] = f"screening prerequisite is {screening['status']}"
            else:
                promotion["status"] = "waiting-screening"
                promotion["result"] = None
                promotion["error"] = None

        if release["status"] != "running":
            if promotion["status"] == "success":
                if release["status"] in {"waiting-promotion", "blocked-promotion", "blocked-screening"}:
                    release["status"] = "ready"
                    release["error"] = None
            elif promotion["status"] in {"failed", "blocked-screening"}:
                release["status"] = "blocked-promotion"
                release["result"] = None
                release["error"] = f"promotion prerequisite is {promotion['status']}"
            else:
                release["status"] = "waiting-promotion"
                release["result"] = None
                release["error"] = None


def next_ready_run(state: Mapping[str, Any]) -> dict[str, Any] | None:
    runs = state.get("runs") or {}
    for level in LEVELS:
        ready = sorted(
            (run for run in runs.values() if run.get("level") == level and run.get("status") == "ready"),
            key=lambda run: run["configuration_id"],
        )
        if ready:
            return ready[0]
    return None


def mark_running(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    run = state["runs"][run_id]
    if run["status"] != "ready":
        raise ValueError(f"benchmark run is not ready: {run_id}")
    run["status"] = "running"
    run["attempts"] += 1
    run["started_at"] = datetime.now(timezone.utc).isoformat()
    run["error"] = None
    state["updated_at"] = run["started_at"]
    return run


def recover_interrupted(
    state: dict[str, Any], *, active_pids: set[int] | None = None,
    boot_id: str | None = None, legacy_stale_seconds: int = 21600
) -> tuple[str, ...]:
    """Requeue runs whose recorded worker process no longer exists."""
    if active_pids is None:
        active_pids = {
            int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
        }
    if boot_id is None:
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            boot_id = ""
    now = datetime.now(timezone.utc)
    recovered: list[str] = []
    for run_id, run in state["runs"].items():
        if run["status"] != "running":
            continue
        worker_pid = run.get("worker_pid")
        interrupted = isinstance(worker_pid, int) and worker_pid not in active_pids
        worker_boot_id = run.get("worker_boot_id")
        if worker_boot_id and boot_id and worker_boot_id != boot_id:
            interrupted = True
        if worker_pid is None:
            try:
                started = datetime.fromisoformat(run["started_at"])
                interrupted = (now - started).total_seconds() >= legacy_stale_seconds
            except (KeyError, TypeError, ValueError):
                interrupted = True
        if interrupted:
            run["status"] = "ready"
            run["worker_pid"] = None
            run["worker_boot_id"] = None
            run["error"] = "requeued after interrupted benchmark worker"
            recovered.append(run_id)
    if recovered:
        state["updated_at"] = now.isoformat()
    return tuple(recovered)


def complete_run(
    state: dict[str, Any], run_id: str, *, result: str | None, error: str | None = None
) -> None:
    run = state["runs"][run_id]
    if run["status"] != "running":
        raise ValueError(f"benchmark run is not running: {run_id}")
    run["status"] = "success" if error is None else "failed"
    run["result"] = result
    run["error"] = error
    run["worker_pid"] = None
    run["worker_boot_id"] = None
    run["finished_at"] = datetime.now(timezone.utc).isoformat()
    if error is None:
        index = LEVELS.index(run["level"])
        if index + 1 < len(LEVELS):
            successor = state["runs"][f"{run['configuration_id']}::{LEVELS[index + 1]}"]
            if successor["status"] == f"waiting-{run['level']}":
                successor["status"] = "ready"
    refresh_dependencies(state)
    state["updated_at"] = run["finished_at"]


def retry_failed(state: dict[str, Any], *, limit: int | None = None) -> tuple[str, ...]:
    selected = []
    for run_id, run in state["runs"].items():
        if run["status"] == "failed":
            run["status"] = "ready"
            run["error"] = None
            selected.append(run_id)
            if limit is not None and len(selected) >= limit:
                break
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return tuple(selected)


def summary(state: Mapping[str, Any]) -> dict[str, Any]:
    runs = tuple((state.get("runs") or {}).values())
    by_level = {}
    for level in LEVELS:
        selected = [run for run in runs if run.get("level") == level]
        counts: dict[str, int] = {}
        for run in selected:
            counts[run["status"]] = counts.get(run["status"], 0) + 1
        by_level[level] = {"total": len(selected), "statuses": counts}
    return {"total_configurations": len(runs) // len(LEVELS), "levels": by_level}
