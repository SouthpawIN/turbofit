from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from turbofit_runtime.managed_engine import (
    EngineLaunch,
    ManagedEngineRuntime,
    publish_engine_routes,
)


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0



def launch(tmp_path: Path) -> EngineLaunch:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    executable = tmp_path / "python"
    executable.write_text("python")
    executable.chmod(0o755)
    return EngineLaunch(
        engine_id="mlx",
        model_id="qwen3-8-27b-uncensored-mlx-8bit",
        model_path=model,
        port=18081,
        context_length=262_144,
        bind_host="127.0.0.1",
        upstream_model_id=str(model),
        command=(
            str(executable),
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            "18081",
        ),
    )


def test_owned_engine_is_published_only_after_exact_model_readiness(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    state = tmp_path / "owned.json"
    process = FakeProcess()
    calls: list[tuple[str, ...]] = []
    observed_command = ("/real/Python", *spec.command[1:])
    listeners = iter(((), (process.pid,)))

    runtime = ManagedEngineRuntime(
        state_path=state,
        process_factory=lambda command, **_kwargs: calls.append(tuple(command)) or process,
        command_line=lambda _pid: " ".join(observed_command),
        listener_pids=lambda _port: next(listeners),
        json_get=lambda url, _timeout: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"data": [{"id": spec.upstream_model_id}]}
        ),
        sleep=lambda _seconds: None,
        clock=iter((0.0, 0.0)).__next__,
    )

    resident = runtime.start_owned(spec, timeout_s=1.0)

    assert resident.owned is True
    assert resident.pid == process.pid
    assert calls == [spec.command]
    persisted = json.loads(state.read_text())
    assert persisted["model_id"] == spec.model_id
    assert persisted["command"] == list(observed_command)


def test_wrong_upstream_model_never_becomes_owned_or_routable(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    process = FakeProcess()
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        process_factory=lambda _command, **_kwargs: process,
        listener_pids=lambda _port: (),
        command_line=lambda _pid: " ".join(spec.command),
        json_get=lambda url, _timeout: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"data": [{"id": "another-model"}]}
        ),
        sleep=lambda _seconds: None,
        clock=iter((0.0, 2.0)).__next__,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        runtime.start_owned(spec, timeout_s=1.0)

    assert not (tmp_path / "owned.json").exists()


def test_malformed_models_response_fails_closed_as_not_ready(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    process = FakeProcess()
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        process_factory=lambda _command, **_kwargs: process,
        listener_pids=lambda _port: (),
        json_get=lambda _url, _timeout: [],
        sleep=lambda _seconds: None,
        clock=iter((0.0, 2.0)).__next__,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        runtime.start_owned(spec, timeout_s=1.0)


def test_external_engine_is_never_signalled(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    signalled: list[tuple[int, int]] = []
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        command_line=lambda _pid: "external mtplx serve --port 18081",
        listener_pids=lambda _port: (999,),
        signal_process=lambda pid, signal: signalled.append((pid, signal)),
        json_get=lambda url, _timeout: (
            {"ok": True, "model": spec.model_id}
            if url.endswith("/health")
            else {"data": [{"id": spec.upstream_model_id}]}
        ),
    )

    resident = runtime.adopt_external(spec, pid=999)

    assert resident.owned is False
    assert runtime.stop() is True
    assert signalled == []
    assert not (tmp_path / "owned.json").exists()


def test_external_adoption_requires_reported_pid_to_own_the_listener(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        listener_pids=lambda _port: (998,),
        json_get=lambda url, _timeout: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"data": [{"id": spec.upstream_model_id}]}
        ),
    )

    with pytest.raises(RuntimeError, match="does not own"):
        runtime.adopt_external(spec, pid=999)


def test_recovered_pid_must_still_match_full_recorded_command(tmp_path: Path) -> None:
    assert not ManagedEngineRuntime._command_matches(
        ("python", "-m", "server", "--port", "18081"),
        "python --port 18081 -m server",
    )
    spec = launch(tmp_path)
    state = tmp_path / "owned.json"
    state.write_text(json.dumps({
        "schema": "turbofit.managed-engine/v1",
        "engine_id": spec.engine_id,
        "model_id": spec.model_id,
        "model_path": str(spec.model_path),
        "port": spec.port,
        "context_length": spec.context_length,
        "bind_host": spec.bind_host,
        "upstream_model_id": spec.upstream_model_id,
        "pid": 4321,
        "command": list(spec.command),
        "owned": True,
        "status": "ready",
    }))
    signalled: list[tuple[int, int]] = []
    runtime = ManagedEngineRuntime(
        state_path=state,
        command_line=lambda _pid: "python -m unrelated --port 18081",
        signal_process=lambda pid, signal: signalled.append((pid, signal)),
    )

    assert runtime.stop() is False
    assert signalled == []
    assert state.exists()
    assert json.loads(state.read_text())["status"] == "command-mismatch"


def test_lifecycle_lock_serializes_cross_process_start_and_stop(tmp_path: Path) -> None:
    runtime = ManagedEngineRuntime(state_path=tmp_path / "owned.json")

    with runtime._lifecycle_lock():
        with runtime.lock_path.open("a+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_owned_start_refuses_an_occupied_port_before_spawning(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    spawned: list[tuple[str, ...]] = []
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        process_factory=lambda command, **_kwargs: spawned.append(tuple(command)),
        listener_pids=lambda _port: (999,),
    )

    with pytest.raises(RuntimeError, match="already occupied"):
        runtime.start_owned(spec)

    assert spawned == []


def test_readiness_requires_the_launched_pid_to_own_the_listener(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    process = FakeProcess()
    listeners = iter(((), (999,), (999,)))
    runtime = ManagedEngineRuntime(
        state_path=tmp_path / "owned.json",
        process_factory=lambda _command, **_kwargs: process,
        command_line=lambda _pid: " ".join(spec.command),
        listener_pids=lambda _port: next(listeners, (999,)),
        json_get=lambda url, _timeout: (
            {"status": "ok"}
            if url.endswith("/health")
            else {"data": [{"id": spec.upstream_model_id}]}
        ),
        sleep=lambda _seconds: None,
        clock=iter((0.0, 0.0, 0.0, 2.0)).__next__,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        runtime.start_owned(spec, timeout_s=1.0)

    assert process.returncode == 0


def test_non_loopback_bind_is_rejected_even_if_another_argument_mentions_localhost(
    tmp_path: Path,
) -> None:
    spec = launch(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        EngineLaunch(
            **{
                **spec.__dict__,
                "bind_host": "0.0.0.0",
                "command": (*spec.command, "/tmp/localhost-cache"),
            }
        )


def test_owned_stop_waits_for_verified_process_exit(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    state = tmp_path / "owned.json"
    state.write_text(json.dumps({
        "schema": "turbofit.managed-engine/v1",
        "engine_id": spec.engine_id,
        "model_id": spec.model_id,
        "model_path": str(spec.model_path),
        "port": spec.port,
        "context_length": spec.context_length,
        "bind_host": spec.bind_host,
        "upstream_model_id": spec.upstream_model_id,
        "pid": 77,
        "command": list(spec.command),
        "owned": True,
        "status": "ready",
    }))
    observed = iter((" ".join(spec.command), " ".join(spec.command), ""))
    signals: list[tuple[int, int]] = []
    sleeps: list[float] = []
    runtime = ManagedEngineRuntime(
        state_path=state,
        command_line=lambda _pid: next(observed, ""),
        signal_process=lambda pid, sig: signals.append((pid, sig)),
        sleep=lambda seconds: sleeps.append(seconds),
        clock=iter((0.0, 0.0, 0.1)).__next__,
    )

    assert runtime.stop(timeout_s=5.0) is True
    assert len(signals) == 1
    assert sleeps
    assert not state.exists()


def test_status_reports_stale_state_instead_of_treating_file_existence_as_health(
    tmp_path: Path,
) -> None:
    spec = launch(tmp_path)
    state = tmp_path / "owned.json"
    state.write_text(json.dumps({
        "schema": "turbofit.managed-engine/v1",
        "engine_id": spec.engine_id,
        "model_id": spec.model_id,
        "model_path": str(spec.model_path),
        "port": spec.port,
        "context_length": spec.context_length,
        "bind_host": spec.bind_host,
        "upstream_model_id": spec.upstream_model_id,
        "pid": 77,
        "command": list(spec.command),
        "owned": True,
        "status": "ready",
    }))
    runtime = ManagedEngineRuntime(
        state_path=state,
        command_line=lambda _pid: "",
        listener_pids=lambda _port: (),
    )

    result = runtime.inspect()

    assert result["ok"] is False
    assert result["status"] == "stale"
    assert state.exists()


def test_engine_routes_use_real_upstream_model_id_and_shared_aux(tmp_path: Path) -> None:
    spec = launch(tmp_path)
    path = tmp_path / "runtime-state.json"

    publish_engine_routes(path, spec)

    payload = json.loads(path.read_text())
    assert payload["active"] == "apple-mlx"
    assert payload["routes"] == {
        "main": {
            "kind": "local",
            "alias": spec.model_id,
            "model_id": spec.upstream_model_id,
            "port": 18081,
            "context_length": 262_144,
            "engine": "mlx",
        },
        "aux": {"kind": "shared-main", "context_length": 262_144},
    }
