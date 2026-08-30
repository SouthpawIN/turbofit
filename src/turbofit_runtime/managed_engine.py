"""Ownership-safe lifecycle for loopback OpenAI-compatible model engines."""
from __future__ import annotations

import ipaddress
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .routes import publish_route_state


@dataclass(frozen=True)
class EngineLaunch:
    engine_id: str
    model_id: str
    model_path: Path
    port: int
    context_length: int
    bind_host: str
    upstream_model_id: str
    command: tuple[str, ...]
    required_model_files: tuple[str, ...] = ("config.json",)

    def __post_init__(self) -> None:
        if not self.engine_id or not self.model_id or not self.upstream_model_id or not self.command:
            raise ValueError("engine, model, and command must be non-empty")
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("engine port must be in 1..65535")
        if isinstance(self.context_length, bool) or self.context_length <= 0:
            raise ValueError("context_length must be positive")
        try:
            address = ipaddress.ip_address(self.bind_host)
        except ValueError as exc:
            raise ValueError("managed engine bind host must be an IP address") from exc
        if not address.is_loopback:
            raise ValueError("managed engines must bind to loopback")


@dataclass(frozen=True)
class EngineResident:
    engine_id: str
    model_id: str
    model_path: str
    port: int
    context_length: int
    bind_host: str
    upstream_model_id: str
    pid: int
    command: tuple[str, ...]
    owned: bool
    status: str = "ready"


class ManagedEngineRuntime:
    """Start or adopt an engine while signalling only a verified owned PID."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        process_factory: Callable[..., Any] = subprocess.Popen,
        command_line: Callable[[int], str] | None = None,
        listener_pids: Callable[[int], tuple[int, ...]] | None = None,
        signal_process: Callable[[int, int], None] = os.kill,
        json_get: Callable[[str, float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.state_path = Path(state_path)
        self.lock_path = self.state_path.with_suffix(".lock")
        self.process_factory = process_factory
        self.command_line = command_line or self._command_line
        self.listener_pids = listener_pids or self._listener_pids
        self.signal_process = signal_process
        self.json_get = json_get or self._json_get
        self.sleep = sleep
        self.clock = clock
        self._resident: EngineResident | None = None
        self._process: Any | None = None

    @contextmanager
    def _lifecycle_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _command_line(pid: int) -> str:
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def _listener_pids(port: int) -> tuple[int, ...]:
        try:
            result = subprocess.run(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ()
        if result.returncode not in {0, 1}:
            return ()
        return tuple(
            int(line)
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        )

    @staticmethod
    def _json_get(url: str, timeout: float) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _resident_for(spec: EngineLaunch, *, pid: int, owned: bool) -> EngineResident:
        return EngineResident(
            engine_id=spec.engine_id,
            model_id=spec.model_id,
            model_path=str(spec.model_path),
            port=spec.port,
            context_length=spec.context_length,
            bind_host=spec.bind_host,
            upstream_model_id=spec.upstream_model_id,
            pid=pid,
            command=spec.command,
            owned=owned,
        )

    def _ready(self, spec: EngineLaunch) -> bool:
        base = f"http://127.0.0.1:{spec.port}"
        try:
            models = self.json_get(base + "/v1/models", 2.0)
        except (OSError, ValueError):
            return False
        if not isinstance(models, Mapping):
            return False
        ids = {
            str(item.get("id"))
            for item in (models.get("data") or [])
            if isinstance(item, dict)
        }
        if spec.upstream_model_id not in ids:
            return False
        try:
            health = self.json_get(base + "/health", 2.0)
        except (OSError, ValueError):
            return True
        if not isinstance(health, dict):
            return False
        return bool(
            health.get("ok") is True
            or health.get("status") in {"ok", "ready", "healthy", "loaded"}
        )

    def _write_owned(self, resident: EngineResident) -> None:
        payload = {
            "schema": "turbofit.managed-engine/v1",
            **asdict(resident),
            "command": list(resident.command),
        }
        publish_route_state(self.state_path, payload)

    def start_owned(self, spec: EngineLaunch, *, timeout_s: float = 900.0) -> EngineResident:
        with self._lifecycle_lock():
            return self._start_owned(spec, timeout_s=timeout_s)

    def _start_owned(self, spec: EngineLaunch, *, timeout_s: float) -> EngineResident:
        executable = Path(spec.command[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"engine executable is unavailable: {executable}")
        if not spec.model_path.is_dir() or any(
            not (spec.model_path / relative).is_file()
            for relative in spec.required_model_files
        ):
            raise RuntimeError(f"model snapshot is incomplete: {spec.model_path}")
        occupants = self.listener_pids(spec.port)
        if occupants:
            raise RuntimeError(
                f"engine port {spec.port} is already occupied by PID(s) "
                + ",".join(str(pid) for pid in occupants)
            )
        log_path = self.state_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = self.process_factory(
            list(spec.command),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log.close()
        self._process = process
        resident = replace(
            self._resident_for(spec, pid=int(process.pid), owned=True),
            status="launching",
        )
        self._write_owned(resident)
        deadline = self.clock() + timeout_s
        while True:
            if process.poll() is not None:
                break
            if self._ready(spec) and resident.pid in self.listener_pids(spec.port):
                actual = self.command_line(resident.pid)
                if actual:
                    try:
                        resident = replace(resident, command=tuple(shlex.split(actual)))
                    except ValueError:
                        pass
                resident = replace(resident, status="ready")
                self._resident = resident
                self._write_owned(resident)
                return resident
            if self.clock() >= deadline:
                break
            self.sleep(min(0.5, max(0.0, deadline - self.clock())))
        if process.poll() is None:
            process.terminate()
        cleanup_deadline = time.monotonic() + 30.0
        while process.poll() is None and time.monotonic() < cleanup_deadline:
            self.sleep(0.2)
        if process.poll() is None:
            resident = replace(resident, status="cleanup-timeout")
            self._resident = resident
            self._write_owned(resident)
        else:
            self._resident = None
            self.state_path.unlink(missing_ok=True)
        self._process = None
        raise RuntimeError(f"{spec.engine_id} did not become ready with model {spec.model_id}")

    def adopt_external(self, spec: EngineLaunch, *, pid: int) -> EngineResident:
        if pid <= 0 or not self._ready(spec):
            raise RuntimeError(f"external {spec.engine_id} endpoint is not ready")
        if pid not in self.listener_pids(spec.port):
            raise RuntimeError(
                f"external {spec.engine_id} PID {pid} does not own port {spec.port}"
            )
        resident = self._resident_for(spec, pid=pid, owned=False)
        self._resident = resident
        return resident

    @staticmethod
    def _command_matches(expected: tuple[str, ...], actual: str) -> bool:
        if not actual:
            return False
        try:
            observed = shlex.split(actual)
        except ValueError:
            return False
        return tuple(observed) == expected

    def _load_record(self) -> EngineResident | None:
        if self._resident is not None:
            return self._resident
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if raw.get("schema") != "turbofit.managed-engine/v1":
                return None
            return EngineResident(
                engine_id=str(raw["engine_id"]),
                model_id=str(raw["model_id"]),
                model_path=str(raw["model_path"]),
                port=int(raw["port"]),
                context_length=int(raw["context_length"]),
                bind_host=str(raw["bind_host"]),
                upstream_model_id=str(raw["upstream_model_id"]),
                pid=int(raw["pid"]),
                command=tuple(str(item) for item in raw["command"]),
                owned=raw["owned"] is True,
                status=str(raw.get("status") or "ready"),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def inspect(self) -> dict[str, Any]:
        resident = self._load_record()
        if resident is None:
            return {"ok": False, "status": "down", "runtime": None}
        payload = {
            "engine_id": resident.engine_id,
            "model_id": resident.model_id,
            "pid": resident.pid,
            "port": resident.port,
            "owned": resident.owned,
        }
        if resident.owned and not self._command_matches(
            resident.command, self.command_line(resident.pid)
        ):
            return {"ok": False, "status": "stale", "runtime": payload}
        if resident.pid not in self.listener_pids(resident.port):
            return {"ok": False, "status": "stale", "runtime": payload}
        spec = EngineLaunch(
            engine_id=resident.engine_id,
            model_id=resident.model_id,
            model_path=Path(resident.model_path),
            port=resident.port,
            context_length=resident.context_length,
            bind_host=resident.bind_host,
            upstream_model_id=resident.upstream_model_id,
            command=resident.command,
            required_model_files=(),
        )
        if not self._ready(spec):
            return {"ok": False, "status": "loading", "runtime": payload}
        return {"ok": True, "status": "ready", "runtime": payload}

    def stop(self, *, timeout_s: float = 30.0) -> bool:
        with self._lifecycle_lock():
            return self._stop(timeout_s=timeout_s)

    def _stop(self, *, timeout_s: float) -> bool:
        resident = self._load_record()
        if resident is None or not resident.owned:
            self._resident = None
            self.state_path.unlink(missing_ok=True)
            return True
        actual = self.command_line(resident.pid)
        if not self._command_matches(resident.command, actual):
            resident = replace(resident, status="command-mismatch")
            self._resident = resident
            self._write_owned(resident)
            return False
        if self._process is not None:
            self._process.terminate()
        else:
            self.signal_process(resident.pid, signal.SIGTERM)
        deadline = self.clock() + max(0.0, timeout_s)
        while self.clock() < deadline:
            actual = self.command_line(resident.pid)
            if not self._command_matches(resident.command, actual):
                self._resident = None
                self._process = None
                self.state_path.unlink(missing_ok=True)
                return True
            self.sleep(0.2)
        resident = replace(resident, status="stop-timeout")
        self._resident = resident
        self._write_owned(resident)
        return False


def publish_engine_routes(path: str | Path, spec: EngineLaunch) -> None:
    publish_route_state(
        path,
        {
            "schema": "turbofit.runtime-routes/v1",
            "active": f"apple-{spec.engine_id}",
            "rung_id": f"{spec.engine_id}-{spec.model_id}",
            "routes": {
                "main": {
                    "kind": "local",
                    "alias": spec.model_id,
                    "model_id": spec.upstream_model_id,
                    "port": spec.port,
                    "context_length": spec.context_length,
                    "engine": spec.engine_id,
                },
                "aux": {
                    "kind": "shared-main",
                    "context_length": spec.context_length,
                },
            },
        },
    )
