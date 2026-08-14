"""Native llama.cpp effect backend for adaptive Turbofit transitions."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .backend import CampaignBackend
from .recipes import RecipeBook, ResolvedComponent
from .reconciler import ReconcileError, ReconcilerState
from .routes import RuntimeResolutions, build_route_state, publish_route_state
from .runtime_profile import AuxMode, Turbofile


@dataclass
class OwnedRuntime:
    role: str
    pid: int
    alias: str
    port: int
    command: tuple[str, ...]
    process: subprocess.Popen[str] | None = None


class NativeRuntimeBackend:
    """Own only Turbofit's pinned loopback llama.cpp processes."""

    def __init__(
        self,
        *,
        profile: Turbofile,
        resolutions: RuntimeResolutions,
        recipe_book: RecipeBook,
        route_state_path: str | Path,
        state_dir: str | Path,
        manager_port: int,
        current_state: ReconcilerState,
        accelerator_backend: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        verification_timeout_s: float = 900.0,
    ) -> None:
        if current_state.profile_id != profile.id:
            raise ValueError("backend state profile does not match profile")
        if verification_timeout_s < 0:
            raise ValueError("verification_timeout_s must be non-negative")
        self.profile = profile
        self.resolutions = resolutions
        self.recipe_book = recipe_book
        self.route_state_path = Path(route_state_path)
        self.state_dir = Path(state_dir)
        self.manager_port = manager_port
        self.current_state = current_state
        self.accelerator_backend = accelerator_backend or recipe_book.backend_name
        self.sleep = sleep
        self.clock = clock
        self.verification_timeout_s = verification_timeout_s
        self._target_rung_id: str | None = None
        self._target_aux_mode: AuxMode | None = None
        self._blocked_previous: dict[str, Any] | None = None
        self._retiring_aux: OwnedRuntime | None = None
        self._owned: dict[str, OwnedRuntime] = {}
        self._recover_owned()

    def _pid_path(self, role: str) -> Path:
        return self.state_dir / f"{role}.json"

    def _recover_owned(self) -> None:
        for role in ("main", "aux"):
            try:
                data = json.loads(self._pid_path(role).read_text(encoding="utf-8"))
                runtime = OwnedRuntime(
                    role=role,
                    pid=int(data["pid"]),
                    alias=str(data["alias"]),
                    port=int(data["port"]),
                    command=tuple(str(item) for item in data["command"]),
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if self._is_owned(runtime):
                self._owned[role] = runtime
            else:
                self._pid_path(role).unlink(missing_ok=True)

    def _write_owned(self, runtime: OwnedRuntime) -> None:
        publish_route_state(self._pid_path(runtime.role), {
            "pid": runtime.pid,
            "alias": runtime.alias,
            "port": runtime.port,
            "command": list(runtime.command),
        })

    @staticmethod
    def _command_line(pid: int) -> str:
        proc = Path(f"/proc/{pid}/cmdline")
        if proc.is_file():
            try:
                return proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                return ""
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    def _is_owned(self, runtime: OwnedRuntime) -> bool:
        command_line = self._command_line(runtime.pid)
        if not command_line or not runtime.command:
            return False
        return (
            Path(runtime.command[0]).name in command_line
            and f"--alias {runtime.alias}" in command_line
            and f"--port {runtime.port}" in command_line
        )

    def _roles(self, rung_id: str) -> dict[str, dict[str, int | str]]:
        try:
            return self.resolutions[self.profile.id][rung_id]
        except KeyError as exc:
            raise ReconcileError(
                f"missing native runtime resolution for {self.profile.id}/{rung_id}"
            ) from exc

    def _component(self, role: str, item: dict[str, int | str], context: int) -> ResolvedComponent:
        for key in ("family", "gpu", "port", "model_tag"):
            if key not in item:
                raise ReconcileError(f"native resolution for {role} lacks {key}")
        return self.recipe_book.resolve_component(
            str(item["family"]),
            role=role,
            gpu=str(item["gpu"]),
            port=int(item["port"]),
            context=context,
            alias=str(item["model_tag"]),
        )

    def _start(self, component: ResolvedComponent) -> OwnedRuntime:
        binary = Path(component.command[0])
        if not binary.is_file():
            raise ReconcileError(f"native runtime binary is missing: {binary}")
        for path in (component.model_path, component.projector_path):
            if path and not Path(path).is_file():
                raise ReconcileError(f"model artifact is missing: {path}")
        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"{component.role}.log").open("a", encoding="utf-8")
        kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "text": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(
            list(component.command),
            env=CampaignBackend.process_environment(
                component.command,
                gpu=component.gpu,
                backend_name=self.accelerator_backend,
            ),
            **kwargs,
        )
        log.close()
        runtime = OwnedRuntime(
            role=component.role,
            pid=process.pid,
            alias=component.alias,
            port=component.port,
            command=component.command,
            process=process,
        )
        self._write_owned(runtime)
        self._owned[component.role] = runtime
        return runtime

    @staticmethod
    def _url(runtime: OwnedRuntime, path: str) -> str:
        return f"http://127.0.0.1:{runtime.port}{path}"

    def _json(self, runtime: OwnedRuntime, path: str, timeout: float = 2.0) -> Any:
        with urllib.request.urlopen(self._url(runtime, path), timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _healthy(self, runtime: OwnedRuntime) -> bool:
        if runtime.process is not None and runtime.process.poll() is not None:
            return False
        try:
            health = self._json(runtime, "/health")
            models = self._json(runtime, "/v1/models")
        except (OSError, ValueError, urllib.error.URLError):
            return False
        ids = {str(item.get("id")) for item in models.get("data", [])}
        return health.get("status") == "ok" and runtime.alias in ids

    def _wait_healthy(self, runtime: OwnedRuntime) -> bool:
        deadline = self.clock() + self.verification_timeout_s
        while True:
            if self._healthy(runtime):
                return True
            if runtime.process is not None and runtime.process.poll() is not None:
                return False
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.5, max(0.0, deadline - self.clock())))

    def _processing(self, runtime: OwnedRuntime) -> int:
        try:
            with urllib.request.urlopen(self._url(runtime, "/metrics"), timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace")
        except OSError:
            return 0
        values = []
        for line in body.splitlines():
            if line.startswith("#") or "processing" not in line.lower():
                continue
            match = re.search(r"\s(-?\d+(?:\.\d+)?)$", line)
            if match:
                values.append(max(0, int(float(match.group(1)))))
        return max(values, default=0)

    def _stop(self, role: str, *, force: bool = False, timeout: float = 30.0) -> bool:
        runtime = self._owned.get(role)
        if runtime is None:
            self._pid_path(role).unlink(missing_ok=True)
            return True
        if not self._is_owned(runtime):
            self._owned.pop(role, None)
            self._pid_path(role).unlink(missing_ok=True)
            return True
        try:
            if runtime.process is not None and not force:
                runtime.process.terminate()
            else:
                os.kill(runtime.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = self.clock() + timeout
        while self._command_line(runtime.pid):
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.2, max(0.0, deadline - self.clock())))
        self._owned.pop(role, None)
        self._pid_path(role).unlink(missing_ok=True)
        return True

    def reset_managed(self) -> None:
        for role in ("aux", "main"):
            if not self._stop(role) and not self._stop(role, force=True, timeout=5):
                raise ReconcileError(f"could not stop owned {role} runtime")

    def block_aux_admission(self) -> None:
        try:
            state = json.loads(self.route_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReconcileError("cannot block auxiliary admission without valid routes") from exc
        routes = state.get("routes")
        if not isinstance(routes, dict) or not isinstance(routes.get("main"), dict):
            raise ReconcileError("cannot block auxiliary admission without valid routes")
        self._blocked_previous = state
        self._retiring_aux = self._owned.get("aux")
        staged = json.loads(json.dumps(state))
        staged["routes"]["aux"] = (
            {"kind": "shared-main"}
            if staged["routes"]["main"].get("kind") == "local"
            else {"kind": "api-policy", "policy": "api:auto"}
        )
        publish_route_state(self.route_state_path, staged)

    def drain_aux(self, timeout_s: float) -> bool:
        runtime = self._retiring_aux
        if runtime is None:
            return True
        deadline = self.clock() + timeout_s
        while self._processing(runtime) > 0:
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.25, max(0.0, deadline - self.clock())))
        return True

    def clean_unload_aux(self) -> bool:
        if self._retiring_aux is None:
            return True
        return self._stop("aux")

    def owned_pids(self) -> tuple[int, ...]:
        runtime = self._retiring_aux
        return (runtime.pid,) if runtime is not None and self._is_owned(runtime) else ()

    def escalate_owned(self, pids: tuple[int, ...]) -> None:
        owned = set(self.owned_pids())
        if set(pids) - owned:
            raise ReconcileError("refusing to signal a process not owned by Turbofit")
        if pids and not self._stop("aux", force=True, timeout=5):
            raise ReconcileError("owned auxiliary runtime survived forced termination")

    def activate_local(self, rung_id: str) -> None:
        rung = next((item for item in self.profile.rungs if item.id == rung_id), None)
        if rung is None or rung.aux_mode is AuxMode.API:
            raise ReconcileError(f"invalid local rung: {rung_id}")
        roles = self._roles(rung_id)
        self.reset_managed()
        for role in ("aux", "main"):
            item = roles.get(role)
            if item is not None:
                self._start(self._component(role, item, rung.context))
        self._target_rung_id = rung_id

    def activate_api(self, main_policy: str, aux_policy: str) -> None:
        if main_policy != "api:auto" or aux_policy != "api:auto":
            raise ReconcileError("unsupported API policy")
        self.reset_managed()
        self._target_rung_id = self.profile.rungs[-1].id
        self._target_aux_mode = AuxMode.API

    def route_aux_to_main(self) -> None:
        self._target_aux_mode = AuxMode.SHARED_MAIN

    def route_aux_dedicated(self) -> None:
        self._target_aux_mode = AuxMode.DEDICATED

    def verify_rung(self, rung_id: str) -> bool:
        rung = next((item for item in self.profile.rungs if item.id == rung_id), None)
        if rung is None:
            return False
        if rung.aux_mode is AuxMode.API:
            return self._target_aux_mode is AuxMode.API and not self._owned
        roles = self._roles(rung_id)
        if set(roles) != set(self._owned):
            return False
        return all(self._wait_healthy(runtime) for runtime in self._owned.values())

    def publish_routes(self, state: ReconcilerState) -> None:
        publish_route_state(
            self.route_state_path,
            build_route_state(
                self.profile,
                state.rung_index,
                self.resolutions,
                manager_port=self.manager_port,
            ),
        )
        self.current_state = state
        self._blocked_previous = None
        self._retiring_aux = None

    def restore(self, state: ReconcilerState) -> None:
        rung = self.profile.rungs[state.rung_index]
        if rung.aux_mode is AuxMode.API:
            assert rung.main_api_policy and rung.aux_api_policy
            self.activate_api(rung.main_api_policy, rung.aux_api_policy)
        else:
            self.activate_local(rung.id)
            if rung.aux_mode is AuxMode.SHARED_MAIN:
                self.route_aux_to_main()
            else:
                self.route_aux_dedicated()
        if not self.verify_rung(rung.id):
            raise ReconcileError(f"could not restore rung {rung.id}")
        self.publish_routes(state)

    def verify_restore(self, state: ReconcilerState) -> bool:
        try:
            published = json.loads(self.route_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            published.get("active") == state.profile_id
            and published.get("rung_index") == state.rung_index
            and self.verify_rung(self.profile.rungs[state.rung_index].id)
        )
