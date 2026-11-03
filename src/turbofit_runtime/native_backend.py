"""Native llama.cpp effect backend for adaptive Turbofit transitions."""
from __future__ import annotations

import json
import os
import shlex
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .backend import CampaignBackend
from .recipes import RecipeBook, ResolvedComponent
from .reconciler import ReconcileError, ReconcilerState
from .routes import RuntimeResolutions, build_route_state, publish_route_state
from .runtime_profile import LAUNCHER_SYSTEMD_UNIT, AuxMode, Turbofile


@dataclass
class OwnedRuntime:
    role: str
    pid: int
    alias: str
    port: int
    command: tuple[str, ...]
    process: subprocess.Popen[str] | None = None
    start_identity: str | None = None


def process_start_identity(pid: int) -> str | None:
    """Return a stable process-start identity for ``pid``.

    Linux uses the /proc/<pid>/stat starttime field; Windows derives a
    comparable identity from the process creation time via GetProcessTimes.
    Anything that cannot be proven returns None (fail-safe: callers must
    refuse to signal).
    """
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError:
        raw = b""
    if raw:
        try:
            # The comm field may contain spaces; everything after the final ')' is stable.
            fields = raw[raw.rindex(b")") + 2:].split()
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"linux:{boot}:{int(fields[19])}"
        except (IndexError, ValueError, OSError):
            return None
    if os.name == "nt":
        try:  # pragma: no cover - exercised only on Windows
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                return f"windows:{creation.dwHighDateTime}_{creation.dwLowDateTime}"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    return None


def _without_mmproj(command: list[str] | tuple[str, ...]) -> list[str]:
    """Return the launch command with any ``--mmproj <path>`` pair removed."""
    items = list(command)
    cleaned: list[str] = []
    skip_next = False
    for item in items:
        if skip_next:
            skip_next = False
            continue
        if item == "--mmproj":
            skip_next = True
            continue
        cleaned.append(item)
    return cleaned


def rung_artifact_problem(
    profile: Turbofile,
    rung_id: str,
    resolutions: RuntimeResolutions,
    recipe_book: RecipeBook,
    *,
    exists: Callable[[str], bool] | None = None,
) -> str | None:
    """Return why a rung cannot launch on this box, or ``None`` when it can.

    Resolution failures and absent model files are reported rather than raised:
    one unavailable model is a smaller ladder, not a stopped runtime.
    """
    is_file = exists or (lambda path: Path(path).is_file())
    rung = next((item for item in profile.rungs if item.id == rung_id), None)
    if rung is None:
        return "rung is not in the selected profile"
    if rung.aux_mode is AuxMode.API:
        return None
    if rung.launcher == LAUNCHER_SYSTEMD_UNIT:
        # The unit owns the artifacts; nothing here needs to resolve a recipe.
        return None
    try:
        roles = resolutions[profile.id][rung_id]
    except KeyError:
        return f"missing native runtime resolution for {profile.id}/{rung_id}"
    for role, item in roles.items():
        for key in ("family", "gpu", "port", "model_tag"):
            if key not in item:
                return f"native resolution for {role} lacks {key}"
        try:
            component = recipe_book.resolve_component(
                str(item["family"]),
                role=role,
                gpu=str(item["gpu"]),
                port=int(item["port"]),
                context=int(item.get("context", rung.context)),
                alias=str(item["model_tag"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            return f"{role}: {exc}"
        if not is_file(component.command[0]):
            return f"{role}: native runtime binary is missing: {component.command[0]}"
        if component.model_path and not is_file(component.model_path):
            return f"{role}: model artifact is missing: {component.model_path}"
        # A missing projector costs vision, not the rung: the text model still
        # serves, so it must not make the ladder look unavailable.
    return None


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
        identity_reader: Callable[[int], Optional[str]] | None = None,
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
        self._identity_reader: Callable[[int], Optional[str]] = (
            identity_reader or process_start_identity
        )
        self._target_rung_id: str | None = None
        self._target_aux_mode: AuxMode | None = None
        self._blocked_previous: dict[str, Any] | None = None
        self._retiring_aux: OwnedRuntime | None = None
        self._owned: dict[str, OwnedRuntime] = {}
        self._lost_roles: set[str] = set()
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
                    start_identity=(
                        str(data["start_identity"]) if data.get("start_identity") else None
                    ),
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            if self._is_owned(runtime):
                self._owned[role] = runtime
            else:
                self._lost_roles.add(role)
                self._pid_path(role).unlink(missing_ok=True)

    def _write_owned(self, runtime: OwnedRuntime) -> None:
        publish_route_state(self._pid_path(runtime.role), {
            "pid": runtime.pid,
            "alias": runtime.alias,
            "port": runtime.port,
            "command": list(runtime.command),
            "start_identity": runtime.start_identity,
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

    def _is_owned(self, runtime) -> bool:
        # A process this backend started and still holds a live handle for is
        # ours by construction: poll() is None means that exact child is alive,
        # which a reused PID cannot fake. The identity/cmdline proofs below only
        # guard *recovered* records (a pid read back from disk with no live
        # handle), where signalling a reused PID would be unsafe.
        if runtime.process is not None:
            return runtime.process.poll() is None
        command_line = self._command_line(runtime.pid)
        if not command_line or not runtime.command:
            return False
        identity = self._identity_reader(runtime.pid)
        if runtime.start_identity is not None:
            # Strong proof: a reused PID with the same cmdline but a different
            # creation identity must never be treated as ours.
            if identity != runtime.start_identity:
                return False
        else:
            # A legacy record cannot prove which incarnation it originally owned.
            return False
        try:
            words = shlex.split(command_line)
            alias_at, port_at = words.index("--alias"), words.index("--port")
            return (Path(words[0]).name == Path(runtime.command[0]).name
                    and words[alias_at + 1] == runtime.alias
                    and words[port_at + 1] == str(runtime.port))
        except (ValueError, IndexError):
            return False

    def rung_problem(self, rung_id: str) -> str | None:
        """Report why this backend's current-profile rung cannot be launched."""
        return rung_artifact_problem(
            self.profile, rung_id, self.resolutions, self.recipe_book
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
            context=int(item.get("context", context)),
            alias=str(item["model_tag"]),
        )

    def _start(self, component: ResolvedComponent) -> OwnedRuntime:
        binary = Path(component.command[0])
        if not binary.is_file():
            raise ReconcileError(f"native runtime binary is missing: {binary}")
        command = list(component.command)
        if component.model_path and not Path(component.model_path).is_file():
            raise ReconcileError(f"model artifact is missing: {component.model_path}")
        if component.projector_path and not Path(component.projector_path).is_file():
            # Vision is optional: serve text-only instead of losing the rung.
            # Rebind the component so residency identity checks still match the
            # command that was actually launched.
            component = replace(
                component, command=_without_mmproj(component.command), projector_path=""
            )
            print(json.dumps({"warning": "projector missing; serving text-only",
                              "role": component.role}, sort_keys=True),
                  file=sys.stderr, flush=True)
        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"{component.role}.log").open("a", encoding="utf-8")
        kwargs: dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "text": True}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        with socket.socket() as probe:
            probe.settimeout(0.25)
            if probe.connect_ex(("127.0.0.1", component.port)) == 0:
                raise ReconcileError("native port is occupied; refusing to adopt or replace its owner")
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
            start_identity=self._identity_reader(process.pid),
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

    def _stop(self, role: str, *, force: bool = False, timeout: float = 30.0,
              require_owned: bool = False) -> bool:
        runtime = self._owned.get(role)
        if runtime is None:
            self._pid_path(role).unlink(missing_ok=True)
            return not require_owned
        if not self._is_owned(runtime):
            self._owned.pop(role, None)
            self._pid_path(role).unlink(missing_ok=True)
            return not require_owned
        # Re-verify identity immediately before signalling: the process may
        # have died and its PID been reused between the ownership check and
        # the signal. Identity proof must be current, not cached.
        if runtime.start_identity is not None:
            identity = self._identity_reader(runtime.pid)
            if identity != runtime.start_identity:
                self._owned.pop(role, None)
                self._pid_path(role).unlink(missing_ok=True)
                return not require_owned
        try:
            if runtime.process is not None and not force:
                runtime.process.terminate()
            else:
                os.kill(runtime.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            if require_owned:
                return False
        deadline = self.clock() + timeout
        while self._is_owned(runtime):
            if self.clock() >= deadline:
                return False
            self.sleep(min(0.2, max(0.0, deadline - self.clock())))
        if runtime.process is not None:
            try:
                runtime.process.wait(timeout=max(0.0, deadline - self.clock()))
            except subprocess.TimeoutExpired:
                return False
        self._owned.pop(role, None)
        self._pid_path(role).unlink(missing_ok=True)
        return True

    def residency_snapshot(self) -> dict:
        """Read owned identity and health, never adopt, launch, stop or repair.

        Absence is unknown: only the lifecycle owner's successful idle release
        can establish intentional unload. Port health alone is not ownership.
        """
        rung = self.profile.rungs[self.current_state.rung_index]
        if rung.aux_mode is AuxMode.API:
            return {}
        unit = self._unit_rung(rung)
        if unit is not None:
            ports = self._unit_ports(rung.id)
            ready = self._unit_active(unit) and all(self._port_healthy(port) for port in ports)
            return {
                role: {
                    "backing_model": str(item.get("model_tag") or rung.id),
                    "context_length": int(item.get("context", rung.context)),
                    "residency": "ready" if ready else "unknown",
                }
                for role, item in self._roles(rung.id).items()
            }
        result = {}
        for role, item in self._launch_roles(rung).items():
            component = self._component(role, item, rung.context)
            resident = self._owned.get(role)
            state = "error" if role in self._lost_roles else "unknown"
            if resident is not None:
                matches = (resident.alias == component.alias and resident.port == component.port
                           and resident.command == component.command
                           and self._is_owned(resident))
                state = "ready" if matches and self._healthy(resident) and self._is_owned(resident) else "error"
            result[role] = {"backing_model": component.alias,
                            "context_length": int(item.get("context", rung.context)),
                            "residency": state}
        if rung.aux_mode is AuxMode.SHARED_MAIN and "main" in result:
            # The auxiliary role is served by main; report it as such rather
            # than inventing a second residency it does not have.
            result["aux"] = dict(result["main"])
        return result

    def release_idle_role(self, role: str) -> str:
        """Distinguish verified unload from absence or ownership loss."""
        if role not in {"main", "aux"}:
            raise ValueError("invalid residency role")
        unit = self._unit_rung(self.profile.rungs[self.current_state.rung_index])
        if unit is not None:
            if not self._unit_active(unit):
                return "unknown"
            return "idle" if self._systemctl("stop", unit) else "error"
        resident = self._owned.get(role)
        if resident is None:
            return "error" if role in self._lost_roles else "unknown"
        if not self._is_owned(resident):
            return "error"
        return "idle" if self._stop(role, require_owned=True) else "error"

    def stop_role(self, role: str) -> bool:
        if role not in {"main", "aux"}:
            raise ValueError("invalid residency role")
        unit = self._unit_rung(self.profile.rungs[self.current_state.rung_index])
        if unit is not None:
            return self._systemctl("stop", unit) if self._unit_active(unit) else True
        return self._stop(role)

    def release_current(self) -> str:
        """Yield VRAM for an external GPU app by unloading our own current rung.

        Only PID-verified children this backend launched are signalled; an
        external process is never touched. Returns ``released`` when at least
        one owned runtime was stopped and confirmed gone, ``absent`` when
        nothing of ours was resident, and ``not-local`` for an API rung.
        """
        rung = self.profile.rungs[self.current_state.rung_index]
        if rung.aux_mode is AuxMode.API:
            return "not-local"
        unit = self._unit_rung(rung)
        if unit is not None:
            if not self._unit_active(unit):
                return "absent"
            if not self._systemctl("stop", unit) or self._unit_active(unit):
                raise ReconcileError(f"declared unit {unit} survived residency release")
            return "released"
        if not self._owned:
            return "absent"
        for role in ("aux", "main"):
            if role not in self._owned:
                continue
            if not self._stop(role, require_owned=True):
                raise ReconcileError(f"owned {role} runtime survived residency release")
        if self._owned:
            raise ReconcileError("residency release left an owned runtime running")
        return "released"

    def reset_managed(self) -> None:
        current = self.profile.rungs[self.current_state.rung_index]
        unit = self._unit_rung(current)
        if unit is not None and self._unit_active(unit):
            if not self._systemctl("stop", unit):
                raise ReconcileError(f"could not stop declared unit {unit}")
        for role in ("aux", "main"):
            if not self._stop(role) and not self._stop(role, force=True, timeout=5):
                raise ReconcileError(f"could not stop owned {role} runtime")

    def ensure_role(self, role: str) -> None:
        """Wake one current local role without unloading the other GPU lane."""
        rung = self.profile.rungs[self.current_state.rung_index]
        if role not in {"main", "aux"}:
            raise ReconcileError("no native resolution for requested role")
        if role == "aux" and rung.aux_mode is AuxMode.SHARED_MAIN:
            role = "main"
        item = self._launch_roles(rung).get(role)
        if item is None:
            raise ReconcileError("no native resolution for requested role")
        component = self._component(role, item, rung.context)
        resident = self._owned.get(role)
        if (resident and resident.alias == component.alias and resident.port == component.port
                and resident.command == component.command
                and self._is_owned(resident) and self._healthy(resident) and self._is_owned(resident)):
            return
        if not self._stop(role):
            raise ReconcileError("cannot replace unhealthy owned runtime")
        try:
            resident = self._start(component)
            deadline = self.clock() + self.verification_timeout_s
            while not self._healthy(resident):
                if self.clock() >= deadline or not self._is_owned(resident):
                    raise ReconcileError("native wake verification failed or timed out")
                self.sleep(min(0.2, max(0.0, deadline - self.clock())))
            if not self._is_owned(resident):
                raise ReconcileError("native wake ownership lost after health verification")
            self._lost_roles.discard(role)
        except Exception:
            if not self._stop(role) and not self._stop(role, force=True, timeout=5):
                raise ReconcileError("failed wake child could not be reclaimed")
            raise

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

    # --- externally-managed rungs -------------------------------------------
    #
    # A rung may declare `launcher: systemd-unit` + `unit: <name>`. The unit,
    # not a PID, is then the lifecycle authority: we start and stop it through
    # `systemctl --user` and never signal its process. Declaring the unit in the
    # portable profile is what makes that authority explicit and reviewable.

    @staticmethod
    def _unit_rung(rung) -> str | None:
        return rung.unit if rung.launcher == LAUNCHER_SYSTEMD_UNIT else None

    def _systemctl(self, action: str, unit: str) -> bool:
        try:
            done = subprocess.run(
                ["systemctl", "--user", action, unit],
                text=True, capture_output=True, timeout=180, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0

    def _unit_active(self, unit: str) -> bool:
        try:
            done = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", unit],
                text=True, capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return done.returncode == 0

    def _unit_ports(self, rung_id: str) -> tuple[int, ...]:
        try:
            roles = self._roles(rung_id)
        except ReconcileError:
            return ()
        ports: list[int] = []
        for item in roles.values():
            try:
                ports.append(int(item["port"]))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(sorted(set(ports)))

    @staticmethod
    def _port_healthy(port: int, timeout: float = 2.0) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as handle:
                return 200 <= getattr(handle, "status", 200) < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _wait_unit_healthy(self, rung) -> bool:
        unit = self._unit_rung(rung)
        if unit is None or not self._unit_active(unit):
            return False
        ports = self._unit_ports(rung.id)
        if not ports:
            return True
        deadline = self.clock() + self.verification_timeout_s
        while True:
            if all(self._port_healthy(port) for port in ports):
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(min(1.0, max(0.0, deadline - self.clock())))

    def _launch_roles(self, rung) -> dict[str, dict[str, int | str]]:
        """Roles that own a process for this rung.

        A ``shared-main`` rung serves auxiliary requests from the main process.
        An aux resolution on such a rung is metadata describing that shared
        lane, not a second server: starting it would double VRAM for a rung the
        profile calls shared.
        """
        roles = self._roles(rung.id)
        if rung.aux_mode is AuxMode.SHARED_MAIN and "main" in roles:
            return {"main": roles["main"]}
        return roles

    def activate_local(self, rung_id: str) -> None:
        rung = next((item for item in self.profile.rungs if item.id == rung_id), None)
        if rung is None or rung.aux_mode is AuxMode.API:
            raise ReconcileError(f"invalid local rung: {rung_id}")
        unit = self._unit_rung(rung)
        self.reset_managed()
        if unit is not None:
            if not self._unit_active(unit) and not self._systemctl("start", unit):
                raise ReconcileError(f"could not start declared unit {unit}")
            if not self._wait_unit_healthy(rung):
                raise ReconcileError(f"declared unit {unit} did not become healthy")
            self._target_rung_id = rung_id
            self._target_aux_mode = rung.aux_mode
            return
        roles = self._launch_roles(rung)
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
        if self._unit_rung(rung) is not None:
            unit = self._unit_rung(rung)
            assert unit is not None
            return self._unit_active(unit) and all(
                self._port_healthy(port) for port in self._unit_ports(rung_id)
            )
        roles = self._launch_roles(rung)
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
