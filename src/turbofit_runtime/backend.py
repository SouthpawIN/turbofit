"""Native process backend for controlled campaign runs."""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .hardware import _nvidia_compatibility_library_dir
from .gpu import probe_gpus
from .artifacts import (
    describe_load_failure,
    expected_size_for_model_root,
    inspect_file,
    is_weight_path,
    log_tail,
    parse_shard,
    shard_names,
)
from .recipes import ResolvedComponent, ResolvedRecipe, resolve_native_backend
from .runtime_backends import llama_environment


class CampaignBackend:
    def __init__(
        self,
        *,
        gateway_script: Path,
        gateway_port: int = 18091,
        result_dir: Path,
        runtime_state: Path,
        production_gateway_service: str = "turbofit-gateway.service",
        production_controller_service: str = "turbofit-controller.service",
        campaign_lease_path: Path | None = None,
        accelerator_backend: str | None = None,
    ) -> None:
        self.gateway_script = gateway_script
        self.gateway_port = gateway_port
        self.result_dir = result_dir
        self.runtime_state = runtime_state
        self.production_gateway_service = production_gateway_service
        self.production_controller_service = production_controller_service
        self.campaign_lease_path = campaign_lease_path or (
            Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state"))
            / "turbofit/campaign-lease.json"
        )
        self.accelerator_backend = resolve_native_backend(accelerator_backend)
        self._handles: list[dict[str, Any]] = []
        self._gateway: subprocess.Popen[str] | None = None
        self._samples: list[list[dict[str, Any]]] = []
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._production_suspended = False
        self._production_services_to_restore: list[str] = []
        self._campaign_lease_depth = 0

    @staticmethod
    def _request_json(url: str, payload: dict | None = None, timeout: int = 10) -> tuple[int, dict, dict]:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response), dict(response.headers)
        except urllib.error.HTTPError as exc:
            return exc.code, {"error": exc.read().decode(errors="replace")}, {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"error": str(exc)}, {}

    @staticmethod
    def _port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.25)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def _port_in_tcp_tables(port: int) -> bool:
        target = f"{port:04X}"
        for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
            try:
                lines = table.read_text(encoding="utf-8").splitlines()[1:]
            except OSError:
                continue
            for line in lines:
                columns = line.split()
                if len(columns) > 3 and columns[1].rsplit(":", 1)[-1] == target:
                    return True
        return False

    def _wait_port_reusable(self, port: int, timeout_s: float = 90.0) -> None:
        deadline = time.monotonic() + timeout_s
        clear_samples = 0
        while time.monotonic() < deadline:
            if not self._port_open(port) and not self._port_in_tcp_tables(port):
                clear_samples += 1
                if clear_samples >= 3:
                    return
            else:
                clear_samples = 0
            time.sleep(0.5)
        raise RuntimeError(f"port {port} did not become reusable within {timeout_s:.0f}s")


    def _start_monitor(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        if self.accelerator_backend == "cpu":
            return
        self._samples = []
        self._monitor_stop.clear()

        def monitor() -> None:
            while not self._monitor_stop.is_set():
                try:
                    snapshot = [sample.to_dict() for sample in probe_gpus()]
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                    snapshot = []
                if snapshot:
                    self._samples.append(snapshot)
                time.sleep(0.15)

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def _suspend_production(self) -> None:
        if self._production_suspended:
            return
        self.campaign_lease_path.parent.mkdir(parents=True, exist_ok=True)
        self.campaign_lease_path.write_text(json.dumps({
            "schema": "turbofit.campaign-lease/v1",
            "owner_pid": os.getpid(),
            "gateway_policy": "api-fallback-only",
        }, sort_keys=True) + "\n", encoding="utf-8")
        # The production gateway does not own accelerator memory and must remain
        # available so requests can fall back to the configured API while a
        # physical campaign has exclusive ownership of local model processes.
        # Only the controller can repopulate managed residency, so only it is
        # suspended by the campaign lease.
        services = (self.production_controller_service,)
        self._production_services_to_restore = []
        if sys.platform == "win32":
            self._production_suspended = True
            return
        try:
            for service in services:
                try:
                    active = subprocess.run(
                        ["systemctl", "--user", "is-active", "--quiet", service],
                        capture_output=True,
                    ).returncode == 0
                except FileNotFoundError:
                    active = False
                if active:
                    self._production_services_to_restore.append(service)
            if self._production_services_to_restore:
                subprocess.run(
                    ["systemctl", "--user", "stop", *self._production_services_to_restore],
                    check=True, capture_output=True, text=True,
                )
        except FileNotFoundError:
            self._production_services_to_restore = []
            self._production_suspended = True
            return
        except Exception:
            self._production_services_to_restore = []
            self.campaign_lease_path.unlink(missing_ok=True)
            raise
        self._production_suspended = True

    def _resume_production(self) -> None:
        if not self._production_suspended:
            return
        services = list(self._production_services_to_restore)
        self._production_services_to_restore = []
        self._production_suspended = False
        if services:
            subprocess.run(
                ["systemctl", "--user", "start", *services],
                check=True, capture_output=True, text=True,
            )
        self.campaign_lease_path.unlink(missing_ok=True)

    def acquire_campaign_lease(self) -> None:
        self._suspend_production()
        self._campaign_lease_depth += 1

    def release_campaign_lease(self) -> None:
        if self._campaign_lease_depth <= 0:
            raise RuntimeError("campaign lease is not held")
        self._campaign_lease_depth -= 1
        if self._campaign_lease_depth == 0 and not self._handles:
            self._resume_production()

    @staticmethod
    def process_environment(
        command: tuple[str, ...], *, gpu: str, base: dict[str, str] | None = None,
        platform_name: str | None = None,
        backend_name: str = "cuda",
        compatibility_library_dir: Callable[[], str | None] | None = None,
    ) -> dict[str, str]:
        platform_id = platform_name or sys.platform
        if platform_id == "darwin":
            backend_name = "metal"
        env = llama_environment(backend_name, devices=gpu, base=base)
        if backend_name == "cuda":
            find_compatibility_dir = (
                compatibility_library_dir or _nvidia_compatibility_library_dir
            )
            compatibility_dir = find_compatibility_dir()
            if compatibility_dir:
                existing = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = compatibility_dir + (
                    f":{existing}" if existing else ""
                )
        if command:
            binary_path = Path(command[0]).resolve()
            binary_dir = binary_path.parent
            build_root = binary_dir.parent if binary_dir.name == "bin" else binary_dir
            library_dirs = (
                {path.parent for path in build_root.rglob("lib*.so*")}
                if binary_path.exists() else set()
            )
            if library_dirs:
                existing = env.get("LD_LIBRARY_PATH", "")
                ordered = sorted(library_dirs, key=lambda path: (path != binary_dir, str(path)))
                prefix = ":".join(str(path) for path in ordered)
                env["LD_LIBRARY_PATH"] = prefix + (f":{existing}" if existing else "")
        return env

    def start(self, component: ResolvedComponent) -> dict[str, Any]:
        self._suspend_production()
        try:
            self._wait_port_reusable(component.port)
            self._start_monitor()
            self.result_dir.mkdir(parents=True, exist_ok=True)
            if component.kind != "process":
                raise RuntimeError(f"unsupported native component kind: {component.kind}")
            model = Path(component.model_path) if component.model_path else None
            if model is not None and is_weight_path(model):
                # Checkpoints that are directories (vLLM/MLX) are validated by
                # the engine; single weight files are where placeholders hide.
                expected = expected_size_for_model_root(model)
                state = inspect_file(model, expected)
                if not state.present:
                    # Catch placeholders/partial downloads before llama.cpp
                    # prints a bare "failed to load model" the user cannot act on.
                    raise RuntimeError(describe_load_failure(
                        role=component.role,
                        model=component.family or component.alias,
                        file=model,
                        expected_bytes=expected,
                    ))
            log_path = self.result_dir / f"campaign-{component.role}-{component.port}.log"
            # One log per start: two engines writing one file (a retry, or a
            # stray process on the same port) interleave their lines and destroy
            # the evidence used to explain a readiness failure.
            if log_path.exists():
                for index in range(2, 1000):
                    candidate = log_path.with_name(
                        f"campaign-{component.role}-{component.port}-{index}.log"
                    )
                    if not candidate.exists():
                        log_path = candidate
                        break
            log = log_path.open("w")
            env = self.process_environment(
                component.command,
                gpu=component.gpu,
                backend_name=self.accelerator_backend,
            )
            try:
                process = subprocess.Popen(
                    component.command, env=env, stdout=log, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True,
                )
            except FileNotFoundError as exc:
                # A missing engine binary surfaces as a bare OSError with no path,
                # which reads like a model problem. Name the binary, the family and
                # the runtime directory so the operator knows what to install.
                log.close()
                binary = str(component.command[0]) if component.command else "<empty>"
                raise RuntimeError(
                    f"engine binary not found for {component.family or component.alias}: {binary!r}. "
                    "Install the runtime (scripts/install.sh) or set the engine path in "
                    "references/model-recipes.json for this family."
                ) from exc
            log.close()
            handle = {"kind": "process", "pid": process.pid, "process": process, "port": component.port, "log": str(log_path)}
            self._handles.append(handle)
            return handle
        except Exception:
            if not self._handles and self._campaign_lease_depth == 0:
                self._resume_production()
            raise

    def _load_failure(self, component: ResolvedComponent, handle: dict[str, Any], exit_code: int | None = None) -> str:
        """Explain a load exit with model, file, sizes and the engine's own log.

        The engine log only says ``failed to load model``; that reads like a
        runtime bug when the real state is a 15-byte placeholder left by an
        interrupted download.  The message never guesses a cause when the log
        has something to say, and it always quotes the log tail.
        """
        path = Path(component.model_path) if component.model_path else None
        expected = expected_size_for_model_root(path) if path else None
        missing: list[str] = []
        if path is not None:
            parsed = parse_shard(path.name)
            if parsed is not None:
                for name in shard_names(path.name, parsed[1]):
                    sibling = path.with_name(name)
                    if not inspect_file(sibling).present:
                        missing.append(name)
        return describe_load_failure(
            role=component.role,
            model=component.family or component.alias,
            file=path,
            expected_bytes=expected,
            log_path=handle.get("log"),
            missing_shards=missing,
            exit_code=exit_code,
        )

    #: Lines of engine log quoted when a readiness failure is reported.
    LOG_TAIL_LINES = 15
    #: Readiness budget.  A rebuilt engine JITs CUDA kernels and can spend many
    #: minutes before it binds its port, so this is generous by design.
    READY_TIMEOUT_S = 1800
    #: Markers that mean the engine never got the port, not that it failed to
    #: read the weights.
    _BIND_ERROR_MARKERS = (
        "couldn't bind",
        "could not bind",
        "failed to bind",
        "address already in use",
        "eaddrinuse",
    )

    def _probe_readiness(self, component: ResolvedComponent) -> tuple[str, dict]:
        """Ask the serving port whether a model is up.

        Returns ``(state, payload)`` where state is ``"ready"`` (server answered
        ``/health`` and ``/v1/models``), ``"loading"`` (health answered OK but
        the model list is not ready yet), ``"starting"`` (no healthy answer
        yet), ``"foreign"`` (a *different* model id is being served on this
        port) or ``"down"`` (no HTTP answer at all).
        """
        base = f"http://127.0.0.1:{component.port}"
        code, health, _ = self._request_json(f"{base}/health", timeout=3)
        healthy = code == 200 and (
            health.get("status") == "ok" or health.get("ok") is True
        )
        if not healthy:
            if code not in (404, 405):
                return ("starting" if code else "down"), health
            # Engines without /health: fall back to the model list.
        model_code, models, _ = self._request_json(f"{base}/v1/models", timeout=10)
        data = models.get("data") or []
        if model_code != 200 or not data:
            return ("loading" if healthy else "starting"), models
        served = str(data[0].get("id") or "")
        if component.alias and served and served != component.alias:
            return "foreign", {"model": served}
        return "ready", {
            "context": int((data[0].get("meta") or {}).get("n_ctx", 0)),
            "model": served,
            "health": health,
        }

    def _readiness_failure(self, component: ResolvedComponent, handle: dict[str, Any], exit_code: int | None) -> str:
        """Classify an exit that happened before the port answered.

        A port that is already owned by another engine is not a load failure,
        and saying so matters: the weights can be perfect while the run dies.
        """
        log_path = handle.get("log")
        tail = log_tail(log_path, limit=self.LOG_TAIL_LINES)
        blob = "\n".join(tail).lower()
        if any(marker in blob for marker in self._BIND_ERROR_MARKERS):
            return (
                f"{component.role} could not bind port {component.port}: the engine exited"
                f"{f' with code {exit_code}' if exit_code is not None else ''} because another "
                f"process already owns the port, most likely a stray engine from an earlier run. "
                f"Stop it (`pkill -f llama-server`) and re-run. Engine log {log_path} "
                f"(last {len(tail)} lines):\n" + "\n".join(tail)
            )
        return self._load_failure(component, handle, exit_code=exit_code)

    def _foreign_port_message(self, component: ResolvedComponent, served: str, handle: dict[str, Any]) -> str:
        return (
            f"{component.role} port {component.port} is answering as model {served!r} but this run "
            f"expects {component.alias!r}: a stray engine from an earlier run still owns the port. "
            f"Stop it (`pkill -f llama-server`) and re-run. Engine log of the new process: "
            f"{handle.get('log')}"
        )

    def wait_ready(self, component: ResolvedComponent, handle: dict[str, Any]) -> dict:
        deadline = time.monotonic() + self.READY_TIMEOUT_S
        last: dict = {}
        loaded = False
        while time.monotonic() < deadline:
            state, payload = self._probe_readiness(component)
            last = payload
            if state == "ready":
                return payload
            if state == "foreign":
                if handle["process"].poll() is not None:
                    raise RuntimeError(self._foreign_port_message(component, payload["model"], handle))
                # Our own process is up and serving; an alias difference is the
                # engine's report, and the caller already validates it.
                return {
                    "context": 0,
                    "model": payload["model"],
                    "health": {"status": "ok"},
                }
            if state == "loading":
                # llama.cpp answers 503 on /health until the weights are loaded
                # and 200 afterwards, so a 200 proves the load succeeded.
                loaded = True
            exit_code = handle["process"].poll()
            if exit_code is not None:
                # The engine can answer and then exit (clean shutdown after
                # serving, or a supervisor handing the port over).  Only call it
                # a failure once the port really stopped answering, and never
                # call it a load failure when the model demonstrably loaded.
                for _ in range(3):
                    state, payload = self._probe_readiness(component)
                    last = payload
                    if state == "ready":
                        return payload
                    if state == "foreign":
                        raise RuntimeError(self._foreign_port_message(component, payload["model"], handle))
                    if state == "loading":
                        loaded = True
                    time.sleep(0.5)
                if loaded and exit_code == 0:
                    # It answered /health and shut down cleanly: the load
                    # succeeded, the process just did not outlive readiness.
                    return {
                        "context": 0,
                        "model": str((last or {}).get("model") or component.alias),
                        "health": {"status": "ok"},
                        "exited_cleanly": True,
                    }
                if loaded:
                    tail = log_tail(handle.get("log"), limit=self.LOG_TAIL_LINES)
                    raise RuntimeError(
                        f"{component.role} exited with code {exit_code} on port {component.port} "
                        f"after loading the model but before readiness completed; the weights are "
                        f"not at fault. Engine log {handle.get('log')} (last {len(tail)} lines):\n"
                        + "\n".join(tail)
                    )
                raise RuntimeError(self._readiness_failure(component, handle, exit_code))
            time.sleep(2)
        raise RuntimeError(
            f"{component.role} failed readiness on {component.port} after "
            f"{self.READY_TIMEOUT_S}s: {last}"
        )

    def route(self, recipe: ResolvedRecipe, handles: dict[str, Any]) -> dict:
        components = []
        for component in recipe.components:
            handle = handles[component.role]
            components.append({
                "role": component.role, "kind": handle.get("kind"),
                "name": handle.get("name", f"campaign-{component.role}"),
                "pid": handle.get("pid"), "port": component.port,
            })
        self.runtime_state.parent.mkdir(parents=True, exist_ok=True)
        self.runtime_state.write_text(json.dumps({
            "active": f"campaign:{recipe.profile_name}",
            "context": 0,
            "expected": {
                "main_alias": recipe.main_alias,
                "aux_alias": recipe.aux_alias,
                "aux_mode": recipe.aux_mode,
                "aux_residency": recipe.aux_residency,
            },
            "components": components,
            "activating": True,
        }, indent=2) + "\n")
        env = os.environ.copy()
        env.update({
            "TURBOFIT_GATEWAY_PORT": str(self.gateway_port),
            "TURBOFIT_GATEWAY_HOST": "0.0.0.0",
            "TURBOFIT_RUNTIME_STATE": str(self.runtime_state),
            "TURBOFIT_CAMPAIGN_GATEWAY": "true",
            "TURBOFIT_BACKEND_TIMEOUT": "3600",
        })
        log_path = self.result_dir / "campaign-gateway.log"
        log = log_path.open("w")
        self._gateway = subprocess.Popen(["python3", str(self.gateway_script)], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        log.close()
        deadline = time.monotonic() + 90
        last = {}
        while time.monotonic() < deadline:
            code, status, _ = self._request_json(f"http://127.0.0.1:{self.gateway_port}/status", timeout=5)
            last = status
            if code == 200:
                return {"main": status.get("main", {}).get("alias"), "aux": status.get("aux", {}).get("alias"), "status": status}
            if self._gateway.poll() is not None:
                break
            time.sleep(1)
        raise RuntimeError(f"campaign gateway failed: {last}; log={log_path}")

    def infer(self, role: str, recipe: ResolvedRecipe) -> dict:
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Implement merge sort in Python with type hints and explain six design choices."}],
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
            "reasoning": False,
        }
        code, body, headers = self._request_json(
            f"http://127.0.0.1:{self.gateway_port}/{role}/v1/chat/completions",
            payload=payload,
            timeout=3660,
        )
        if code != 200:
            raise RuntimeError(f"{role} inference failed ({code}): {body}")
        return {
            "backend": headers.get("X-Turbofit-Backend"),
            "content": ((body.get("choices") or [{}])[0].get("message") or {}).get("content", ""),
            "usage": body.get("usage", {}),
            "timings": body.get("timings", {}),
        }

    def peak_gpu_mb(self) -> dict[int, int]:
        peak: dict[int, int] = {}
        for snapshot in self._samples:
            for row in snapshot:
                peak[row["gpu"]] = max(peak.get(row["gpu"], 0), row["used_mb"])
        return peak

    def stop(self, component: ResolvedComponent, handle: dict[str, Any]) -> None:
        try:
            process: subprocess.Popen[str] = handle["process"]
            if process.poll() is None:
                try:
                    if platform.system() == "Windows":
                        subprocess.run(
                            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                            check=False, capture_output=True, text=True,
                        )
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        if platform.system() == "Windows":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        finally:
            if handle in self._handles:
                self._handles.remove(handle)
            if not self._handles:
                self._monitor_stop.set()
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)
                if self._gateway and self._gateway.poll() is None:
                    try:
                        if platform.system() == "Windows":
                            subprocess.run(
                                ["taskkill.exe", "/PID", str(self._gateway.pid), "/T", "/F"],
                                check=False, capture_output=True, text=True,
                            )
                        else:
                            os.killpg(self._gateway.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try: self._gateway.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            if platform.system() == "Windows":
                                self._gateway.kill()
                            else:
                                os.killpg(self._gateway.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                self.runtime_state.write_text(json.dumps({"active": None, "components": []}, indent=2) + "\n")
                if self._campaign_lease_depth == 0:
                    self._resume_production()
