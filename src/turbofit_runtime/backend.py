"""Native process backend for controlled campaign runs."""
from __future__ import annotations

import json
import os
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
        self._samples = []
        self._monitor_stop.clear()

        def monitor() -> None:
            while not self._monitor_stop.is_set():
                result = subprocess.run([
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,fan.speed",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, text=True)
                snapshot = []
                for line in result.stdout.strip().splitlines():
                    values = [item.strip() for item in line.split(",")]
                    if len(values) == 6:
                        snapshot.append({
                            "gpu": int(values[0]), "used_mb": int(values[1]),
                            "free_mb": int(values[2]), "util_pct": int(values[3]),
                            "power_w": float(values[4]), "fan_pct": int(values[5]),
                        })
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
        try:
            for service in services:
                active = subprocess.run(
                    ["systemctl", "--user", "is-active", "--quiet", service],
                    capture_output=True,
                ).returncode == 0
                if active:
                    self._production_services_to_restore.append(service)
            if self._production_services_to_restore:
                subprocess.run(
                    ["systemctl", "--user", "stop", *self._production_services_to_restore],
                    check=True, capture_output=True, text=True,
                )
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
            model = Path(component.model_path)
            if not model.exists():
                raise FileNotFoundError(model)
            log_path = self.result_dir / f"campaign-{component.role}-{component.port}.log"
            log = log_path.open("w")
            env = self.process_environment(
                component.command,
                gpu=component.gpu,
                backend_name=self.accelerator_backend,
            )
            process = subprocess.Popen(component.command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            log.close()
            handle = {"kind": "process", "pid": process.pid, "process": process, "port": component.port, "log": str(log_path)}
            self._handles.append(handle)
            return handle
        except Exception:
            if not self._handles and self._campaign_lease_depth == 0:
                self._resume_production()
            raise

    def wait_ready(self, component: ResolvedComponent, handle: dict[str, Any]) -> dict:
        deadline = time.monotonic() + 1800
        last: dict = {}
        while time.monotonic() < deadline:
            code, health, _ = self._request_json(f"http://127.0.0.1:{component.port}/health", timeout=3)
            last = health
            if code == 200 and (health.get("status") == "ok" or health.get("ok") is True):
                model_code, models, _ = self._request_json(f"http://127.0.0.1:{component.port}/v1/models", timeout=10)
                data = models.get("data") or []
                if model_code == 200 and data:
                    return {
                        "context": int((data[0].get("meta") or {}).get("n_ctx", 0)),
                        "model": data[0].get("id"),
                        "health": health,
                    }
            if handle["process"].poll() is not None:
                raise RuntimeError(f"{component.role} process exited during load; log={handle.get('log')}")
            time.sleep(2)
        raise RuntimeError(f"{component.role} failed readiness on {component.port}: {last}")

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
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        code, body, headers = self._request_json(
            f"http://127.0.0.1:{self.gateway_port}/{role}/v1/chat/completions",
            payload=payload,
            timeout=600,
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
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
        finally:
            if handle in self._handles:
                self._handles.remove(handle)
            if not self._handles:
                self._monitor_stop.set()
                if self._monitor_thread:
                    self._monitor_thread.join(timeout=5)
                if self._gateway and self._gateway.poll() is None:
                    try: os.killpg(self._gateway.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: self._gateway.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(self._gateway.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                self.runtime_state.write_text(json.dumps({"active": None, "components": []}, indent=2) + "\n")
                if self._campaign_lease_depth == 0:
                    self._resume_production()
