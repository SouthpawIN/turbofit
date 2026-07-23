"""Real Docker/process backend for controlled campaign runs."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .recipes import ResolvedComponent, ResolvedRecipe


class CampaignBackend:
    def __init__(
        self,
        *,
        gateway_script: Path,
        gateway_port: int = 18091,
        result_dir: Path,
        runtime_state: Path,
        production_gateway_service: str = "turbofit-gateway.service",
    ) -> None:
        self.gateway_script = gateway_script
        self.gateway_port = gateway_port
        self.result_dir = result_dir
        self.runtime_state = runtime_state
        self.production_gateway_service = production_gateway_service
        self._handles: list[dict[str, Any]] = []
        self._gateway: subprocess.Popen[str] | None = None
        self._samples: list[list[dict[str, Any]]] = []
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

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

    def start(self, component: ResolvedComponent) -> dict[str, Any]:
        if self._port_open(component.port):
            raise RuntimeError(f"port {component.port} is occupied before {component.role} launch")
        self._start_monitor()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if component.kind == "docker":
            name = f"turbofit-campaign-{component.role}"
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            command = ["docker", "run", "-d", "--name", name, "--gpus", f"device={component.gpu}", "--network", "host"]
            for key, value in (component.environment or {}).items():
                command.extend(["-e", f"{key}={value}"])
            for mount in component.mounts:
                source = Path(mount.split(":", 1)[0])
                if not source.exists():
                    raise FileNotFoundError(source)
                command.extend(["-v", mount])
            command.append(component.image)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"docker launch failed for {component.role}: {result.stderr.strip()}")
            handle = {"kind": "docker", "name": name, "id": result.stdout.strip(), "port": component.port}
        else:
            model = Path(component.model_path)
            if not model.exists():
                raise FileNotFoundError(model)
            log_path = self.result_dir / f"campaign-{component.role}-{component.port}.log"
            log = log_path.open("w")
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = component.gpu
            process = subprocess.Popen(component.command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            log.close()
            handle = {"kind": "process", "pid": process.pid, "process": process, "port": component.port, "log": str(log_path)}
        self._handles.append(handle)
        return handle

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
            if handle.get("kind") == "process" and handle["process"].poll() is not None:
                raise RuntimeError(f"{component.role} process exited during load; log={handle.get('log')}")
            time.sleep(2)
        raise RuntimeError(f"{component.role} failed readiness on {component.port}: {last}")

    def route(self, recipe: ResolvedRecipe, handles: dict[str, Any]) -> dict:
        subprocess.run(["systemctl", "--user", "stop", self.production_gateway_service], capture_output=True, text=True)
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
            "TURBOFIT_RUNTIME_STATE": str(self.runtime_state),
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
            "max_tokens": 256,
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
            if handle.get("kind") == "docker":
                subprocess.run(["docker", "rm", "-f", str(handle["name"])], capture_output=True, text=True)
            else:
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
                subprocess.run(["systemctl", "--user", "restart", self.production_gateway_service], capture_output=True, text=True)
