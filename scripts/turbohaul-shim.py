#!/usr/bin/env python3
"""
turbohaul-shim — portable, zero-dependency Turbohaul-compatible API server.

Single file. Pure Python 3.8+ stdlib. No pip installs. No Docker. No WSL2 required.
Copy to any machine, run it, done.

  python3 turbohaul-shim.py              # auto-detect everything
  python3 turbohaul-shim.py --port 11401 # explicit port

Speaks the exact Turbohaul Manager v0.7 HTTP API that TurboFit expects:
  GET  /status                → resident/loading/active/queue/grace state
  GET  /api/tags              → {models: [...]}
  GET  /api/show?name=X       → model manifest
  PUT  /api/manifests/{tag}   → install/update a model manifest
  POST /api/pull-hf           → download GGUF from HuggingFace + SHA-256
  POST /v1/chat/completions   → proxy to the matching backend
  POST /v1/completions        → proxy to the matching backend

Backend auto-detection order:
  1. Native llama-server binary (LLAMA_SERVER_BIN or PATH)
  2. Ollama (OLLAMA_HOST or localhost:11434)
  3. Remote Turbohaul Manager (TURBOHAUL_UPSTREAM)

Platform support: Linux, WSL2, macOS, Windows (via WSL or native Python).
MoE expert offloading: --n-cpu-moe / --n-gpu-moe flags passed through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

__version__ = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shim] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("turbohaul-shim")


# ═══════════════════════════════════════════════════════════════════════════════
# Platform detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_platform() -> dict:
    """Detect OS, WSL, GPU, and available backends."""
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "is_windows": platform.system() == "Windows",
        "is_macos": platform.system() == "Darwin",
        "is_linux": platform.system() == "Linux",
        "gpu": None,
        "cuda_version": None,
        "home": str(Path.home()),
    }
    # GPU detection via nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines = out.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpus.append({"name": parts[0], "vram": parts[1]})
            info["gpu"] = gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # CUDA version
    try:
        out = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.split("\n"):
                if "release" in line:
                    info["cuda_version"] = line.strip()
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def find_llama_server() -> Optional[str]:
    """Find a llama-server binary."""
    # Explicit env var
    env_path = os.environ.get("LLAMA_SERVER_BIN")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    # Common locations
    candidates = [
        "llama-server",  # PATH
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"),
        str(Path.home() / "turbofit" / "llama.cpp" / "llama-server"),
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ]
    if platform.system() == "Darwin":
        candidates.append("/opt/homebrew/bin/llama-server")
    if platform.system() == "Windows":
        candidates.extend([
            str(Path.home() / "turbofit" / "llama.cpp" / "llama-server.exe"),
            r"C:\llama.cpp\llama-server.exe",
        ])
    for c in candidates:
        resolved = shutil.which(c) if not os.path.isabs(c) else c
        if resolved and os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
    return None


def detect_ollama() -> Optional[str]:
    """Detect a reachable Ollama instance."""
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        req = Request(f"{host}/api/tags", headers={"Accept": "application/json"})
        with urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            if isinstance(data.get("models"), list):
                return host
    except Exception:
        pass
    # WSL2: try Windows host via gateway IP
    if "microsoft" in platform.release().lower():
        try:
            out = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                gw = out.stdout.split()[2] if len(out.stdout.split()) > 2 else None
                if gw:
                    alt = f"http://{gw}:11434"
                    req = Request(f"{alt}/api/tags", headers={"Accept": "application/json"})
                    with urlopen(req, timeout=3) as resp:
                        data = json.loads(resp.read())
                        if isinstance(data.get("models"), list):
                            return alt
        except Exception:
            pass
    return None


def detect_upstream_turbohaul() -> Optional[str]:
    """Detect a remote Turbohaul Manager."""
    url = os.environ.get("TURBOHAUL_UPSTREAM")
    if not url:
        return None
    try:
        req = Request(f"{url.rstrip('/')}/status", headers={"Accept": "application/json"})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if isinstance(data.get("residents"), list):
                return url.rstrip("/")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Backend abstraction
# ═══════════════════════════════════════════════════════════════════════════════

class Backend:
    """Abstract backend for model lifecycle."""
    name = "abstract"

    def status(self) -> dict:
        raise NotImplementedError

    def list_models(self) -> list[dict]:
        raise NotImplementedError

    def show_model(self, name: str) -> dict:
        raise NotImplementedError

    def pull_hf(self, repo_id: str, filename: str, revision: str, expected_sha256: str) -> dict:
        raise NotImplementedError

    def chat(self, payload: dict) -> dict:
        raise NotImplementedError

    def unload(self, model: str) -> dict:
        raise NotImplementedError

    def save_manifest(self, tag: str, manifest: dict) -> None:
        raise NotImplementedError

    def load_manifests(self) -> dict[str, dict]:
        raise NotImplementedError


class NativeLlamaBackend(Backend):
    """Manages llama-server processes directly. Full flag control."""
    name = "native-llama-server"

    def __init__(self, llama_bin: str, model_store: Path, manifest_store: Path):
        self.llama_bin = llama_bin
        self.model_store = model_store
        self.manifest_store = manifest_store
        self.model_store.mkdir(parents=True, exist_ok=True)
        self.manifest_store.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, dict] = {}
        self._manifests: dict[str, dict] = {}
        self._model_files: dict[str, str] = {}
        self._ports: dict[str, int] = {}
        self._next_port = int(os.environ.get("TURBOHAUL_BASE_PORT", "8100"))
        self._lock = threading.Lock()
        self._load_manifests()

    def _load_manifests(self):
        for path in self.manifest_store.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                tag = path.stem
                self._manifests[tag] = data
                gguf = data.get("gguf_path") or data.get("filename", "")
                if gguf:
                    self._model_files[tag] = gguf
            except Exception as exc:
                log.warning("Bad manifest %s: %s", path, exc)

    def _alloc_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _build_args(self, tag: str, port: int) -> list[str]:
        manifest = self._manifests.get(tag, {})
        flags = manifest.get("llama_server_flags", {})
        gguf = self._model_files.get(tag)
        if not gguf:
            raise ValueError(f"No GGUF path for tag {tag}")

        args = [self.llama_bin, "--model", gguf, "--port", str(port), "--host", "127.0.0.1"]

        # Flag mapping: acquisitions.json key → llama-server CLI flag
        flag_map = {
            "ctx_size": "--ctx-size",
            "n_gpu_layers": "--n-gpu-layers",
            "main_gpu": "--main-gpu",
            "parallel": "--parallel",
            "cache_reuse": "--cache-reuse",
            "cache_type_k": "--cache-type-k",
            "cache_type_v": "--cache-type-v",
            "flash_attn": "--flash-attn",
            "jinja": "--jinja",
            "no_context_shift": "--no-context-shift",
            "no_perf": "--no-perf",
            "split_mode": "--split-mode",
            "spec_type": "--spec-type",
            "slot_prompt_similarity": "--slot-prompt-similarity",
            # MoE expert offloading
            "n_cpu_moe": "--n-cpu-moe",
            "n_gpu_moe": "--n-gpu-moe",
            # Additional tuning flags
            "threads": "--threads",
            "batch_size": "--batch-size",
            "ubatch_size": "--ubatch-size",
            "mlock": "--mlock",
            "no_mmap": "--no-mmap",
            "numa": "--numa",
            "rope_scaling": "--rope-scaling",
            "rope_freq_base": "--rope-freq-base",
            "rope_freq_scale": "--rope-freq-scale",
            "n_threads_batch": "--n-threads-batch",
            "grp_attn_n": "--grp-attn-n",
            "grp_attn_w": "--grp-attn-w",
        }

        for key, value in flags.items():
            cli_flag = flag_map.get(key)
            if cli_flag is None:
                log.warning("Unknown flag %r for %s — skipping", key, tag)
                continue
            if isinstance(value, bool):
                if value:
                    args.append(cli_flag)
            elif isinstance(value, (int, float)):
                args.extend([cli_flag, str(value)])
            elif isinstance(value, str):
                args.extend([cli_flag, value])

        return args

    def _start_model(self, tag: str) -> dict:
        with self._lock:
            existing = self._processes.get(tag)
            if existing and existing.get("proc") and existing["proc"].poll() is None:
                return existing

            port = self._ports.get(tag) or self._alloc_port()
            self._ports[tag] = port
            args = self._build_args(tag, port)
            log.info("Starting %s on :%d — %s", tag, port, " ".join(args))

            kwargs = {}
            if platform.system() != "Windows":
                kwargs["preexec_fn"] = os.setsid

            proc = subprocess.Popen(
                args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, **kwargs,
            )
            info = {
                "proc": proc, "pid": proc.pid, "port": port,
                "tag": tag, "state": "loading", "started_at": time.time(),
                "log_lines": [],
            }
            self._processes[tag] = info

            def _read_logs():
                try:
                    for line in proc.stdout:
                        info["log_lines"].append(line.rstrip())
                        if len(info["log_lines"]) > 200:
                            info["log_lines"] = info["log_lines"][-100:]
                        if "server is listening" in line or "HTTP server listening" in line:
                            info["state"] = "ready"
                            log.info("%s ready on :%d (pid %d)", tag, port, proc.pid)
                except Exception:
                    pass

            def _health_probe():
                """Fallback: poll /health until the server responds, then mark ready."""
                import urllib.request as _ur
                deadline = time.time() + 300  # 5 min max
                while time.time() < deadline and info.get("state") != "ready":
                    if proc.poll() is not None:
                        return  # process died
                    try:
                        r = _ur.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
                        if r.status == 200:
                            info["state"] = "ready"
                            log.info("%s ready on :%d via health probe (pid %d)", tag, port, proc.pid)
                            return
                    except Exception:
                        pass
                    time.sleep(2)

            threading.Thread(target=_read_logs, daemon=True).start()
            threading.Thread(target=_health_probe, daemon=True).start()
            return info

    def _stop_model(self, tag: str, timeout: float = 10):
        with self._lock:
            info = self._processes.get(tag)
            if not info:
                return
            proc = info.get("proc")
            if proc and proc.poll() is None:
                log.info("Stopping %s (pid %d)", tag, proc.pid)
                try:
                    if platform.system() != "Windows":
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    else:
                        proc.terminate()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        if platform.system() != "Windows":
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        else:
                            proc.kill()
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            info["state"] = "stopped"
            self._processes.pop(tag, None)

    def _model_state(self, tag: str) -> str:
        info = self._processes.get(tag)
        if not info:
            return "down"
        proc = info.get("proc")
        if proc and proc.poll() is not None:
            return "down"
        return info.get("state", "loading")

    def _wait_ready(self, tag: str, timeout: float = 600) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = self._model_state(tag)
            if s == "ready":
                return True
            if s == "down":
                return False
            time.sleep(0.5)
        return False

    # ── Backend interface ──

    def status(self) -> dict:
        residents, active, loading = [], [], []
        for tag in list(self._processes.keys()):
            state = self._model_state(tag)
            info = self._processes.get(tag, {})
            entry = {
                "model_tag": tag, "port": info.get("port"),
                "pid": info.get("pid"), "state": state,
                "split_mode": self._manifests.get(tag, {}).get(
                    "llama_server_flags", {}).get("split_mode", "none"),
            }
            if state == "ready":
                residents.append(entry)
                active.append(entry)
            elif state == "loading":
                loading.append(entry)
                residents.append(entry)
        return {
            "residents": residents, "active": active, "loading": loading,
            "grace": [], "idle_hot": [], "queue": [],
            "manager": f"turbohaul-shim/{__version__}",
            "backend": self.name,
        }

    def list_models(self) -> list[dict]:
        models = []
        for tag, m in self._manifests.items():
            models.append({
                "name": tag, "model": tag, "tag": tag,
                "size": m.get("size_bytes", 0),
                "digest": m.get("sha256", ""),
                "details": {"format": "gguf", "family": m.get("family", "llama")},
            })
        return models

    def show_model(self, name: str) -> dict:
        m = self._manifests.get(name)
        if m:
            return m
        raise KeyError(f"model {name} not found")

    def save_manifest(self, tag: str, manifest: dict):
        path = self.manifest_store / f"{tag}.json"
        path.write_text(json.dumps(manifest, indent=2))
        self._manifests[tag] = manifest
        gguf = manifest.get("gguf_path") or manifest.get("filename", "")
        if gguf:
            self._model_files[tag] = gguf

    def load_manifests(self) -> dict[str, dict]:
        return dict(self._manifests)

    def pull_hf(self, repo_id: str, filename: str, revision: str, expected_sha256: str) -> dict:
        safe = repo_id.replace("/", "--")
        dest_dir = self.model_store / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        if dest.exists():
            sha = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
            if sha.hexdigest() == expected_sha256:
                log.info("%s/%s already verified", repo_id, filename)
                return {"status": "ok", "path": str(dest), "cached": True}
            dest.unlink()

        hf = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = f"{hf}/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
        log.info("Downloading %s", url)
        tmp = dest.with_suffix(".tmp")
        try:
            req = Request(url, headers={"User-Agent": f"turbohaul-shim/{__version__}"})
            with urlopen(req, timeout=3600) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                sha = hashlib.sha256()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        sha.update(chunk)
                        downloaded += len(chunk)
                        if total > 0 and downloaded % (64 * 1024 * 1024) < 8 * 1024 * 1024:
                            log.info("  %s: %d%%", filename, downloaded * 100 // total)
        except (HTTPError, URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            return {"status": "error", "error": str(exc)}

        if sha.hexdigest() != expected_sha256:
            tmp.unlink(missing_ok=True)
            return {"status": "error", "error": f"SHA-256 mismatch: got {sha.hexdigest()}"}

        tmp.rename(dest)
        log.info("Verified %s → %s", filename, dest)
        return {"status": "ok", "path": str(dest), "cached": False}

    def chat(self, payload: dict) -> dict:
        model = payload.get("model", "")
        keep_alive = payload.get("keep_alive")

        if keep_alive == 0:
            self._stop_model(model)
            return {
                "id": "shim-unload", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            }

        state = self._model_state(model)
        if state in ("down", "stopped"):
            if model not in self._manifests:
                raise KeyError(f"model {model} not found")
            self._start_model(model)

        if not self._wait_ready(model):
            raise RuntimeError(f"model {model} failed to become ready")

        proxy_body = {k: v for k, v in payload.items()
                      if k not in ("keep_alive", "chat_template_kwargs", "stream")}
        proxy_body["stream"] = False  # shim returns single JSON, not SSE
        port = self._ports.get(model)
        if not port:
            raise RuntimeError(f"no port for {model}")

        data = json.dumps(proxy_body).encode()
        req = Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        last_err = None
        for attempt in range(3):
            try:
                with urlopen(req, timeout=600) as resp:
                    raw = resp.read()
                    if not raw or not raw.strip():
                        raise RuntimeError("llama-server returned empty body")
                    return json.loads(raw)
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_err = exc
                log.warning("chat attempt %d/3 failed: %s", attempt + 1, exc)
                if attempt < 2:
                    import time as _t
                    _t.sleep(2)
            except (HTTPError, URLError, OSError) as exc:
                last_err = exc
                log.warning("chat attempt %d/3 network error: %s", attempt + 1, exc)
                if attempt < 2:
                    import time as _t
                    _t.sleep(2)
        raise RuntimeError(f"llama-server failed after 3 attempts: {last_err}")

    def unload(self, model: str) -> dict:
        self._stop_model(model)
        return self.status()

    def shutdown(self):
        for tag in list(self._processes.keys()):
            self._stop_model(tag, timeout=5)


class OllamaBackend(Backend):
    """Proxies through an existing Ollama instance. Simpler, fewer flags."""
    name = "ollama"

    def __init__(self, ollama_url: str):
        self.url = ollama_url.rstrip("/")
        self._manifests: dict[str, dict] = {}
        self._manifest_store = Path(os.environ.get(
            "TURBOHAUL_MANIFEST_STORE",
            str(Path.home() / ".turbohaul" / "manifests"),
        ))
        self._manifest_store.mkdir(parents=True, exist_ok=True)
        self._load_manifests()

    def _load_manifests(self):
        for path in self._manifest_store.glob("*.json"):
            try:
                self._manifests[path.stem] = json.loads(path.read_text())
            except Exception:
                pass

    def _api(self, method: str, path: str, payload: dict = None, timeout: float = 600) -> dict:
        url = f"{self.url}{path}"
        data = json.dumps(payload).encode() if payload else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def status(self) -> dict:
        try:
            ps = self._api("GET", "/api/ps")
            models = ps.get("models", [])
        except Exception:
            models = []
        residents = [{"model_tag": m.get("name", ""), "state": "ready"} for m in models]
        return {
            "residents": residents, "active": residents, "loading": [],
            "grace": [], "idle_hot": [], "queue": [],
            "manager": f"turbohaul-shim/{__version__}", "backend": self.name,
        }

    def list_models(self) -> list[dict]:
        try:
            data = self._api("GET", "/api/tags")
            return data.get("models", [])
        except Exception:
            return []

    def show_model(self, name: str) -> dict:
        return self._api("POST", "/api/show", {"name": name})

    def save_manifest(self, tag: str, manifest: dict):
        path = self._manifest_store / f"{tag}.json"
        path.write_text(json.dumps(manifest, indent=2))
        self._manifests[tag] = manifest

    def load_manifests(self) -> dict[str, dict]:
        return dict(self._manifests)

    def pull_hf(self, repo_id: str, filename: str, revision: str, expected_sha256: str) -> dict:
        # Ollama can't pull from HF directly — fall back to native download
        safe = repo_id.replace("/", "--")
        store = Path(os.environ.get(
            "TURBOHAUL_MODEL_STORE", str(Path.home() / ".turbohaul" / "models")))
        dest_dir = store / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        if dest.exists():
            sha = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
            if sha.hexdigest() == expected_sha256:
                return {"status": "ok", "path": str(dest), "cached": True}
            dest.unlink()

        hf = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        url = f"{hf}/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
        log.info("Downloading %s", url)
        tmp = dest.with_suffix(".tmp")
        try:
            req = Request(url, headers={"User-Agent": f"turbohaul-shim/{__version__}"})
            with urlopen(req, timeout=3600) as resp:
                sha = hashlib.sha256()
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        sha.update(chunk)
        except (HTTPError, URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            return {"status": "error", "error": str(exc)}

        if sha.hexdigest() != expected_sha256:
            tmp.unlink(missing_ok=True)
            return {"status": "error", "error": "SHA-256 mismatch"}
        tmp.rename(dest)
        return {"status": "ok", "path": str(dest), "cached": False}

    def chat(self, payload: dict) -> dict:
        return self._api("POST", "/v1/chat/completions", payload)

    def unload(self, model: str) -> dict:
        # Ollama: send keep_alive=0 to unload
        try:
            self._api("POST", "/v1/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 1, "keep_alive": 0,
            })
        except Exception:
            pass
        return self.status()

    def shutdown(self):
        pass


class UpstreamBackend(Backend):
    """Pure proxy to a remote Turbohaul Manager."""
    name = "upstream-turbohaul"

    def __init__(self, upstream_url: str):
        self.url = upstream_url.rstrip("/")

    def _api(self, method: str, path: str, payload: dict = None, timeout: float = 600) -> dict:
        url = f"{self.url}{path}"
        data = json.dumps(payload).encode() if payload else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = Request(url, data=data, headers=headers, method=method)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def status(self) -> dict:
        return self._api("GET", "/status")

    def list_models(self) -> list[dict]:
        return self._api("GET", "/api/tags").get("models", [])

    def show_model(self, name: str) -> dict:
        return self._api("GET", f"/api/show?name={urllib.parse.quote(name)}")

    def save_manifest(self, tag: str, manifest: dict):
        self._api("PUT", f"/api/manifests/{urllib.parse.quote(tag)}", manifest)

    def load_manifests(self) -> dict[str, dict]:
        return {}

    def pull_hf(self, repo_id: str, filename: str, revision: str, expected_sha256: str) -> dict:
        return self._api("POST", "/api/pull-hf", {
            "repo_id": repo_id, "filename": filename,
            "revision": revision, "expected_sha256": expected_sha256,
        }, timeout=3600)

    def chat(self, payload: dict) -> dict:
        return self._api("POST", "/v1/chat/completions", payload)

    def unload(self, model: str) -> dict:
        return self.chat({
            "model": model,
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 1, "keep_alive": 0,
        })

    def shutdown(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Backend selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_backend(force: str = "auto") -> Backend:
    """Auto-detect the best available backend."""
    if force == "ollama":
        url = detect_ollama()
        if url:
            log.info("Backend: Ollama at %s", url)
            return OllamaBackend(url)
        raise RuntimeError("Ollama not reachable")

    if force == "upstream":
        url = detect_upstream_turbohaul()
        if url:
            log.info("Backend: upstream Turbohaul at %s", url)
            return UpstreamBackend(url)
        raise RuntimeError("Upstream Turbohaul not reachable")

    if force == "native":
        llama = find_llama_server()
        if llama:
            log.info("Backend: native llama-server at %s", llama)
            store = Path(os.environ.get("TURBOHAUL_MODEL_STORE",
                                        str(Path.home() / ".turbohaul" / "models")))
            manifests = Path(os.environ.get("TURBOHAUL_MANIFEST_STORE",
                                            str(Path.home() / ".turbohaul" / "manifests")))
            return NativeLlamaBackend(llama, store, manifests)
        raise RuntimeError("llama-server binary not found")

    # Auto-detect: prefer native > ollama > upstream
    llama = find_llama_server()
    if llama:
        log.info("Auto-detected: native llama-server at %s", llama)
        store = Path(os.environ.get("TURBOHAUL_MODEL_STORE",
                                    str(Path.home() / ".turbohaul" / "models")))
        manifests = Path(os.environ.get("TURBOHAUL_MANIFEST_STORE",
                                        str(Path.home() / ".turbohaul" / "manifests")))
        return NativeLlamaBackend(llama, store, manifests)

    ollama_url = detect_ollama()
    if ollama_url:
        log.info("Auto-detected: Ollama at %s", ollama_url)
        return OllamaBackend(ollama_url)

    upstream = detect_upstream_turbohaul()
    if upstream:
        log.info("Auto-detected: upstream Turbohaul at %s", upstream)
        return UpstreamBackend(upstream)

    raise RuntimeError(
        "No backend found. Install llama-server, start Ollama, "
        "or set TURBOHAUL_UPSTREAM=http://host:11401"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Server
# ═══════════════════════════════════════════════════════════════════════════════

_backend: Backend = None  # set in main()


class ShimHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, result: dict):
        """Convert a complete chat response into an SSE stream for OpenAI SDK clients."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        chunk_id = result.get("id", "shim-stream")
        model = result.get("model", "unknown")
        created = result.get("created", 0)
        choice = (result.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        finish = choice.get("finish_reason", "stop")

        def emit(delta: dict, fr=None):
            chunk = {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fr}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Role chunk
        emit({"role": "assistant"})

        # Reasoning content (Bonsai thinking tokens)
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            emit({"reasoning_content": reasoning})

        # Content
        content = msg.get("content", "")
        if content:
            emit({"content": content})

        # Tool calls
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                emit({"tool_calls": [{
                    "index": i, "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }]})

        # Final chunk with finish_reason
        emit({}, fr=finish)

        # Usage chunk (if present)
        usage = result.get("usage")
        if usage:
            usage_chunk = {
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [], "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/status":
            self._json(_backend.status())
        elif path == "/api/tags":
            self._json({"models": _backend.list_models()})
        elif path == "/api/show":
            qs = urllib.parse.parse_qs(parsed.query)
            name = qs.get("name", [""])[0]
            try:
                self._json(_backend.show_model(name))
            except KeyError:
                self._json({"error": f"model {name} not found"}, 404)
        elif path == "/health":
            self._json({"status": "ok", "backend": _backend.name})
        elif path == "/v1/models":
            models = [{"id": m.get("name", m.get("tag", "")), "object": "model",
                       "owned_by": "turbohaul-shim"} for m in _backend.list_models()]
            self._json({"object": "list", "data": models})
        else:
            self._json({"error": "not found"}, 404)

    def do_PUT(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path.startswith("/api/manifests/"):
            tag = urllib.parse.unquote(path[len("/api/manifests/"):])
            body = self._read_body()
            _backend.save_manifest(tag, body)
            log.info("Manifest saved: %s", tag)
            self._json({"status": "ok", "tag": tag})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/pull-hf":
            body = self._read_body()
            result = _backend.pull_hf(
                body["repo_id"], body["filename"],
                body["revision"], body["expected_sha256"],
            )
            self._json(result, 200 if result.get("status") == "ok" else 500)

        elif path in ("/v1/chat/completions", "/v1/completions"):
            body = self._read_body()
            want_stream = body.get("stream", False)
            try:
                result = _backend.chat(body)
                if want_stream:
                    self._sse(result)
                else:
                    self._json(result)
            except KeyError as exc:
                self._json({"error": str(exc)}, 404)
            except Exception as exc:
                self._json({"error": str(exc)}, 502)
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):
        log.info("%s %s", self.command, self.path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global _backend

    parser = argparse.ArgumentParser(
        description="turbohaul-shim — portable Turbohaul-compatible API server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TURBOHAUL_SHIM_PORT", "11401")))
    parser.add_argument("--bind", default=os.environ.get("TURBOHAUL_SHIM_BIND", "0.0.0.0"))
    parser.add_argument("--backend", choices=["auto", "native", "ollama", "upstream"],
                        default=os.environ.get("TURBOHAUL_BACKEND", "auto"))
    args = parser.parse_args()

    plat = detect_platform()
    log.info("Platform: %s %s (%s)", plat["system"], plat["release"], plat["machine"])
    if plat["gpu"]:
        for g in plat["gpu"]:
            log.info("GPU: %s (%s)", g["name"], g["vram"])
    if plat["is_wsl"]:
        log.info("Running inside WSL2")

    _backend = select_backend(args.backend)
    log.info("Backend: %s", _backend.name)

    def _shutdown(signum, frame):
        log.info("Shutting down...")
        _backend.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server = ThreadingHTTPServer((args.bind, args.port), ShimHandler)
    log.info("turbohaul-shim/%s on %s:%d", __version__, args.bind, args.port)
    log.info("  Turbohaul API: /status /api/tags /api/show /api/manifests /api/pull-hf")
    log.info("  OpenAI proxy:  /v1/chat/completions /v1/completions")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
