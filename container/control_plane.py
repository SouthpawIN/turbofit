#!/usr/bin/env python3
"""Small dependency-free Turbofit control plane.

The control plane is intentionally boring: one stable OpenAI-compatible endpoint
in front of whichever model Turbofit has selected. Model containers can be added
without changing Hermes configuration.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from hardware_probe import probe


HOST = os.getenv("TURBOFIT_HOST", "0.0.0.0")
PORT = int(os.getenv("TURBOFIT_PORT", "8091"))
MANIFEST = Path(os.getenv("TURBOFIT_MANIFEST", "/etc/turbofit/manifest.json"))
STATE = Path(os.getenv("TURBOFIT_STATE", "/var/lib/turbofit/state.json"))


def load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def gpu_state() -> list[dict[str, Any]]:
    return probe()["gpus"]


def manifest() -> dict[str, Any]:
    return load_json(MANIFEST, {"models": [], "profiles": [], "fallback": []})


def state() -> dict[str, Any]:
    default = {"main": {"alias": "bonsai-27b-1bit", "base_url": "http://127.0.0.1:11610"}, "aux": None}
    return load_json(STATE, default)


def backend_url(role: str) -> str | None:
    entry = state().get(role)
    if not isinstance(entry, dict):
        return None
    return str(entry.get("base_url", "")).rstrip("/") or None


def recommendation() -> dict[str, Any]:
    fleet = manifest()
    hardware = probe()
    gpus = hardware["gpus"]
    free_gb = hardware["free_vram_gb"]
    if not gpus:
        ram = hardware["system_ram_gb"]
        preferred = "ternary-bonsai-27b-dspark" if ram >= 16 else "bonsai-27b-1bit" if ram >= 6 else None
        model = next((m for m in fleet.get("models", []) if m.get("alias") == preferred), None)
        return {"hardware": hardware, "mode": "local-cpu" if model else "api", "profile": model.get("profile") if model else None, "recommended_main": model, "fallback": fleet.get("fallback", [])}
    profiles = sorted(fleet.get("profiles", []), key=lambda x: float(x.get("minimum_free_vram_gb", x.get("minimum_memory_gb", 0))))
    selected = None
    for profile in profiles:
        if free_gb >= float(profile.get("minimum_free_vram_gb", profile.get("minimum_memory_gb", 0))):
            selected = profile
    if selected is None and profiles:
        selected = profiles[0]
    preferred = selected.get("preferred") if selected else None
    model = next((m for m in fleet.get("models", []) if m.get("alias") == preferred), None)
    return {"hardware": hardware, "mode": "local", "profile": selected, "recommended_main": model, "fallback": fleet.get("fallback", [])}


class Handler(BaseHTTPRequestHandler):
    server_version = "TurbofitControlPlane/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"turbofit: {format % args}", flush=True)

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "service": "turbofit"})
        elif self.path == "/status":
            self.send_json(200, {"status": "ok", "gpus": gpu_state(), "state": state(), "recommendation": recommendation(), "manifest": manifest()})
        elif self.path == "/recommendation":
            self.send_json(200, recommendation())
        elif self.path == "/fallback":
            self.send_json(200, {"fallback": manifest().get("fallback", []), "active": state()})
        elif self.path == "/v1/models":
            main = state().get("main") or {}
            self.send_json(200, {"object": "list", "data": [{"id": main.get("model", main.get("alias", "turbofit")), "object": "model", "owned_by": "turbofit"}]})
        else:
            self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/"):
            self.send_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        target = backend_url("aux" if self.path.startswith("/v1/aux/") else "main")
        if not target:
            self.send_json(503, {"error": {"message": "no active backend", "type": "backend_unavailable"}})
            return
        suffix = self.path.removeprefix("/v1/aux") if self.path.startswith("/v1/aux/") else self.path.removeprefix("/v1")
        url = target + "/v1" + suffix
        request = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.send_json(503, {"error": {"message": f"backend unavailable: {exc}", "type": "backend_unavailable"}})


def main() -> None:
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()