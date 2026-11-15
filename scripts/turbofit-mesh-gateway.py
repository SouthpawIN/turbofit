#!/usr/bin/env python3
"""
Turbofit Mesh Gateway — discovers models across Tailscale peers.
"""
import os
import sys
import json
import time
import subprocess
import threading
import logging
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# Import everything from the original gateway
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Import specific functions from the original gateway module
import importlib.util
spec = importlib.util.spec_from_file_location("turbofit_gateway_module", str(SCRIPT_DIR / "turbofit-gateway.py"))
turbofit_gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turbofit_gateway)

# Re-export everything we need
GatewayHandler = turbofit_gateway.GatewayHandler
resolve_main = turbofit_gateway.resolve_main
resolve_aux = turbofit_gateway.resolve_aux
active_context_length = turbofit_gateway.active_context_length
load_yaml = turbofit_gateway.load_yaml
CATALOG = turbofit_gateway.CATALOG
SELF_PORT = turbofit_gateway.SELF_PORT
STALL_TIMEOUT_S = turbofit_gateway.STALL_TIMEOUT_S
AUX_MAX_TOKENS = turbofit_gateway.AUX_MAX_TOKENS
AUX_ENABLE_THINKING = turbofit_gateway.AUX_ENABLE_THINKING
MAIN_ENABLE_THINKING = turbofit_gateway.MAIN_ENABLE_THINKING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [mesh] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mesh")

TAILSCALE_BIN = os.environ.get("TAILSCALE_BIN", "tailscale")
MESH_PORT = int(os.environ.get("TURBOFIT_MESH_PORT", "8080"))
MESH_CACHE_TTL = 60
_mesh_cache = {"peers": [], "ts": 0}


def tailscale_status():
    """Get Tailscale peer list."""
    try:
        result = subprocess.run(
            [TAILSCALE_BIN, "status", "--json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return {}


def probe_endpoint(host, port):
    """Check if host:port has a working endpoint."""
    url = f"http://{host}:{port}"
    try:
        req = Request(f"{url}/health", method="GET")
        with urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            if resp.status == 200 and data.get("status") in ("ok", "ready", "healthy"):
                return True, url
    except Exception:
        pass
    try:
        req = Request(f"{url}/v1/models", method="GET")
        with urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode())
            if resp.status == 200 and "data" in data:
                return True, url
    except Exception:
        pass
    return False, None


def get_models_at_endpoint(url):
    """Get models from an endpoint."""
    try:
        req = Request(f"{url}/v1/models", method="GET")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("data", [])
    except Exception:
        return []


def discover_mesh():
    """Discover Turbofit instances on Tailscale."""
    global _mesh_cache
    now = time.time()
    if now - _mesh_cache["ts"] < MESH_CACHE_TTL:
        return _mesh_cache["peers"]

    status = tailscale_status()
    peers = status.get("Peer", {})
    if not peers:
        self_peer = status.get("Self", {})
        if self_peer:
            peers = {"self": self_peer}

    discovered = []
    for name, peer in peers.items():
        if peer.get("Self", False):
            continue
        if not peer.get("Online", False):
            continue

        ips = peer.get("TailscaleIPs", [])
        ip = ips[0] if ips else None
        if not ip:
            continue

        is_alive, base_url = probe_endpoint(ip, MESH_PORT)
        if not is_alive:
            continue

        models = get_models_at_endpoint(base_url)
        hostname = peer.get("HostName", name)

        discovered.append({
            "name": name,
            "hostname": hostname,
            "ip": ip,
            "base_url": base_url,
            "models": models,
        })

    _mesh_cache = {"peers": discovered, "ts": now}
    return discovered


def provider_models_unified():
    """Unified model list: auto + local + mesh (Tailscale)."""
    context_length = active_context_length()
    models = [
        {
            "id": "auto",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Auto-configure best local + mesh model",
            "context_length": context_length,
        },
    ]

    # Add local catalog entries
    try:
        catalog = load_yaml(CATALOG)
        catalog_models = catalog.get("models", {}) or {}
        for alias, m in catalog_models.items():
            role = (m.get("role") or "either").lower()
            if alias not in ("auto", "active:main", "active:aux"):
                models.append({
                    "id": alias,
                    "object": "model",
                    "owned_by": "turbofit",
                    "description": f"Local: {alias} (role={role})",
                    "context_length": m.get("ctx", context_length),
                    "role": role,
                })
    except Exception:
        pass

    # Add mesh (Tailscale) models
    try:
        mesh = discover_mesh()
        for peer in mesh:
            for m in peer.get("models", []):
                model_id = m.get("id", "unknown")
                if model_id in ("auto", "active:main", "active:aux"):
                    continue
                # Get the actual context length from the mesh model data
                mesh_ctx = m.get("context_length", 0)
                if not mesh_ctx or mesh_ctx < 64000:
                    mesh_ctx = 262144  # Default to 262K for Tailscale models
                models.append({
                    "id": f"{model_id} (Tailscale: {peer['hostname']})",
                    "object": "model",
                    "owned_by": "turbofit",
                    "description": f"Tailscale: {model_id} @ {peer['hostname']}",
                    "context_length": mesh_ctx,
                    "role": "either",
                    "mesh_hostname": peer["hostname"],
                    "mesh_ip": peer["ip"],
                    "mesh_base_url": peer["base_url"],
                    "mesh_model_id": model_id,
                })
    except Exception:
        pass

    return models


class MeshGatewayHandler(GatewayHandler):
    """Extended handler that supports Tailscale model routing."""

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": provider_models_unified()})
        elif self.path.startswith("/v1/models/") and "/" not in self.path[len("/v1/models/"):]:
            model_id = self.path[len("/v1/models/"):]
            model = next((m for m in provider_models_unified() if m["id"] == model_id), None)
            if model is None:
                self._send_json(404, {"error": {"message": f"Unknown model: {model_id}"}})
            else:
                self._send_json(200, model)
        else:
            super().do_GET()

    def do_POST(self):
        path = self.path
        if path.startswith("/v1/") and path not in ("/v1/models", "/v1/props"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            model = "auto"
            if body:
                try:
                    model = json.loads(body).get("model") or "auto"
                except Exception:
                    self.send_error(400, "Invalid JSON")
                    return

            # Check if this is a mesh model request
            mesh_model = None
            try:
                mesh = discover_mesh()
                for peer in mesh:
                    for m in peer.get("models", []):
                        mesh_id = m.get("id", "unknown")
                        expected_name = f"{mesh_id} (Tailscale: {peer['hostname']})"
                        if model == expected_name or model == mesh_id:
                            mesh_model = {
                                "hostname": peer["hostname"],
                                "ip": peer["ip"],
                                "base_url": peer["base_url"],
                                "model_id": mesh_id,
                            }
                            break
                    if mesh_model:
                        break
            except Exception:
                pass

            if mesh_model:
                # Proxy directly to mesh endpoint
                base_url = mesh_model["base_url"]
                model_id = mesh_model["model_id"]
                
                # Rewrite body with correct model ID
                if body:
                    try:
                        payload = json.loads(body)
                        payload["model"] = model_id
                        body = json.dumps(payload).encode()
                    except Exception:
                        pass

                # Proxy the request
                upstream_path = path[len("/v1/"):]
                try:
                    req = Request(
                        f"{base_url}/v1/{upstream_path}",
                        data=body,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urlopen(req, timeout=300) as resp:
                        resp_body = resp.read()
                        self.send_response(resp.status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(resp_body)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(resp_body)
                        return
                except Exception as e:
                    self._send_503(f"Tailscale proxy failed: {e}")
                    return
            else:
                # Fall through to local handling
                super().do_POST()
        else:
            super().do_POST()


if __name__ == "__main__":
    port = int(os.environ.get("TURBOFIT_GATEWAY_PORT", "8091"))
    host = os.environ.get("TURBOFIT_GATEWAY_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), MeshGatewayHandler)
    log.info(f"Turbofit Mesh Gateway on :{port}")
    log.info("  /v1/models → unified local + Tailscale model list")
    log.info("  /v1/chat/completions → route to local or mesh backend")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown")
        server.shutdown()
