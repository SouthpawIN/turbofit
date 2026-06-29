#!/usr/bin/env python3
"""
turbofit-status-server — tiny HTTP server for nginx /status endpoint.

Returns JSON with current active main/aux model, VRAM, and daemon state.
Also serves as the health check for the turbofit gateway.

Runs on port 8090.
"""

import json
import subprocess
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
PREFS = os.environ.get("TURBOFIT_PREFS", f"{HOME}/.config/turbofit/preferences.yaml")
HERMES_HOME = os.environ.get("HERMES_HOME", f"{HOME}/.hermes")


def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def check_port(port):
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "2", f"http://127.0.0.1:{port}/v1/models"],
            capture_output=True, text=True, timeout=5
        )
        return "data" in r.stdout
    except:
        return False


def get_gpu_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in r.stdout.strip().split("\n"):
            if line.strip():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "id": int(parts[0]),
                        "free_gb": round(int(parts[1]) / 1024, 1),
                        "total_gb": round(int(parts[2]) / 1024, 1),
                        "used_gb": round(int(parts[3]) / 1024, 1),
                    })
        return gpus
    except:
        return []


def get_active_models():
    catalog = load_yaml(CATALOG)
    prefs = load_yaml(PREFS)
    models = catalog.get("models", {})

    local = prefs.get("api_fallback", {}).get("local", {})
    preferred = local.get("main", "darwin-28b-reason")
    ladder = [preferred, "darwin-28b-coder", "prism-eagle-27b", "darwin-apex-36b", "carnice-v2-27b"]

    main_info = {"alias": "none", "type": "down", "base_url": "", "port": 0}
    for alias in ladder:
        if alias not in models:
            continue
        port = models[alias].get("port", 0)
        if port and check_port(port):
            main_info = {
                "alias": alias,
                "type": "local",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "port": port,
                "size_gb": models[alias].get("size_gb", 0),
                "ctx": models[alias].get("ctx", 262144),
            }
            break

    if main_info["type"] == "down":
        senter_cfg = load_yaml(f"{HERMES_HOME}/profiles/senter/config.yaml")
        senter_model = senter_cfg.get("model", {})
        url = senter_model.get("base_url", "")
        if "inference-api" in url or ("127.0.0.1" not in url and url):
            main_info = {
                "alias": "api-fallback",
                "type": "api",
                "base_url": url,
                "port": 0,
                "model_id": senter_model.get("default", ""),
            }

    aux_info = {"alias": "none", "type": "down", "base_url": "", "port": 0}
    for alias, model in models.items():
        if model.get("role") != "aux":
            continue
        port = model.get("port", 0)
        if port and check_port(port):
            aux_info = {
                "alias": alias,
                "type": "local",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "port": port,
                "size_gb": model.get("size_gb", 0),
            }
            break

    return main_info, aux_info


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/status":
            main, aux = get_active_models()
            gpus = get_gpu_vram()
            response = {
                "main": main,
                "aux": aux,
                "gpus": gpus,
                "gateway": "turbofit",
                "endpoints": {
                    "main": "/main/v1/chat/completions",
                    "aux": "/aux/v1/chat/completions",
                    "hermes": "/hermes/",
                    "dashboard": "/dashboard/",
                    "desktop": "/desktop/",
                    "status": "/status",
                },
            }
            data = json.dumps(response, indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8090), Handler)
    print("turbofit-status-server on :8090")
    server.serve_forever()
