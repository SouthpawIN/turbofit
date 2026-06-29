#!/usr/bin/env python3
"""
turbofit-gateway — dynamic reverse proxy for nginx.

Sits behind nginx on port 8091 and dynamically routes /main/ requests
to whatever model the scaling watcher has decided should be running.

When the scaling watcher contracts (Darwin -> Prism Eagle -> API fallback),
this proxy automatically follows. No nginx reload needed.

Also handles /aux/ routing the same way.

Runs on :8091
"""

import json
import subprocess
import os
import sys
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gate] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gate")

HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
PREFS = os.environ.get("TURBOFIT_PREFS", f"{HOME}/.config/turbofit/preferences.yaml")
HERMES_HOME = os.environ.get("HERMES_HOME", f"{HOME}/.hermes")

_cache = {"main": None, "aux": None, "ts": 0}
CACHE_TTL = 10


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


def resolve_main():
    now = time.time()
    if _cache["main"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["main"]

    catalog = load_yaml(CATALOG)
    prefs = load_yaml(PREFS)
    models = catalog.get("models", {})

    local = prefs.get("api_fallback", {}).get("local", {})
    preferred = local.get("main", "darwin-28b-reason")
    ladder = [preferred, "darwin-28b-coder", "prism-eagle-27b", "darwin-apex-36b", "carnice-v2-27b"]

    result = None
    for alias in ladder:
        if alias not in models:
            continue
        port = models[alias].get("port", 0)
        if port and check_port(port):
            result = {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
            }
            break

    if not result:
        senter_cfg = load_yaml(f"{HERMES_HOME}/profiles/senter/config.yaml")
        senter_model = senter_cfg.get("model", {})
        url = senter_model.get("base_url", "")
        if "inference-api" in url or ("127.0.0.1" not in url and url):
            result = {
                "alias": "api-fallback",
                "base_url": url.rstrip("/v1").rstrip("/"),
                "port": 0,
                "is_api": True,
            }

    if result:
        _cache["main"] = result
        _cache["ts"] = now
    return result


def resolve_aux():
    catalog = load_yaml(CATALOG)
    models = catalog.get("models", {})

    for alias, model in models.items():
        if model.get("role") != "aux":
            continue
        port = model.get("port", 0)
        if port and check_port(port):
            return {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
            }
    return None


class GatewayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    def _proxy(self):
        path = self.path

        if path.startswith("/main/"):
            backend = resolve_main()
            if not backend:
                self.send_error(503, "No main model available")
                return
            upstream_path = path[len("/main/"):]
            if not upstream_path:
                upstream_path = "/"
            target = f"{backend['base_url']}/{upstream_path}"

        elif path.startswith("/aux/"):
            backend = resolve_aux()
            if not backend:
                self.send_error(503, "No aux model available")
                return
            upstream_path = path[len("/aux/"):]
            if not upstream_path:
                upstream_path = "/"
            target = f"{backend['base_url']}/{upstream_path}"

        elif path == "/status":
            self._send_status()
            return

        else:
            self.send_error(404, f"Unknown path: {path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "transfer-encoding")}
        req = Request(target, data=body, headers=headers, method=self.command)

        try:
            with urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                # Only forward safe headers, skip duplicates
                sent_headers = set()
                for k, v in resp.headers.items():
                    kl = k.lower()
                    if kl in ("transfer-encoding", "connection", "content-length", "content-encoding"):
                        continue
                    if kl in sent_headers:
                        continue
                    sent_headers.add(kl)
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except URLError as e:
            log.error(f"Proxy error for {target}: {e}")
            self.send_error(502, f"Backend error: {e}")
        except Exception as e:
            log.error(f"Unexpected error for {target}: {e}")
            self.send_error(500, str(e))

    def _send_status(self):
        main = resolve_main()
        aux = resolve_aux()
        response = {
            "main": main or {"alias": "none", "type": "down"},
            "aux": aux or {"alias": "none", "type": "down"},
            "gateway": "turbofit-gateway",
        }
        data = json.dumps(response, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        log.info(f"{self.command} {self.path}")


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8091), GatewayHandler)
    log.info("turbofit-gateway on :8091 - dynamic model routing")
    log.info("  /main/ -> follows scaling watcher (Darwin -> Prism Eagle -> API)")
    log.info("  /aux/  -> follows scaling watcher (Carnice -> none)")
    log.info("  /status -> JSON status")
    server.serve_forever()
