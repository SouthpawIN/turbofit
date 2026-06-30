#!/usr/bin/env python3
"""
turbofit-gateway — dynamic reverse proxy for nginx with graceful degradation.

Sits behind nginx on port 8091 and dynamically routes /main/ requests
to whatever model the scaling watcher has decided should be running.

Graceful degradation (the whole point of turbofit):
  1. If the preferred local model is LOADING (port bound but model not yet
     serving), STALL the request with backoff up to STALL_TIMEOUT_S — the
     user's first request after a daemon restart just waits, instead of
     failing.
  2. If the local model is genuinely DEAD (port not bound, or daemon
     crashed), fall through to the next model in the local ladder.
  3. If the entire local ladder is dead, fall back to the API chain
     configured in preferences.yaml (api_fallback).
  4. If even the API is down, return 503 with a clear reason — never
     silently proxy to a dead backend.

When the scaling watcher contracts (Darwin -> Prism Eagle -> API fallback),
this proxy automatically follows. No nginx reload needed.

Also handles /aux/ routing the same way.

Runs on :8091
"""

import json
import socket
import subprocess
import os
import sys
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

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
SELF_PORT = int(os.environ.get("TURBOFIT_GATEWAY_PORT", "8091"))  # never pick a model on our own port

_cache = {"main": None, "aux": None, "ts": 0}
CACHE_TTL = 10

# Graceful-degradation tunables (overridable via env)
STALL_TIMEOUT_S = float(os.environ.get("TURBOFIT_STALL_TIMEOUT", "90"))  # max wait while local loads
STALL_POLL_S = float(os.environ.get("TURBOFIT_STALL_POLL", "2"))  # poll interval while waiting
PROXY_BACKEND_TIMEOUT_S = float(os.environ.get("TURBOFIT_BACKEND_TIMEOUT", "300"))  # per-request upstream timeout
PORT_PROBE_TIMEOUT_S = float(os.environ.get("TURBOFIT_PORT_PROBE", "1.5"))  # TCP connect check


def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ─── Health probes ────────────────────────────────────────────────────────────

def port_is_open(port, timeout=PORT_PROBE_TIMEOUT_S):
    """Cheap TCP connect check — port is bound, even if the model isn't loaded yet."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def check_port(port):
    """Full HTTP health check — model is actually serving completions."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/health"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return True
        # Some llama-server builds return JSON in /health; fall back to /v1/models
        r2 = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/v1/models"],
            capture_output=True, text=True, timeout=5
        )
        return "data" in r2.stdout
    except Exception:
        return False


def backend_state(port):
    """Returns one of: 'ready', 'loading', 'down'.

    The port-SELF_PORT guard prevents a model registered on the gateway's own
    port from being picked (which would create a recursive proxy loop).
    """
    if not port or port == SELF_PORT:
        return "down"
    if not port_is_open(port):
        return "down"
    if check_port(port):
        return "ready"
    return "loading"


# ─── Backend resolvers ────────────────────────────────────────────────────────

def _find_api_fallback_in_profiles():
    """Search every profile's config + the global config for an API endpoint.

    Order:
      1. ~/.hermes/config.yaml (default profile)
      2. ~/.hermes/profiles/senter/config.yaml (the orchestrator, often a useful default)
      3. Any other profile that has a non-localhost base_url
    """
    candidates = [
        f"{HERMES_HOME}/config.yaml",
        f"{HERMES_HOME}/profiles/senter/config.yaml",
    ]
    # Append other profiles sorted by name for determinism
    profiles_dir = os.path.join(HERMES_HOME, "profiles")
    if os.path.isdir(profiles_dir):
        for name in sorted(os.listdir(profiles_dir)):
            cfg = os.path.join(profiles_dir, name, "config.yaml")
            if cfg not in candidates and os.path.isfile(cfg):
                candidates.append(cfg)

    for cfg in candidates:
        try:
            data = load_yaml(cfg)
            model = data.get("model", {}) or {}
            url = (model.get("base_url") or "").strip()
            default = (model.get("default") or "").strip()
            provider = (model.get("provider") or "").strip()
            if not url or not default:
                continue
            # Skip localhost — we want the API fallback
            if "127.0.0.1" in url or "localhost" in url:
                continue
            return {
                "alias": "api-fallback",
                "base_url": url.rstrip("/v1").rstrip("/"),
                "port": 0,
                "is_api": True,
                "model_id": default,
                "provider": provider,
                "source": os.path.relpath(cfg, HOME),
            }
        except Exception:
            continue
    return None


def _local_ladder():
    """Build the ordered local-main ladder from preferences + catalog."""
    prefs = load_yaml(PREFS)
    catalog = load_yaml(CATALOG)
    models = catalog.get("models", {}) or {}

    local_cfg = prefs.get("api_fallback", {}).get("local", {}) or {}
    preferred = local_cfg.get("main", "darwin-28b-reason")

    # Prefer the configured main first, then fall through by catalog order
    ladder = [preferred]
    for alias in models:
        if alias != preferred:
            # Only include role=main (or role=either) catalog entries
            role = (models[alias].get("role") or "either").lower()
            if role in ("main", "either"):
                ladder.append(alias)

    # De-dupe while preserving order
    seen, ordered = set(), []
    for a in ladder:
        if a not in seen and a in models:
            seen.add(a)
            ordered.append(a)
    return ordered, models


def resolve_main():
    """Resolve the best available MAIN backend.

    Returns: dict with alias, base_url, port, [is_api, model_id, provider]
             OR None if nothing is reachable.
    """
    now = time.time()
    if _cache["main"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["main"]

    ladder, models = _local_ladder()

    # 1. First local that is READY wins
    for alias in ladder:
        m = models.get(alias, {}) or {}
        port = m.get("port", 0)
        state = backend_state(port)
        if state == "ready":
            result = {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
                "state": "ready",
            }
            _cache["main"] = result
            _cache["ts"] = now
            return result

    # 2. Any local LOADING? Caller will stall on this — surface it
    for alias in ladder:
        m = models.get(alias, {}) or {}
        port = m.get("port", 0)
        if backend_state(port) == "loading":
            result = {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
                "state": "loading",
            }
            _cache["main"] = result
            _cache["ts"] = now
            return result

    # 3. Nothing local is reachable — fall through to API
    api = _find_api_fallback_in_profiles()
    if api:
        result = {**api, "state": "ready"}
        _cache["main"] = result
        _cache["ts"] = now
        return result

    # 4. Nothing anywhere — caller should 503 with a clear reason
    _cache["main"] = None
    _cache["ts"] = now
    return None


def resolve_aux():
    """Resolve the best available AUX backend (read-only resolution; aux failures
    don't stall the request — they degrade silently so the main path keeps working)."""
    now = time.time()
    if _cache["aux"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["aux"]

    catalog = load_yaml(CATALOG)
    models = catalog.get("models", {}) or {}

    for alias, model in models.items():
        if (model.get("role") or "either").lower() != "aux":
            continue
        port = model.get("port", 0)
        if backend_state(port) == "ready":
            result = {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
                "state": "ready",
            }
            _cache["aux"] = result
            _cache["ts"] = now
            return result

    _cache["aux"] = None
    _cache["ts"] = now
    return None


# ─── Stall-while-loading ─────────────────────────────────────────────────────

def stall_until_ready(port, deadline_ts):
    """Block (with periodic progress logs) until the local model is ready
    OR the deadline elapses. Returns the final state."""
    waited = 0.0
    poll = STALL_POLL_S
    last_log = 0.0
    while time.time() < deadline_ts:
        state = backend_state(port)
        if state == "ready":
            if waited > 1.0:
                log.info(f"Local backend :{port} ready after {waited:.1f}s stall")
            return state
        if state == "down":
            # Was loading, now down — likely crashed mid-load
            log.warning(f"Local backend :{port} went DOWN while we were waiting")
            return state
        if time.time() - last_log >= 5.0:
            log.info(f"Stalling — :{port} still loading ({waited:.0f}s / {STALL_TIMEOUT_S:.0f}s)")
            last_log = time.time()
        time.sleep(poll)
        waited += poll
        # Gentle backoff capped at 5s
        poll = min(poll * 1.1, 5.0)
    return backend_state(port)  # final state at deadline


# ─── HTTP handler ─────────────────────────────────────────────────────────────

class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "turbofit-gateway/2.0"

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
            self._handle_main(path)
        elif path.startswith("/aux/"):
            self._handle_aux(path)
        elif path == "/status":
            self._send_status()
        elif path == "/health":
            self._send_health()
        else:
            self.send_error(404, f"Unknown path: {path}")

    def _handle_main(self, path):
        upstream_path = path[len("/main/"):] or "/"

        deadline = time.time() + STALL_TIMEOUT_S
        stalled = False

        backend = resolve_main()
        if not backend:
            self._send_503("No backend available (no local model, no API fallback)", tried=None)
            return

        # Stall-while-loading: if local is loading, wait up to STALL_TIMEOUT_S
        if backend.get("state") == "loading":
            port = backend.get("port", 0)
            log.info(f"Stall-while-loading: :{port} (timeout {STALL_TIMEOUT_S:.0f}s)")
            stalled = True
            new_state = stall_until_ready(port, deadline)
            if new_state == "ready":
                backend["state"] = "ready"
            else:
                # Still not ready / went down — invalidate cache, try to resolve again
                _cache["main"] = None
                _cache["ts"] = 0
                backend = resolve_main()
                if not backend:
                    self._send_503("Local model failed to load and no API fallback",
                                   tried=f"local :{port} ({new_state})")
                    return
                if backend.get("state") == "loading":
                    # Still loading after stall timeout — last resort: API
                    api = _find_api_fallback_in_profiles()
                    if api:
                        backend = {**api, "state": "ready"}
                    else:
                        self._send_503("Local model still loading and no API fallback",
                                       tried=f"local :{port}")
                        return

        tried = [backend.get("alias") or backend.get("source") or "?"]
        result = self._proxy_to(backend, upstream_path)
        status = result["status"]

        # Graceful fallback: 4xx/5xx from a LOCAL backend → try API before giving up
        if status >= 400 and not backend.get("is_api"):
            api = _find_api_fallback_in_profiles()
            if api and api.get("source") not in tried:
                tried.append(api.get("source"))
                log.warning(f"Local {backend.get('alias')} returned {status} — falling back to API ({api.get('source')})")
                result = self._proxy_to({**api, "state": "ready"}, upstream_path)
                status = result["status"]

        if status >= 400:
            self._send_503(f"All backends failed (last status {status})", tried=" → ".join(tried))

    def _handle_aux(self, path):
        upstream_path = path[len("/aux/"):] or "/"
        backend = resolve_aux()
        if not backend:
            # Aux failures are non-fatal — return 200 with a structured "no aux" marker
            # so the main path's request still completes
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("X-Turbofit-Aux", "unavailable")
            self.end_headers()
            return
        result = self._proxy_to(backend, upstream_path)
        if result["status"] >= 400 and not backend.get("is_api"):
            api = _find_api_fallback_in_profiles()
            if api:
                result = self._proxy_to({**api, "state": "ready"}, upstream_path)
        # If even the aux fallback failed, the main path still got its response
        # above (we proxied main first), so we just return 204 here

    def _proxy_to(self, backend, upstream_path):
        target = f"{backend['base_url']}/{upstream_path.lstrip('/')}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "transfer-encoding", "content-length", "content-encoding")}

        # For local backends, inject the model id header (llama-server doesn't care,
        # but OpenAI-compatible APIs and the chat completions endpoint do require it).
        # We don't actually mutate the JSON body here — we trust the caller sent
        # the right model name. We just record the alias in an X- header for debugging.

        req = Request(target, data=body, headers=headers, method=self.command)
        start = time.time()
        try:
            with urlopen(req, timeout=PROXY_BACKEND_TIMEOUT_S) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
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
                self.send_header("X-Turbofit-Backend", str(backend.get("alias") or backend.get("source") or "api"))
                self.send_header("X-Turbofit-Latency-Ms", str(int((time.time() - start) * 1000)))
                self.end_headers()
                self.wfile.write(resp_body)
                return {"status": resp.status, "ms": int((time.time() - start) * 1000)}
        except HTTPError as e:
            # Read the upstream error body so we can forward it verbatim
            try:
                err_body = e.read()
            except Exception:
                err_body = str(e).encode()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("X-Turbofit-Backend", str(backend.get("alias") or backend.get("source") or "api"))
            self.end_headers()
            self.wfile.write(err_body)
            return {"status": e.code, "ms": int((time.time() - start) * 1000)}
        except (URLError, OSError) as e:
            log.error(f"Proxy error for {target}: {e}")
            return {"status": 502, "ms": int((time.time() - start) * 1000), "error": str(e)}
        except Exception as e:
            log.error(f"Unexpected error for {target}: {e}")
            return {"status": 500, "ms": int((time.time() - start) * 1000), "error": str(e)}

    def _send_503(self, reason, tried=None):
        body = json.dumps({
            "error": "no_backend",
            "message": reason,
            "tried": tried,
            "hint": "If this persists, run `serve status` and check `serve vram`.",
        }, indent=2).encode()
        log.error(f"503: {reason} (tried={tried})")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_status(self):
        main = resolve_main()
        aux = resolve_aux()
        response = {
            "main": main or {"alias": "none", "type": "down"},
            "aux": aux or {"alias": "none", "type": "down"},
            "stall_timeout_s": STALL_TIMEOUT_S,
            "backend_timeout_s": PROXY_BACKEND_TIMEOUT_S,
            "gateway": "turbofit-gateway/2.0",
        }
        data = json.dumps(response, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_health(self):
        main = resolve_main()
        aux = resolve_aux()
        ok = (main is not None) or (aux is not None)
        body = json.dumps({"ok": ok, "main": (main or {}).get("state", "down"),
                           "aux": (aux or {}).get("state", "down")}).encode()
        self.send_response(200 if ok else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        try:
            log.info(f"{self.command} {self.path}")
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("TURBOFIT_GATEWAY_PORT", "8091"))
    server = HTTPServer(("127.0.0.1", port), GatewayHandler)
    log.info(f"turbofit-gateway/2.0 on :{port} — graceful degradation active")
    log.info(f"  /main/ → stall-while-loading ({STALL_TIMEOUT_S:.0f}s) → API fallback")
    log.info(f"  /aux/  → ready-or-skip (no stall)")
    log.info(f"  /status → JSON, /health → 200/503")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown")
        server.shutdown()
