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
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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
STATE_HOME = os.environ.get("XDG_STATE_HOME", f"{HOME}/.local/state")
RUNTIME_STATE = os.environ.get(
    "TURBOFIT_RUNTIME_STATE",
    f"{STATE_HOME}/turbofit/runtime-state.json",
)
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILES = os.environ.get(
    "TURBOFIT_RUNTIME_PROFILES",
    str(SCRIPT_DIR.parent / "references" / "successful-runtime-profiles.json"),
)
RUNTIME_CLI = os.environ.get("TURBOFIT_RUNTIME_CLI", str(SCRIPT_DIR / "turbofit-runtime"))
RECOMMENDER = os.environ.get("TURBOFIT_RECOMMENDER", str(SCRIPT_DIR / "turbofit-runtime-recommend"))
SELF_PORT = int(os.environ.get("TURBOFIT_GATEWAY_PORT", "8091"))  # never pick a model on our own port

_activation_lock = threading.Lock()

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


def runtime_profiles():
    """Return the tested profile catalog keyed by stable, portable IDs."""
    try:
        with open(PROFILES) as f:
            data = json.load(f)
    except Exception:
        return {}
    profiles = data.get("profiles") or {}
    return profiles if isinstance(profiles, dict) else {}


def provider_models():
    """OpenAI-compatible catalog exposed by the single Turbofit provider."""
    models: list[dict] = [
        {
            "id": "auto",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Hardware-matched tested main + auxiliary configuration",
        },
        {
            "id": "active:main",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Stable route to the currently reconciled main role",
        },
        {
            "id": "active:aux",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Stable route to the currently reconciled auxiliary role",
        },
    ]
    for profile_id, profile in runtime_profiles().items():
        models.append({
            "id": profile_id,
            "object": "model",
            "owned_by": "turbofit",
            "context_length": int(profile.get("context") or 0),
            "description": profile.get("description") or profile_id,
        })
    return models


def parse_provider_model(model):
    """Parse universal IDs. Role suffixes are internal routes, not providers."""
    value = str(model or "auto").strip() or "auto"
    for suffix, role in ((":aux", "aux"), (":main", "main")):
        if value.endswith(suffix):
            return value[:-len(suffix)] or "auto", role
    return value, "main"


def active_profile():
    try:
        with open(RUNTIME_STATE) as f:
            return (json.load(f).get("active") or "").strip() or None
    except Exception:
        return None


def recommend_profile():
    """Scan hardware and return the best evidence-backed tested configuration."""
    try:
        result = subprocess.run(
            [RECOMMENDER, "--fit-only", "--limit", "1", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        rows = json.loads(result.stdout)
        profile = rows[0].get("profile") if rows else None
        return profile if profile in runtime_profiles() else None
    except Exception as exc:
        log.error("Hardware recommendation failed: %s", exc)
        return None


def activate_profile(profile):
    """Activate one tested main+aux pair by its universal catalog ID."""
    try:
        subprocess.run(
            [RUNTIME_CLI, "use", profile],
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
        _cache.update({"main": None, "aux": None, "ts": 0})
        return True
    except Exception as exc:
        log.error("Profile activation failed for %s: %s", profile, exc)
        return False


def resolve_requested_profile(model):
    """Resolve/activate a requested provider model and return the active profile.

    ``auto`` reuses an already-active pair so idle/warm requests never rerun a
    free-VRAM fit scan against Turbofit's own loaded models. It invokes the
    hardware recommender only when no runtime is active. ``active:main`` and
    ``active:aux`` likewise follow the selected pair, so auxiliary calls cannot
    undo a manual choice.
    """
    requested, role = parse_provider_model(model)
    if requested == "active":
        target = active_profile() or recommend_profile()
        if not target:
            return None
        with _activation_lock:
            if active_profile() != target and not activate_profile(target):
                return None
        return target

    if requested == "auto":
        target = active_profile()
        if target:
            return target
        if role == "aux":
            return None

    profiles = runtime_profiles()
    if requested == "auto":
        target = recommend_profile()
    elif requested in profiles:
        target = requested
    else:
        return None
    if not target:
        return None

    with _activation_lock:
        if active_profile() != target and not activate_profile(target):
            return None
    return target


# ─── Health probes ────────────────────────────────────────────────────────────

def port_is_open(port, timeout=PORT_PROBE_TIMEOUT_S):
    """Cheap TCP connect check — port is bound, even if the model isn't loaded yet."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _contains_model(value, alias):
    if isinstance(value, str):
        return value == alias
    if isinstance(value, list):
        return any(_contains_model(item, alias) for item in value)
    if isinstance(value, dict):
        if any(value.get(key) == alias for key in ("model_tag", "model", "name", "tag")):
            return True
        return any(_contains_model(item, alias) for item in value.values())
    return False


def _turbohaul_model_state(port, alias):
    """Return a model-specific manager state, or None when this is not Turbohaul."""
    try:
        response = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/status"],
            capture_output=True, text=True, timeout=5,
        )
        if response.returncode != 0:
            return None
        status = json.loads(response.stdout)
        if not isinstance(status, dict) or not isinstance(status.get("residents"), list):
            return None
        if any(
            _contains_model(status.get(key), alias)
            for key in ("active", "grace", "idle_hot", "residents")
        ):
            return "ready"
        if any(
            _contains_model(status.get(key), alias)
            for key in ("loading", "queue")
        ):
            return "loading"
        return "down"
    except Exception:
        return None


def check_port(port):
    """Full HTTP health check — model is actually serving completions."""
    try:
        r = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/health"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            health = r.stdout.strip()
            if health == "ok":
                return True
            try:
                if json.loads(health).get("status") in {"ok", "ready", "healthy", "loaded"}:
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
        # Some OpenAI servers expose only /v1/models.
        r2 = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/v1/models"],
            capture_output=True, text=True, timeout=5
        )
        if r2.returncode == 0 and '"data"' in r2.stdout:
            return True
        # Turbohaul Manager owns model sidecars behind one stable port and
        # exposes /status instead of pretending the manager itself is a model.
        r3 = subprocess.run(
            ["curl", "-s", "--max-time", "3", f"http://127.0.0.1:{port}/status"],
            capture_output=True, text=True, timeout=5
        )
        if r3.returncode == 0:
            try:
                return isinstance(json.loads(r3.stdout).get("residents"), list)
            except (json.JSONDecodeError, AttributeError):
                pass
        return False
    except Exception:
        return False


def runtime_override(role):
    """Resolve an explicitly activated evidence-backed runtime before catalog roles.

    This lets a model serve as main in one profile and auxiliary in another
    without mutating the global catalog between swaps.
    """
    try:
        with open(RUNTIME_STATE) as f:
            state = json.load(f)
    except Exception:
        return None
    if not state.get("active"):
        return None
    routes = state.get("routes")
    if isinstance(routes, dict):
        return _runtime_policy_route(state, role)
    expected = state.get("expected") or {}
    components = state.get("components") or []
    if role == "main":
        component = next((item for item in components if item.get("role") == "main"), None)
        alias = expected.get("main_alias")
        mode = None
    else:
        component = next((item for item in components if item.get("role") == "aux"), None)
        alias = expected.get("aux_alias")
        mode = expected.get("aux_mode")
        if component is None and mode == "shared-main":
            component = next((item for item in components if item.get("role") == "main"), None)
    if not component or not alias:
        return None
    port = int(component.get("port") or 0)
    state_name = backend_state(port, alias)
    if state_name == "down":
        return None
    result = {
        "alias": alias,
        "base_url": f"http://127.0.0.1:{port}",
        "port": port,
        "state": state_name,
        "runtime_profile": state.get("active"),
    }
    if role == "aux":
        result["mode"] = mode or "dedicated"
        if result["mode"] == "shared-main":
            result["shared_main_alias"] = expected.get("main_alias")
    return result


def _runtime_policy_route(state, role):
    """Resolve one atomically published reconciler route without stale caching."""
    routes = state.get("routes") or {}
    route = routes.get(role)
    if not isinstance(route, dict):
        return None
    kind = route.get("kind")
    if role == "aux" and kind == "shared-main":
        main = _runtime_policy_route(state, "main")
        if not main:
            return None
        return {
            **main,
            "alias": f"auto:{main.get('alias', 'main')}",
            "mode": "shared-main",
            "shared_main_alias": main.get("alias"),
        }
    if kind == "api-policy":
        if route.get("policy") != "api:auto":
            return None
        fallback = _find_api_fallback_in_profiles()
        if not fallback:
            return None
        result = {
            **fallback,
            "state": "ready",
            "runtime_profile": state.get("active"),
            "runtime_rung": state.get("rung_id"),
        }
        if role == "aux":
            result["mode"] = "api"
        return result
    alias = str(route.get("alias") or "").strip()
    if not alias:
        return None
    if kind == "local":
        try:
            port = int(route.get("port") or 0)
        except (TypeError, ValueError):
            return None
        state_name = backend_state(port, alias)
        if state_name == "down":
            return None
        result = {
            "alias": alias,
            "base_url": f"http://127.0.0.1:{port}",
            "port": port,
            "state": state_name,
            "runtime_profile": state.get("active"),
            "runtime_rung": state.get("rung_id"),
        }
        if role == "aux":
            result["mode"] = str(route.get("mode") or "dedicated")
        return result
    if kind == "api":
        base_url = str(route.get("base_url") or "").rstrip("/")
        model_id = str(route.get("model_id") or "").strip()
        if not base_url.startswith(("https://", "http://")) or not model_id:
            return None
        result = {
            "alias": alias,
            "base_url": base_url.removesuffix("/v1").rstrip("/"),
            "port": 0,
            "state": "ready",
            "is_api": True,
            "model_id": model_id,
            "provider": str(route.get("provider") or ""),
            "runtime_profile": state.get("active"),
            "runtime_rung": state.get("rung_id"),
        }
        if role == "aux":
            result["mode"] = "api"
        return result
    return None


def backend_state(port, alias=None):
    """Returns one of: 'ready', 'loading', 'down'.

    The port-SELF_PORT guard prevents a model registered on the gateway's own
    port from being picked (which would create a recursive proxy loop).
    Turbohaul's shared manager port is checked against the requested model tag;
    another resident model cannot make this route appear ready.
    """
    if not port or port == SELF_PORT:
        return "down"
    if not port_is_open(port):
        return "down"
    if alias:
        manager_state = _turbohaul_model_state(port, alias)
        if manager_state is not None:
            return manager_state
    if check_port(port):
        return "ready"
    return "loading"


# ─── Backend resolvers ────────────────────────────────────────────────────────

def _get_api_key(provider):
    """Read the API key for a provider from ~/.hermes/auth.json."""
    if not provider:
        return None
    auth_file = os.path.join(HERMES_HOME, "auth.json")
    try:
        with open(auth_file) as f:
            auth = json.load(f)
        # Strip "custom:" prefix if present
        prov_key = provider.replace("custom:", "") if provider.startswith("custom:") else provider
        return auth.get("providers", {}).get(prov_key, {}).get("access_token")
    except Exception:
        return None


def _find_api_fallback_in_profiles():
    """Resolve the explicit Turbofit fallback, then search Hermes profiles.

    OAuth/browser-session endpoints are not valid unattended fallbacks. The
    explicit ``preferences.yaml`` route wins so a profile's current interactive
    provider cannot accidentally become the runtime safety net.
    """
    prefs = load_yaml(PREFS)
    configured = prefs.get("api_fallback", {}) or {}
    url = str(configured.get("base_url") or "").strip()
    default = str(configured.get("main") or "").strip()
    provider = str(configured.get("provider") or "").strip()
    if url and default and "127.0.0.1" not in url and "localhost" not in url:
        return {
            "alias": "api-fallback",
            "base_url": url.rstrip("/").removesuffix("/v1").rstrip("/"),
            "port": 0,
            "is_api": True,
            "model_id": default,
            "provider": provider,
            "source": os.path.relpath(PREFS, HOME),
        }

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
            if provider in {"openai-codex", "custom:openai-codex"} or "chatgpt.com" in url:
                continue
            return {
                "alias": "api-fallback",
                "base_url": url.rstrip("/").removesuffix("/v1").rstrip("/"),
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
    override = runtime_override("main")
    if override:
        return override
    now = time.time()
    if _cache["main"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["main"]

    ladder, models = _local_ladder()

    # 1. First local that is READY wins
    for alias in ladder:
        m = models.get(alias, {}) or {}
        port = m.get("port", 0)
        state = backend_state(port, alias)
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
        if backend_state(port, alias) == "loading":
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
    override = runtime_override("aux")
    if override:
        return override
    now = time.time()
    if _cache["aux"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["aux"]

    catalog = load_yaml(CATALOG)
    models = catalog.get("models", {}) or {}

    for alias, model in models.items():
        if (model.get("role") or "either").lower() != "aux":
            continue
        port = model.get("port", 0)
        if backend_state(port, alias) == "ready":
            result = {
                "alias": alias,
                "base_url": f"http://127.0.0.1:{port}",
                "port": port,
                "state": "ready",
                "mode": "dedicated",
            }
            _cache["aux"] = result
            _cache["ts"] = now
            return result

    # `auto` is a real auxiliary policy, not "no route": when no dedicated
    # auxiliary is healthy, reuse the selected main backend. This keeps tool,
    # vision, and background calls working without an external API key while
    # preserving a clear status marker so benchmarks can distinguish shared
    # fallback from a dedicated drafter/auxiliary server.
    main = resolve_main()
    if main and main.get("state") == "ready":
        result = {
            **main,
            "alias": f"auto:{main.get('alias', 'main')}",
            "mode": "shared-main",
            "shared_main_alias": main.get("alias"),
        }
        _cache["aux"] = result
        _cache["ts"] = now
        return result

    _cache["aux"] = None
    _cache["ts"] = now
    return None


# ─── Stall-while-loading ─────────────────────────────────────────────────────

def stall_until_ready(port, deadline_ts, alias=None):
    """Block (with periodic progress logs) until the local model is ready
    OR the deadline elapses. Returns the final state."""
    waited = 0.0
    poll = STALL_POLL_S
    last_log = 0.0
    while time.time() < deadline_ts:
        state = backend_state(port, alias)
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
    return backend_state(port, alias)  # final state at deadline


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

        if path == "/v1/models":
            self._send_provider_models()
        elif path.startswith("/v1/"):
            self._handle_unified(path)
        elif path.startswith("/main/"):
            self._handle_main(path)
        elif path.startswith("/aux/"):
            self._handle_aux(path)
        elif path == "/status":
            self._send_status()
        elif path == "/health":
            self._send_health()
        else:
            self.send_error(404, f"Unknown path: {path}")

    def _send_provider_models(self):
        body = json.dumps({"object": "list", "data": provider_models()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_unified(self, path):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        model = "auto"
        if body:
            try:
                model = json.loads(body).get("model") or "auto"
            except Exception:
                self.send_error(400, "Request body must be valid JSON")
                return
        profile_id, role = parse_provider_model(model)
        selected = resolve_requested_profile(model)
        if not selected:
            self.send_error(400, f"Unknown or unavailable Turbofit model: {profile_id}")
            return
        suffix = path[len("/v1/"):]
        routed_path = f"/{role}/v1/{suffix}"
        if role == "aux":
            self._handle_aux(routed_path, body=body)
        else:
            self._handle_main(routed_path, body=body)

    def _handle_main(self, path, body=None):
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
            new_state = stall_until_ready(port, deadline, backend.get("alias"))
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
        result = self._proxy_to(backend, upstream_path, body=body)
        status = result["status"]

        # Graceful fallback: 4xx/5xx from a LOCAL backend → try API before giving up
        if status >= 400 and not backend.get("is_api"):
            api = _find_api_fallback_in_profiles()
            if api and api.get("source") not in tried:
                tried.append(api.get("source"))
                log.warning(f"Local {backend.get('alias')} returned {status} — falling back to API ({api.get('source')})")
                result = self._proxy_to({**api, "state": "ready"}, upstream_path, body=body)
                status = result["status"]

        if status >= 400:
            self._send_503(f"All backends failed (last status {status})", tried=" → ".join(tried))

    def _handle_aux(self, path, body=None):
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
        result = self._proxy_to(backend, upstream_path, body=body)
        if result["status"] >= 400 and not backend.get("is_api"):
            api = _find_api_fallback_in_profiles()
            if api:
                result = self._proxy_to({**api, "state": "ready"}, upstream_path, body=body)
        # If even the aux fallback failed, the main path still got its response
        # above (we proxied main first), so we just return 204 here

    def _proxy_to(self, backend, upstream_path, body=None):
        target = f"{backend['base_url']}/{upstream_path.lstrip('/')}"
        if body is None:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "transfer-encoding", "content-length", "content-encoding",
                                        "authorization")}  # we re-inject auth below

        # Universal provider IDs are profile IDs, not backend model filenames.
        # Rewrite every request to the selected backend's actual model ID.
        if body:
            try:
                payload = json.loads(body)
                payload["model"] = (
                    backend.get("model_id") if backend.get("is_api")
                    else backend.get("alias")
                ) or payload.get("model")
                body = json.dumps(payload).encode()
            except Exception:
                pass

        # API fallback also needs the real provider credential.
        if backend.get("is_api"):
            api_key = _get_api_key(backend.get("provider"))
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

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
    server = ThreadingHTTPServer(("127.0.0.1", port), GatewayHandler)
    log.info(f"turbofit-gateway/2.0 on :{port} — graceful degradation active")
    log.info(f"  /main/ → stall-while-loading ({STALL_TIMEOUT_S:.0f}s) → API fallback")
    log.info(f"  /aux/  → ready-or-skip (no stall)")
    log.info(f"  /status → JSON, /health → 200/503")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutdown")
        server.shutdown()
