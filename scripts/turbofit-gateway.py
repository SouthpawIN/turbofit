#!/usr/bin/env python3
"""
turbofit-gateway — dynamic local reverse proxy for Hermes Agent.

Sits behind nginx on port 8091 and follows the atomically published local
main/aux routes. The default provider is local-only: unavailable local models
return a clear 503/204 rather than invoking an API model. Legacy API routing
code is disabled unless TURBOFIT_ALLOW_API=true is explicitly set.

Runs on :8091
"""

import hashlib
import http.client
import json
import math
import select
import socket
import subprocess
import os
import sys
import time
import logging
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlsplit
from urllib.request import urlopen

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
ALLOW_API = os.environ.get("TURBOFIT_ALLOW_API", "").strip().lower() in {"1", "true", "yes"}

_activation_lock = threading.Lock()
_inflight_lock = threading.Lock()
_inflight_requests: dict[str, float] = {}

_cache = {"main": None, "aux": None, "ts": 0}
CACHE_TTL = 10

# Graceful-degradation tunables (overridable via env)
STALL_TIMEOUT_S = float(os.environ.get("TURBOFIT_STALL_TIMEOUT", "90"))  # max wait while local loads
STALL_POLL_S = float(os.environ.get("TURBOFIT_STALL_POLL", "2"))  # poll interval while waiting
PROXY_BACKEND_TIMEOUT_S = float(os.environ.get("TURBOFIT_BACKEND_TIMEOUT", "300"))  # per-request upstream timeout
AUX_MAX_TOKENS = int(os.environ.get("TURBOFIT_AUX_MAX_TOKENS", "4096"))  # bound aux work to lifecycle deadlines
AUX_ENABLE_THINKING = os.environ.get("TURBOFIT_AUX_ENABLE_THINKING", "0").lower() in ("1", "true", "yes")
MAIN_ENABLE_THINKING = os.environ.get("TURBOFIT_MAIN_ENABLE_THINKING", "1").lower() in ("1", "true", "yes")
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
    context_length = active_context_length()
    models: list[dict] = [
        {
            "id": "auto",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Hardware-matched tested main + auxiliary configuration",
            "context_length": context_length,
        },
        {
            "id": "active:main",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Stable route to the currently reconciled main role",
            "context_length": context_length,
        },
        {
            "id": "active:aux",
            "object": "model",
            "owned_by": "turbofit",
            "description": "Stable route to the currently reconciled auxiliary role",
            "context_length": context_length,
        },
    ]
    return models


def active_context_length(default=65536):
    """Return the active route's configured context, not the model's trained maximum."""
    try:
        with open(RUNTIME_STATE) as f:
            state = json.load(f)
        route = (state.get("routes") or {}).get("main") or {}
        value = route.get("context_length") or state.get("context_length")
        if value is None:
            profile = runtime_profiles().get(state.get("active")) or {}
            value = profile.get("context")
        value = int(value)
        return value if value > 0 else default
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


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
            # Shared auxiliary work is the same concrete Turbohaul resident.
            # `auto:` is a public Turbofit selector, not a valid manager tag.
            "alias": main.get("alias", "main"),
            "mode": "shared-main",
            "shared_main_alias": main.get("alias"),
        }
    if kind == "api-policy":
        if not ALLOW_API or route.get("policy") != "api:auto":
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
        if route.get("context_length"):
            result["context_length"] = route["context_length"]
        if isinstance(route.get("request_policy"), dict):
            result["request_policy"] = dict(route["request_policy"])
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
        if route.get("context_length"):
            result["context_length"] = route["context_length"]
        if isinstance(route.get("request_policy"), dict):
            result["request_policy"] = dict(route["request_policy"])
        if role == "aux":
            result["mode"] = str(route.get("mode") or "dedicated")
        return result
    if kind == "api":
        if not ALLOW_API:
            return None
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

    # 3. API routing is explicit opt-in; the shipped provider is local-only.
    if ALLOW_API:
        api = _find_api_fallback_in_profiles()
        if api:
            result = {**api, "state": "ready"}
            _cache["main"] = result
            _cache["ts"] = now
            return result

    # 4. No local backend is available — caller returns a clear 503.
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
        elif path == "/v1/props":
            self._send_provider_props()
        elif path.startswith("/v1/models/") and "/" not in path[len("/v1/models/"):]:
            self._send_provider_model(path[len("/v1/models/"):])
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
        self._send_json(200, {"object": "list", "data": provider_models()})

    def _send_provider_model(self, model_id):
        model = next((item for item in provider_models() if item["id"] == model_id), None)
        if model is None:
            self._send_json(404, {"error": {"message": f"Unknown model: {model_id}"}})
            return
        self._send_json(200, model)

    def _send_provider_props(self):
        """Adapt llama.cpp's /props endpoint to the /v1/props probe Hermes uses."""
        backend = resolve_main()
        props = None
        if backend and not backend.get("is_api"):
            try:
                with urlopen(f"{backend['base_url']}/props", timeout=3) as response:
                    candidate = json.load(response)
                    if isinstance(candidate, dict):
                        props = candidate
            except Exception:
                props = None
        if props is None:
            props = {
                "model_alias": (backend or {}).get("alias", "auto"),
                "default_generation_settings": {"n_ctx": active_context_length()},
            }
        props["provider_model"] = "auto"
        props["context_length"] = int(
            (props.get("default_generation_settings") or {}).get("n_ctx")
            or (backend or {}).get("context_length")
            or active_context_length()
        )
        self._send_json(200, props)

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
            self._send_503("No local backend available", tried=None)
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
                    # Still loading after the stall timeout. API use remains opt-in.
                    api = _find_api_fallback_in_profiles() if ALLOW_API else None
                    if api:
                        backend = {**api, "state": "ready"}
                    else:
                        self._send_503("Local model still loading and no API fallback",
                                       tried=f"local :{port}")
                        return

        tried = [backend.get("alias") or backend.get("source") or "?"]
        result = self._proxy_to(backend, upstream_path, body=body, role="main")
        status = result["status"]
        if result.get("response_sent"):
            return

        # API fallback is disabled by default; explicit opt-in preserves legacy behavior.
        if status >= 400 and not backend.get("is_api") and ALLOW_API:
            api = _find_api_fallback_in_profiles()
            if api and api.get("source") not in tried:
                tried.append(api.get("source"))
                log.warning(f"Local {backend.get('alias')} returned {status} — falling back to API ({api.get('source')})")
                result = self._proxy_to({**api, "state": "ready"}, upstream_path, body=body, role="main")
                status = result["status"]
                if result.get("response_sent"):
                    return

        if status >= 400:
            self._send_503(f"All backends failed (last status {status})", tried=" → ".join(tried))

    def _handle_aux(self, path, body=None):
        upstream_path = path[len("/aux/"):] or "/"
        backend = resolve_aux()
        if not backend:
            # Aux failures are non-fatal — return 200 with a structured "no aux" marker
            # so the main path's request still completes
            self._send_empty(204, {"X-Turbofit-Aux": "unavailable"})
            return
        result = self._proxy_to(backend, upstream_path, body=body, role="aux")
        if result.get("response_sent"):
            return
        if result["status"] >= 400 and not backend.get("is_api") and ALLOW_API:
            api = _find_api_fallback_in_profiles()
            if api:
                result = self._proxy_to({**api, "state": "ready"}, upstream_path, body=body, role="aux")
                if result.get("response_sent"):
                    return
        # If even the aux fallback failed, the main path still got its response
        # above (we proxied main first), so we just return 204 here.
        self._send_empty(204, {"X-Turbofit-Aux": "unavailable"})

    def _proxy_to(self, backend, upstream_path, body=None, role=None):
        target = f"{backend['base_url']}/{upstream_path.lstrip('/')}"
        if body is None:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "transfer-encoding", "content-length", "content-encoding",
                                        "authorization")}  # we re-inject auth below

        stream_requested = False
        # Universal provider IDs are profile IDs, not backend model filenames.
        # Rewrite every request to the selected backend's actual model ID.
        if body:
            try:
                payload = json.loads(body)
                stream_requested = payload.get("stream") is True
                if role == "main" and not MAIN_ENABLE_THINKING:
                    template_kwargs = payload.get("chat_template_kwargs")
                    if not isinstance(template_kwargs, dict):
                        template_kwargs = {}
                    else:
                        template_kwargs = dict(template_kwargs)
                    template_kwargs["enable_thinking"] = False
                    template_kwargs["thinking_mode"] = "disabled"
                    payload["chat_template_kwargs"] = template_kwargs
                    payload["reasoning_format"] = "none"
                if role == "aux" and AUX_MAX_TOKENS > 0:
                    requested = payload.get("max_tokens")
                    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
                        payload["max_tokens"] = AUX_MAX_TOKENS
                    else:
                        payload["max_tokens"] = min(requested, AUX_MAX_TOKENS)
                    if "n_predict" in payload:
                        requested_predict = payload["n_predict"]
                        if isinstance(requested_predict, bool) or not isinstance(requested_predict, int) or requested_predict <= 0:
                            payload["n_predict"] = AUX_MAX_TOKENS
                        else:
                            payload["n_predict"] = min(requested_predict, AUX_MAX_TOKENS)
                    template_kwargs = payload.get("chat_template_kwargs")
                    if not isinstance(template_kwargs, dict):
                        template_kwargs = {}
                    else:
                        template_kwargs = dict(template_kwargs)
                    template_kwargs.setdefault("enable_thinking", AUX_ENABLE_THINKING)
                    template_kwargs.setdefault("thinking_mode", "enabled" if AUX_ENABLE_THINKING else "disabled")
                    payload["chat_template_kwargs"] = template_kwargs
                    payload.setdefault("reasoning_format", "none")
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

        request_key = self._request_key(target, body)
        if request_key and not self._claim_request(request_key):
            self._send_json(
                409,
                {
                    "error": "request_in_progress",
                    "message": "An identical request is already running",
                    "retryable": True,
                },
                {"Retry-After": "1", "X-Turbofit-Deduplicated": "true"},
            )
            return {"status": 409, "ms": 0, "response_sent": True, "deduplicated": True}

        timeout_s = self._backend_timeout(backend, body)
        parsed = urlsplit(target)
        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_class(parsed.hostname, parsed.port, timeout=timeout_s)
        target_path = parsed.path or "/"
        if parsed.query:
            target_path += f"?{parsed.query}"
        disconnected = threading.Event()
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=self._monitor_disconnect,
            args=(connection, disconnected, monitor_stop),
            daemon=True,
        )
        start = time.time()
        try:
            connection.request(self.command, target_path, body=body, headers=headers)
            monitor.start()
            response = connection.getresponse()
            try:
                content_type = response.headers.get("Content-Type", "")
                streaming = stream_requested or content_type.lower().startswith("text/event-stream")
                if response.status >= 400:
                    return {
                        "status": response.status,
                        "ms": int((time.time() - start) * 1000),
                        "error_body": response.read(),
                    }

                self.send_response(response.status)
                sent_headers = set()
                for key, value in response.headers.items():
                    normalized = key.lower()
                    if normalized in (
                        "transfer-encoding",
                        "connection",
                        "content-length",
                        "content-encoding",
                    ):
                        continue
                    if normalized in sent_headers:
                        continue
                    sent_headers.add(normalized)
                    self.send_header(key, value)
                self.send_header(
                    "X-Turbofit-Backend",
                    str(backend.get("alias") or backend.get("source") or "api"),
                )
                self.send_header(
                    "X-Turbofit-Latency-Ms",
                    str(int((time.time() - start) * 1000)),
                )
                self.send_header("X-Turbofit-Timeout-S", str(round(timeout_s, 3)))
                if streaming:
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        while True:
                            chunk = response.read1(65536)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        log.info(f"Client disconnected from streaming proxy for {target}")
                        disconnected.set()
                        self._close_upstream(connection)
                        return {
                            "status": 499,
                            "ms": int((time.time() - start) * 1000),
                            "client_disconnected": True,
                            "response_sent": True,
                        }
                    return {
                        "status": response.status,
                        "ms": int((time.time() - start) * 1000),
                        "response_sent": True,
                    }

                response_body = response.read()
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self._write_body(response_body)
                return {
                    "status": response.status,
                    "ms": int((time.time() - start) * 1000),
                    "response_sent": True,
                }
            finally:
                try:
                    response.close()
                except Exception:
                    pass
        except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as exc:
            if disconnected.is_set():
                log.info(f"Cancelled upstream request after client disconnect: {target}")
                return {
                    "status": 499,
                    "ms": int((time.time() - start) * 1000),
                    "client_disconnected": True,
                    "response_sent": True,
                }
            log.error(f"Proxy error for {target}: {exc}")
            return {
                "status": 502,
                "ms": int((time.time() - start) * 1000),
                "error": str(exc),
            }
        except Exception as exc:
            if disconnected.is_set():
                log.info(f"Cancelled upstream request after client disconnect: {target}")
                return {
                    "status": 499,
                    "ms": int((time.time() - start) * 1000),
                    "client_disconnected": True,
                    "response_sent": True,
                }
            log.error(f"Unexpected error for {target}: {exc}")
            return {
                "status": 500,
                "ms": int((time.time() - start) * 1000),
                "error": str(exc),
            }
        finally:
            monitor_stop.set()
            self._close_upstream(connection)
            if request_key:
                with _inflight_lock:
                    _inflight_requests.pop(request_key, None)

    def _request_key(self, target, body):
        if self.command != "POST" or not target.endswith("/v1/chat/completions") or not body:
            return None
        return hashlib.sha256(target.encode() + b"\0" + body).hexdigest()

    def _claim_request(self, key):
        with _inflight_lock:
            if key in _inflight_requests:
                return False
            _inflight_requests[key] = time.time()
            return True

    def _backend_timeout(self, backend, body):
        policy = backend.get("request_policy")
        if not isinstance(policy, dict):
            return PROXY_BACKEND_TIMEOUT_S
        base = float(policy.get("initial_response_timeout_s") or PROXY_BACKEND_TIMEOUT_S)
        floor = float(policy.get("prefill_tokens_per_second_floor") or 0)
        maximum = float(policy.get("maximum_timeout_s") or max(base, PROXY_BACKEND_TIMEOUT_S))
        grace = float(policy.get("generation_grace_s") or 120)
        if floor <= 0 or not body:
            return min(base, maximum)
        try:
            payload = json.loads(body)
            prompt_chars = len(json.dumps(payload.get("messages") or [], ensure_ascii=False))
            prompt_chars += len(json.dumps(payload.get("tools") or [], ensure_ascii=False))
            chars_per_token = float(policy.get("estimated_chars_per_token") or 3.0)
            estimated_tokens = max(1, math.ceil(prompt_chars / chars_per_token))
            return min(maximum, max(base, grace + estimated_tokens / floor))
        except (TypeError, ValueError, json.JSONDecodeError):
            return min(base, maximum)

    def _monitor_disconnect(self, upstream, disconnected, stop):
        while not stop.wait(0.1):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    continue
                data = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if data:
                    continue
                disconnected.set()
                self._close_upstream(upstream)
                return
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                disconnected.set()
                self._close_upstream(upstream)
                return

    @staticmethod
    def _close_upstream(connection):
        upstream_socket = getattr(connection, "sock", None)
        if upstream_socket is not None:
            try:
                upstream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            connection.close()
        except Exception:
            pass

    def _send_503(self, reason, tried=None):
        log.error(f"503: {reason} (tried={tried})")
        self._send_json(503, {
            "error": "no_backend",
            "message": reason,
            "tried": tried,
            "hint": "If this persists, run `serve status` and check `serve vram`.",
        })

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
        self._send_json(200, response)

    def _send_health(self):
        main = resolve_main()
        aux = resolve_aux()
        ok = (main is not None) or (aux is not None)
        self._send_json(200 if ok else 503, {
            "ok": ok,
            "main": (main or {}).get("state", "down"),
            "aux": (aux or {}).get("state", "down"),
        })

    def _send_json(self, status, payload, extra_headers=None):
        body = json.dumps(payload, indent=2).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            return self._write_body(body)
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _send_empty(self, status, extra_headers=None):
        try:
            self.send_response(status)
            self.send_header("Content-Length", "0")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def _write_body(self, body):
        try:
            self.wfile.write(body)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

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
