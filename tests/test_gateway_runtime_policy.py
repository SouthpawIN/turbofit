from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace


SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway_runtime_policy",
    Path(__file__).resolve().parents[1] / "scripts/turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


def write_state(path: Path, routes: dict) -> None:
    path.write_text(json.dumps({"active": "quality-24gb", "rung_id": "test", "routes": routes}))


def test_gateway_and_controller_share_the_same_default_route_state() -> None:
    assert GATEWAY.RUNTIME_STATE.endswith("/.local/state/turbofit/runtime-state.json")


def test_gateway_reads_utf8_bom_runtime_state(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    state.write_bytes(
        b"\xef\xbb\xbf"
        + json.dumps(
            {
                "active": "bom-profile",
                "routes": {
                    "main": {
                        "kind": "local",
                        "alias": "main",
                        "port": 8080,
                        "context_length": 131072,
                    },
                    "aux": {"kind": "shared-main", "context_length": 131072},
                },
            }
        ).encode()
    )
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "backend_state", lambda *_args, **_kwargs: "ready")

    assert GATEWAY.active_profile() == "bom-profile"
    assert GATEWAY.active_context_length() == 131072
    assert GATEWAY.runtime_override("main")["alias"] == "main"


def test_nous_api_auth_uses_hermes_refresh_aware_runtime_credentials(monkeypatch) -> None:
    package = types.ModuleType("hermes_cli")
    package.__path__ = []
    auth = types.ModuleType("hermes_cli.auth")
    setattr(
        auth,
        "resolve_nous_runtime_credentials",
        lambda **_kwargs: {"api_key": "fresh-invoke-jwt"},
    )
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth)

    assert GATEWAY._get_api_key("nous") == "fresh-invoke-jwt"


def test_health_probe_accepts_native_runtime_health(monkeypatch) -> None:
    reply = SimpleNamespace(returncode=0, stdout='{"status":"ok"}')
    monkeypatch.setattr(GATEWAY.subprocess, "run", lambda *args, **kwargs: reply)

    assert GATEWAY.check_port(8092) is True


def test_backend_state_rejects_a_healthy_listener_serving_the_wrong_model(monkeypatch) -> None:
    monkeypatch.setattr(GATEWAY, "port_is_open", lambda _port: True)
    monkeypatch.setattr(GATEWAY, "check_port", lambda _port: True)
    monkeypatch.setattr(GATEWAY, "served_model_ids", lambda _port: {"another-model"}, raising=False)

    assert GATEWAY.backend_state(18081, "expected-model") == "down"


def test_backend_state_returns_loading_when_server_is_up_but_model_not_listed(monkeypatch) -> None:
    monkeypatch.setattr(GATEWAY, "port_is_open", lambda _port: True)
    monkeypatch.setattr(GATEWAY, "check_port", lambda _port: True)
    monkeypatch.setattr(GATEWAY, "served_model_ids", lambda _port: set(), raising=False)

    assert GATEWAY.backend_state(18081, "expected-model") == "loading"


def test_explicit_api_fallback_precedes_interactive_profile_provider(tmp_path, monkeypatch) -> None:
    preferences = tmp_path / "preferences.yaml"
    preferences.write_text(
        "api_fallback:\n"
        "  main: z-ai/glm-5.2\n"
        "  base_url: https://inference-api.nousresearch.com/v1\n"
        "  provider: nous\n"
    )
    monkeypatch.setattr(GATEWAY, "PREFS", str(preferences))
    monkeypatch.setattr(GATEWAY, "HERMES_HOME", str(tmp_path / "hermes"))

    fallback = GATEWAY._find_api_fallback_in_profiles()

    assert fallback["provider"] == "nous"
    assert fallback["model_id"] == "z-ai/glm-5.2"
    assert fallback["base_url"] == "https://inference-api.nousresearch.com"


def test_campaign_lease_forces_api_fallback_without_touching_campaign_models(tmp_path, monkeypatch) -> None:
    lease = tmp_path / "campaign-lease.json"
    lease.write_text(json.dumps({
        "schema": "turbofit.campaign-lease/v1",
        "owner_pid": os.getpid(),
        "gateway_policy": "api-fallback-only",
    }))
    fallback = {
        "alias": "api-fallback",
        "base_url": "https://api.example.test",
        "port": 0,
        "is_api": True,
        "model_id": "safe-api-model",
        "provider": "example",
    }
    monkeypatch.setattr(GATEWAY, "CAMPAIGN_LEASE", str(lease))
    monkeypatch.setattr(GATEWAY, "ALLOW_API", True)
    monkeypatch.setattr(GATEWAY, "_find_api_fallback_in_profiles", lambda: fallback)
    monkeypatch.setattr(GATEWAY, "_get_api_key", lambda _provider: "credential")
    monkeypatch.setattr(
        GATEWAY, "runtime_override",
        lambda _role: (_ for _ in ()).throw(AssertionError("campaign local route must not be inspected")),
    )
    GATEWAY._cache.update({"main": None, "aux": None, "ts": 0})

    main = GATEWAY.resolve_main()
    aux = GATEWAY.resolve_aux()

    assert main["model_id"] == "safe-api-model"
    assert main["campaign_lease"] is True
    assert aux["model_id"] == "safe-api-model"
    assert aux["mode"] == "shared-main"


def test_temporary_campaign_gateway_ignores_production_campaign_lease(tmp_path, monkeypatch) -> None:
    lease = tmp_path / "campaign-lease.json"
    lease.write_text(json.dumps({
        "schema": "turbofit.campaign-lease/v1",
        "owner_pid": os.getpid(),
        "gateway_policy": "api-fallback-only",
    }))
    monkeypatch.setattr(GATEWAY, "CAMPAIGN_LEASE", str(lease))
    monkeypatch.setattr(GATEWAY, "CAMPAIGN_GATEWAY", True, raising=False)

    assert GATEWAY.campaign_lease_active() is False


def test_campaign_api_fallback_is_not_ready_without_credentials(tmp_path, monkeypatch) -> None:
    lease = tmp_path / "campaign-lease.json"
    lease.write_text(json.dumps({
        "schema": "turbofit.campaign-lease/v1",
        "owner_pid": os.getpid(),
        "gateway_policy": "api-fallback-only",
    }))
    monkeypatch.setattr(GATEWAY, "CAMPAIGN_LEASE", str(lease))
    monkeypatch.setattr(GATEWAY, "ALLOW_API", True)
    monkeypatch.setattr(GATEWAY, "_find_api_fallback_in_profiles", lambda: {
        "alias": "api-fallback", "provider": "nous", "is_api": True,
    })
    monkeypatch.setattr(GATEWAY, "_get_api_key", lambda _provider: None)
    GATEWAY._cache.update({"main": None, "aux": None, "ts": 0})

    assert GATEWAY.resolve_main() is None


def test_api_fallback_chain_prefers_explicit_then_discovered_free(tmp_path, monkeypatch) -> None:
    preferences = tmp_path / "preferences.yaml"
    preferences.write_text(
        "api_fallback:\n"
        "  main: explicit-model\n"
        "  base_url: https://explicit.example.test\n"
        "  provider: explicit-provider\n"
    )
    manifest = tmp_path / "fallback-manifest.yaml"
    manifest.write_text(
        "schema: turbofit.fallback-manifest/v1\n"
        "models:\n"
        "  free-model:\n"
        "    context_length: 8192\n"
        "    role: aux\n"
        "    source: fallback-manifest:nous\n"
        "    provider: nous\n"
    )
    monkeypatch.setattr(GATEWAY, "PREFS", str(preferences))
    monkeypatch.setattr(GATEWAY, "HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(GATEWAY, "FALLBACK_PATH", str(manifest))

    chain = GATEWAY._api_fallback_chain()

    assert len(chain) == 2
    assert chain[0]["model_id"] == "explicit-model"
    assert chain[0]["source"].endswith("preferences.yaml")
    assert chain[1]["model_id"] == "free-model"
    assert chain[1]["provider"] == "nous"


def test_dynamic_local_main_and_dedicated_aux_routes(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {"kind": "local", "alias": "grm", "port": 8080},
        "aux": {"kind": "local", "alias": "carwin", "port": 8089, "mode": "dedicated"},
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "backend_state", lambda port, alias=None: "ready")

    main = GATEWAY.runtime_override("main")
    aux = GATEWAY.runtime_override("aux")

    assert main["alias"] == "grm"
    assert main["base_url"] == "http://127.0.0.1:8080"
    assert aux["alias"] == "carwin"
    assert aux["mode"] == "dedicated"


def test_local_engine_route_can_separate_stable_alias_from_upstream_model_id(
    tmp_path, monkeypatch
) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {
            "kind": "local",
            "alias": "qwen3-8-27b-uncensored-mlx-8bit",
            "model_id": "/models/Qwen3.8-27B-Uncensored-MLX/8-bit",
            "port": 18081,
        },
        "aux": {"kind": "shared-main"},
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "backend_state", lambda _port, _alias=None: "ready")

    main = GATEWAY.runtime_override("main")

    assert main["alias"] == "qwen3-8-27b-uncensored-mlx-8bit"
    assert main["model_id"] == "/models/Qwen3.8-27B-Uncensored-MLX/8-bit"


def test_shared_main_aux_route_follows_current_main_without_restart(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {"kind": "local", "alias": "grm", "port": 8080},
        "aux": {"kind": "shared-main"},
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "backend_state", lambda port, alias=None: "ready")

    aux = GATEWAY.runtime_override("aux")

    assert aux["base_url"] == "http://127.0.0.1:8080"
    assert aux["alias"] == "grm"
    assert aux["mode"] == "shared-main"
    assert aux["shared_main_alias"] == "grm"


def test_terminal_api_rung_routes_main_and_aux_independently(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {
            "kind": "api",
            "alias": "main-api",
            "base_url": "https://api.example.test",
            "model_id": "main-model",
            "provider": "example",
        },
        "aux": {
            "kind": "api",
            "alias": "aux-api",
            "base_url": "https://api.example.test",
            "model_id": "aux-model",
            "provider": "example",
        },
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "ALLOW_API", True)

    main = GATEWAY.runtime_override("main")
    aux = GATEWAY.runtime_override("aux")

    assert main["is_api"] is True and main["model_id"] == "main-model"
    assert aux["is_api"] is True and aux["model_id"] == "aux-model"


def test_terminal_api_policy_resolves_current_configured_fallback(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {"kind": "api-policy", "policy": "api:auto"},
        "aux": {"kind": "api-policy", "policy": "api:auto"},
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "ALLOW_API", True)
    monkeypatch.setattr(
        GATEWAY,
        "_find_api_fallback_in_profiles",
        lambda: {
            "alias": "configured-api",
            "base_url": "https://api.example.test",
            "model_id": "configured-model",
            "provider": "configured-provider",
            "is_api": True,
            "port": 0,
        },
    )

    main = GATEWAY.runtime_override("main")
    aux = GATEWAY.runtime_override("aux")

    assert main["model_id"] == "configured-model"
    assert aux["model_id"] == "configured-model"
    assert aux["mode"] == "api"


def test_api_routes_fail_closed_without_explicit_opt_in(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {"kind": "api-policy", "policy": "api:auto"},
        "aux": {
            "kind": "api",
            "alias": "aux-api",
            "base_url": "https://api.example.test",
            "model_id": "aux-model",
        },
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "ALLOW_API", False)
    monkeypatch.setattr(
        GATEWAY,
        "_find_api_fallback_in_profiles",
        lambda: (_ for _ in ()).throw(AssertionError("API fallback must not be resolved")),
    )

    assert GATEWAY.runtime_override("main") is None
    assert GATEWAY.runtime_override("aux") is None


def test_warm_requests_observe_newly_published_route_state(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))
    monkeypatch.setattr(GATEWAY, "backend_state", lambda port, alias=None: "ready")
    write_state(state, {
        "main": {"kind": "local", "alias": "first", "port": 8080},
        "aux": {"kind": "shared-main"},
    })
    assert GATEWAY.runtime_override("main")["alias"] == "first"

    write_state(state, {
        "main": {"kind": "local", "alias": "second", "port": 8081},
        "aux": {"kind": "shared-main"},
    })
    assert GATEWAY.runtime_override("main")["alias"] == "second"


def test_provider_catalog_advertises_stable_auto_and_role_ids(tmp_path, monkeypatch) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"profiles": {}}))
    monkeypatch.setattr(GATEWAY, "PROFILES", str(profiles))

    ids = [item["id"] for item in GATEWAY.provider_models()]

    assert ids[0] == "auto"
    assert "active:main" not in ids
    assert "active:aux" not in ids


def test_malformed_or_partial_published_routes_fail_closed(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))

    write_state(state, {
        "main": {"kind": "local", "alias": "missing-port"},
        "aux": {"kind": "shared-main"},
    })
    assert GATEWAY.runtime_override("main") is None
    assert GATEWAY.runtime_override("aux") is None

    write_state(state, {
        "main": {"kind": "shared-main"},
        "aux": {"kind": "shared-main"},
    })
    assert GATEWAY.runtime_override("aux") is None

    write_state(state, {
        "main": {"kind": "api", "alias": "api", "base_url": "https://example.test"},
        "aux": {"kind": "shared-main"},
    })
    assert GATEWAY.runtime_override("main") is None


def test_local_route_cannot_proxy_back_to_gateway_port(tmp_path, monkeypatch) -> None:
    state = tmp_path / "runtime-state.json"
    write_state(state, {
        "main": {"kind": "local", "alias": "loop", "port": GATEWAY.SELF_PORT},
        "aux": {"kind": "shared-main"},
    })
    monkeypatch.setattr(GATEWAY, "RUNTIME_STATE", str(state))

    assert GATEWAY.runtime_override("main") is None
    assert GATEWAY.runtime_override("aux") is None


def test_provider_catalog_always_includes_auto_local_and_tailnet_sections(
    tmp_path, monkeypatch
) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"profiles": {}}))
    monkeypatch.setattr(GATEWAY, "PROFILES", str(profiles))
    monkeypatch.setattr(GATEWAY, "active_context_length", lambda role=None: 8192)
    monkeypatch.setattr(GATEWAY, "provider_model_metadata", lambda: {})
    monkeypatch.setattr(
        GATEWAY, "_discover_local_models",
        lambda: [{"id": "local:deepnet", "object": "model", "owned_by": "turbofit",
                  "metadata": {"role": "local"}}],
    )
    monkeypatch.setattr(
        GATEWAY, "_discover_remote_models",
        lambda: [{"id": "omarchy.ts.net/quality-24gb", "object": "model",
                  "owned_by": "turbofit", "source": "remote",
                  "metadata": {"role": "remote"}}],
    )

    catalog = GATEWAY.provider_models()
    ids = [item["id"] for item in catalog]
    assert ids[0] == "auto"
    assert "local:deepnet" in ids
    assert "omarchy.ts.net/quality-24gb" in ids
    remote = next(item for item in catalog if item["id"] == "omarchy.ts.net/quality-24gb")
    assert remote["source"] == "remote"


def test_remote_discovery_uses_tailnet_peers_and_explicit_overrides(monkeypatch) -> None:
    import turbofit_runtime.tailnet as tailnet

    class FakeHandle:
        def __init__(self, models):
            self._models = models
        def read(self):
            return json.dumps({"data": self._models}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    tailnet_models = [
        {"id": "auto"},  # must be skipped
        {"id": "quality-24gb", "description": "deepnet 24gb"},
    ]
    monkeypatch.setattr(
        GATEWAY, "urlopen",
        lambda url, timeout=None: FakeHandle(tailnet_models),
    )
    found_peers = []

    def fake_discover(*args, **kwargs):
        found_peers.append((args, kwargs))
        return [{"host": "omarchy.ts.net", "port": 8091}]

    monkeypatch.setattr(tailnet, "discover_turbofit_peers", fake_discover)
    monkeypatch.delenv("TURBOFIT_PEERS", raising=False)

    # Automatic tailnet discovery: no config, peer probed over urlopen
    monkeypatch.setenv("TURBOFIT_PEER_DISCOVERY", "1")
    models = GATEWAY._discover_remote_models()
    assert found_peers, "auto-discovery was not consulted"
    assert [m["id"] for m in models] == ["omarchy.ts.net/quality-24gb"]
    assert models[0]["source"] == "remote"
    assert models[0]["metadata"]["mode"] == "tailnet"

    # Explicit TURBOFIT_PEERS wins and is used verbatim
    monkeypatch.setenv("TURBOFIT_PEERS", "klerik.ts.net:8091")
    models = GATEWAY._discover_remote_models()
    assert [m["id"] for m in models] == ["klerik.ts.net/quality-24gb"]

    # Discovery off + no explicit peers -> empty, never an error
    monkeypatch.delenv("TURBOFIT_PEERS", raising=False)
    monkeypatch.setenv("TURBOFIT_PEER_DISCOVERY", "0")
    assert GATEWAY._discover_remote_models() == []


def test_remote_discovery_cached_and_survives_peer_outage(monkeypatch) -> None:
    """Cache hits skip discovery; a failed refresh serves stale models."""
    import turbofit_runtime.tailnet as tailnet

    class FakeHandle:
        def __init__(self, models):
            self._models = models
        def read(self):
            return json.dumps({"data": self._models}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    GATEWAY._remote_cache.clear()
    calls = {"n": 0}

    def flaky_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("peer down")
        return FakeHandle([{"id": "quality-24gb"}])

    monkeypatch.setattr(GATEWAY, "urlopen", flaky_urlopen)
    monkeypatch.setattr(
        tailnet, "discover_turbofit_peers",
        lambda *a, **k: [{"host": "omarchy.ts.net", "port": 8091}],
    )
    monkeypatch.delenv("TURBOFIT_PEERS", raising=False)
    monkeypatch.setenv("TURBOFIT_PEER_DISCOVERY", "1")

    # First call discovers and caches.
    models = GATEWAY._discover_remote_models()
    assert [m["id"] for m in models] == ["omarchy.ts.net/quality-24gb"]

    # Within the TTL: cache hit, no new urlopen.
    assert GATEWAY._discover_remote_models()
    assert calls["n"] == 1

    # Simulate TTL expiry: next call re-discovers; the peer is down this
    # time, so the last known-good catalog must be served (stale-while-
    # revalidate) instead of dropping the model into a 400.
    monkeypatch.setattr(GATEWAY, "REMOTE_CACHE_TTL_S", 0.0)
    models = GATEWAY._discover_remote_models()
    assert [m["id"] for m in models] == ["omarchy.ts.net/quality-24gb"]
    assert calls["n"] == 2

    # A changed peer config re-discovers immediately (cache is keyed on
    # the config, not just time).
    monkeypatch.setattr(GATEWAY, "REMOTE_CACHE_TTL_S", 30.0)
    monkeypatch.setenv("TURBOFIT_PEERS", "klerik.ts.net:8091")
    models = GATEWAY._discover_remote_models()
    assert [m["id"] for m in models] == ["klerik.ts.net/quality-24gb"]
    assert calls["n"] == 3


def test_remote_discovery_does_not_relay_remote_entries(monkeypatch) -> None:
    """A peer's remote entries must never be re-prefixed (relay loop)."""
    class FakeHandle:
        def __init__(self, models):
            self._models = models
        def read(self):
            return json.dumps({"data": self._models}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    GATEWAY._remote_cache.clear()
    peer_catalog = [
        {"id": "quality-24gb", "source": "local"},
        # This entry was itself relayed by the peer from ANOTHER peer.
        # Re-prefixing it would create a nested "host/host/..." id that
        # grows a hop longer on every discovery pass.
        {"id": "omarchy.ts.net/qwen38-27b", "source": "remote"},
        {"id": "auto", "source": "auto"},
    ]
    monkeypatch.setattr(
        GATEWAY, "urlopen", lambda *a, **k: FakeHandle(peer_catalog))
    monkeypatch.delenv("TURBOFIT_PEER_DISCOVERY", raising=False)
    monkeypatch.setenv("TURBOFIT_PEERS", "omarchy.ts.net:8091")
    models = GATEWAY._discover_remote_models()
    assert [m["id"] for m in models] == ["omarchy.ts.net/quality-24gb"]


def test_a_named_model_is_never_silently_answered_by_a_hosted_api(monkeypatch) -> None:
    """Naming a local model is a promise; substituting one is invisible to the caller."""
    monkeypatch.setattr(GATEWAY, "ALLOW_API", True)

    assert GATEWAY.api_fallback_permitted(
        "omarchy.tail6ff78b.ts.net/qwen3-8-flash-next-strata", {"source": "local", "alias": "x"}
    ) is False
    assert GATEWAY.api_fallback_permitted("cyber-tiel-coder-35b-a3b", {"source": "local"}) is False
    # Router aliases promise "whatever is best here", so the API lane stays legal.
    assert GATEWAY.api_fallback_permitted("auto", {"source": "local"}) is True
    assert GATEWAY.api_fallback_permitted("active:aux", {"source": "local"}) is True
    # And an API backend never "falls back" to another API.
    assert GATEWAY.api_fallback_permitted("auto", {"source": "api", "is_api": True}) is False


def test_api_fallback_stays_off_when_the_operator_disabled_it(monkeypatch) -> None:
    monkeypatch.setattr(GATEWAY, "ALLOW_API", False)

    assert GATEWAY.api_fallback_permitted("auto", {"source": "local"}) is False
