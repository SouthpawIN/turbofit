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


def test_nous_api_auth_adds_hermes_source_for_standalone_systemd_service(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "hermes-agent"
    package = source / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "auth.py").write_text(
        "def resolve_nous_runtime_credentials(**kwargs):\n"
        "    return {'api_key': 'systemd-runtime-key'}\n"
    )
    monkeypatch.setattr(GATEWAY, "HERMES_SOURCE", str(source))
    monkeypatch.delitem(sys.modules, "hermes_cli", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.auth", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(source)])

    assert GATEWAY._get_api_key("nous") == "systemd-runtime-key"
    assert sys.path[0] == str(source)


def test_health_probe_accepts_native_runtime_health(monkeypatch) -> None:
    reply = SimpleNamespace(returncode=0, stdout='{"status":"ok"}')
    monkeypatch.setattr(GATEWAY.subprocess, "run", lambda *args, **kwargs: reply)

    assert GATEWAY.check_port(8092) is True


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


def test_api_fallback_accepts_ordered_model_chain(tmp_path, monkeypatch) -> None:
    preferences = tmp_path / "preferences.yaml"
    preferences.write_text(
        "api_fallback:\n"
        "  main:\n"
        "    - qwen/qwen3.5-flash-02-23\n"
        "    - z-ai/glm-5.2\n"
        "    - minimax/minimax-m3\n"
        "  base_url: https://inference-api.nousresearch.com/v1\n"
        "  provider: nous\n"
    )
    monkeypatch.setattr(GATEWAY, "PREFS", str(preferences))
    monkeypatch.setattr(GATEWAY, "HERMES_HOME", str(tmp_path / "hermes"))

    candidates = GATEWAY._api_fallback_candidates()

    assert [candidate["model_id"] for candidate in candidates] == [
        "qwen/qwen3.5-flash-02-23",
        "z-ai/glm-5.2",
        "minimax/minimax-m3",
    ]
    assert all(
        candidate["base_url"] == "https://inference-api.nousresearch.com"
        for candidate in candidates
    )


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

    assert ids == ["auto", "active:main", "active:aux"]


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
