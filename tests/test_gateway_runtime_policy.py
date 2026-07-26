from __future__ import annotations

import importlib.util
import json
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


def test_health_probe_accepts_turbohaul_manager_status(monkeypatch) -> None:
    replies = iter([
        SimpleNamespace(returncode=0, stdout='{"detail":"Not Found"}'),
        SimpleNamespace(returncode=0, stdout='{"detail":"Not Found"}'),
        SimpleNamespace(returncode=0, stdout='{"residents":[]}'),
    ])
    monkeypatch.setattr(GATEWAY.subprocess, "run", lambda *args, **kwargs: next(replies))

    assert GATEWAY.check_port(11401) is True


def test_turbohaul_health_is_model_specific(monkeypatch) -> None:
    status = {
        "loading": {"model_tag": "loading-model"},
        "residents": [{"model_tag": "resident-model"}],
        "queue": [],
    }
    reply = SimpleNamespace(returncode=0, stdout=json.dumps(status))
    monkeypatch.setattr(GATEWAY.subprocess, "run", lambda *args, **kwargs: reply)

    assert GATEWAY._turbohaul_model_state(11401, "resident-model") == "ready"
    assert GATEWAY._turbohaul_model_state(11401, "loading-model") == "loading"
    assert GATEWAY._turbohaul_model_state(11401, "absent-model") == "down"


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
