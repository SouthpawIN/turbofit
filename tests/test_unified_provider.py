from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "turbofit_gateway",
    Path(__file__).resolve().parents[1] / "scripts/turbofit-gateway.py",
)
assert SPEC and SPEC.loader
GATEWAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATEWAY)


def _profiles(tmp_path: Path) -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {
            "grm-carwin-262k": {"context": 262144, "description": "fast pair"},
            "grm-carwin-1m": {"context": 1048576, "description": "long pair"},
        },
    }))
    return path


def test_unified_catalog_uses_stable_profile_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    models = GATEWAY.provider_models()
    assert [item["id"] for item in models] == [
        "auto",
        "active:main",
        "active:aux",
        "grm-carwin-262k",
        "grm-carwin-1m",
    ]
    assert all(item["owned_by"] == "turbofit" for item in models)
    assert models[3]["context_length"] == 262144


def test_universal_model_strings_encode_role_without_a_second_provider():
    assert GATEWAY.parse_provider_model("auto") == ("auto", "main")
    assert GATEWAY.parse_provider_model("auto:aux") == ("auto", "aux")
    assert GATEWAY.parse_provider_model("active:main") == ("active", "main")
    assert GATEWAY.parse_provider_model("active:aux") == ("active", "aux")
    assert GATEWAY.parse_provider_model("grm-carwin-262k") == ("grm-carwin-262k", "main")
    assert GATEWAY.parse_provider_model("grm-carwin-262k:aux") == ("grm-carwin-262k", "aux")


def test_manual_selection_rejects_unknown_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    assert GATEWAY.resolve_requested_profile("not-in-catalog") is None


def test_auto_reuses_active_profile_without_rerunning_hardware_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-1m")
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: (_ for _ in ()).throw(AssertionError("must not recommend while a profile is active")))
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("auto") == "grm-carwin-1m"
    assert calls == []


def test_auto_without_active_profile_uses_hardware_recommender(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: None)
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: "grm-carwin-262k")
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("auto") == "grm-carwin-262k"
    assert calls == ["grm-carwin-262k"]


def test_manual_selection_uses_exact_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(GATEWAY, "PROFILES", str(_profiles(tmp_path)))
    calls = []
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-1m")
    monkeypatch.setattr(GATEWAY, "activate_profile", lambda profile: calls.append(profile) or True)

    assert GATEWAY.resolve_requested_profile("grm-carwin-1m") == "grm-carwin-1m"
    assert calls == []


def test_aux_auto_follows_active_pair_without_rerunning_recommendation(monkeypatch):
    monkeypatch.setattr(GATEWAY, "active_profile", lambda: "grm-carwin-262k")
    monkeypatch.setattr(GATEWAY, "recommend_profile", lambda: (_ for _ in ()).throw(AssertionError("must not recommend")))
    assert GATEWAY.resolve_requested_profile("auto:aux") == "grm-carwin-262k"
