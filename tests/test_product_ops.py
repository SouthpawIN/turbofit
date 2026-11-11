from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import yaml

import product_ops


ROOT = Path(__file__).resolve().parents[1]


def test_shift_up_applies_smarter_ladder_neighbor(monkeypatch) -> None:
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "recommendation_snapshot",
        lambda preference=None: {
            "recommendations": {
                "intelligence": [
                    {"profile": "smart"},
                    {"profile": "mid"},
                    {"profile": "light"},
                ]
            }
        },
    )
    monkeypatch.setattr(product_ops, "_current_profile_id", lambda: "mid")
    applied = {}
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "select_profile",
        lambda profile: applied.setdefault("profile", profile) or {"configured": True, "profile_id": profile},
    )

    result = product_ops.shift_configuration("up")

    assert result["ok"] is True
    assert result["profile"] == "smart"
    assert applied["profile"] == "smart"


def test_shift_down_stops_at_lightest(monkeypatch) -> None:
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "recommendation_snapshot",
        lambda preference=None: {"recommendations": {"intelligence": [{"profile": "smart"}, {"profile": "light"}]}},
    )
    monkeypatch.setattr(product_ops, "_current_profile_id", lambda: "light")

    result = product_ops.shift_configuration("down")

    assert result["ok"] is False
    assert result["shifted"] is False
    assert "lightest" in result["error"]


def test_shift_model_picks_fitting_combination(monkeypatch) -> None:
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "combination_snapshot",
        lambda: {
            "combinations": [
                {"profile": "bonsai-lane", "main": "bonsai-27b", "main_name": "Bonsai 27B", "fit": True},
                {"profile": "gsq-rco-lane", "main": "qwen38-gsq-rco", "main_name": "Qwen 3.8 GSQ-RCO", "fit": True},
            ]
        },
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "recommendation_snapshot",
        lambda preference=None: {"recommendations": {"intelligence": [{"profile": "gsq-rco-lane"}, {"profile": "bonsai-lane"}]}},
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "select_profile",
        lambda profile: {"configured": True, "profile_id": profile},
    )

    result = product_ops.shift_configuration("bonsai")

    assert result["profile"] == "bonsai-lane"


def test_update_products_updates_plugin_desktop_and_turbosouth(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class Result:
        def __init__(self, code=0):
            self.returncode = code
            self.stdout = "updated"
            self.stderr = ""

    monkeypatch.setattr(product_ops.shutil, "which", lambda name: "/usr/bin/hermes")
    monkeypatch.setattr(
        product_ops.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "install_desktop_plugin",
        lambda **kwargs: {"installed": True, "plugin": "turbofit"},
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "install_turbosouth_profile",
        lambda **kwargs: {"installed": True, "profile": "turbosouth"},
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "ensure_recommended_models",
        lambda **kwargs: {"ok": True, "families": ["bonsai-27b"], "downloaded": 0, "verified": 1, "artifacts": []},
    )

    result = product_ops.update_products(hermes_home=tmp_path)

    assert result["ok"] is True
    assert calls[0][:3] == ["/usr/bin/hermes", "plugins", "update"]
    assert result["turbosouth"]["profile"] == "turbosouth"


def test_recommended_families_follow_auto_memory_bands() -> None:
    import plugin_tools

    assert plugin_tools.recommended_artifact_families(8 * 1024) == ["maple-preview-tq2", "ornith-1-5-35a3b"]
    assert plugin_tools.recommended_artifact_families(16 * 1024)[0] == "qwen3-8-27b-gsq-rco-iq3xxs"
    assert plugin_tools.recommended_artifact_families(24 * 1024)[0] == "qwen3-8-27b-gsq-rco-iq3s"
    assert plugin_tools.recommended_artifact_families(96 * 1024)[0] == "qwen3-8-27b-bf16"


def test_retire_archives_old_family_and_refuses_recommended(tmp_path, monkeypatch) -> None:
    import json
    import product_ops

    monkeypatch.setenv("TURBOFIT_MODEL_ROOT", str(tmp_path))
    old = tmp_path / "Bonsai-27B" / "old.gguf"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old-weights")
    monkeypatch.setattr(product_ops, "_artifact_rows", lambda: [
        {"destination": "Bonsai-27B/old.gguf", "families": ["bonsai-27b"]},
        {"destination": "Maple-Preview/maple-tq2_0.gguf", "families": ["maple-preview-tq2"]},
    ])
    monkeypatch.setattr(product_ops.plugin_tools, "recommended_artifact_families", lambda usable_memory_mb=None: ["maple-preview-tq2", "ornith-1-5-35a3b"])

    offer = product_ops.local_model_replacement()
    assert offer["offer"] is True
    assert offer["from_family"] == "bonsai-27b"
    assert offer["to_family"] == "maple-preview-tq2"

    archived = product_ops.retire_local_model("bonsai-27b", "archive")
    assert archived["retired"] is True
    assert not old.exists()
    assert (Path.home() / ".local/share/turbofit/archive/bonsai-27b/old.gguf").is_file() or archived["archived"]

    try:
        product_ops.retire_local_model("maple-preview-tq2", "delete")
        raise AssertionError("should refuse recommended")
    except ValueError as exc:
        assert "recommended" in str(exc)



def test_slash_update_and_shift_are_wired(monkeypatch) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "turbofit_shift_plugin",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    monkeypatch.setattr(plugin, "update_products", lambda: {"ok": True, "updated": True})
    monkeypatch.setattr(plugin, "shift_configuration", lambda target: {"ok": True, "profile": target})

    assert json.loads(plugin._slash_turbofit("update"))["updated"] is True
    assert json.loads(plugin._slash_turbofit("shift up"))["profile"] == "up"
    assert json.loads(plugin._slash_turbofit("shift bonsai"))["profile"] == "bonsai"
    monkeypatch.setattr(plugin, "serve_tailnet", lambda: {"ok": True, "served": True, "provider_base_url": "https://host.ts.net:9443/v1"})
    assert json.loads(plugin._slash_turbofit("serve"))["served"] is True
    monkeypatch.setattr(plugin, "smoke_local_runtime", lambda: {"ok": True, "promoted": False, "suite": "local-runtime-smoke-v1"})
    smoke = json.loads(plugin._slash_turbofit("smoke"))
    assert smoke["promoted"] is False
    assert smoke["suite"] == "local-runtime-smoke-v1"


def test_serve_tailnet_publishes_provider_url(monkeypatch, tmp_path) -> None:
    hermes_home = tmp_path / ".hermes"
    work = hermes_home / "profiles" / "work"
    personal = hermes_home / "profiles" / "personal"
    non_profiles = [hermes_home / "profiles" / name for name in ("desktop-plugins", ".deleted")]
    for directory in non_profiles:
        directory.mkdir(parents=True)
    originals = {}
    for home in (hermes_home, work, personal):
        home.mkdir(parents=True, exist_ok=True)
        originals[home] = {
            "model": {"provider": "local", "default": home.name},
            "fallback_providers": [{"provider": "other", "model": home.name}],
            "providers": {home.name: {"api": f"https://{home.name.lstrip('.')}.example"}},
        }
        (home / "config.yaml").write_text(yaml.safe_dump(originals[home]), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(work))
    tailnet_url = "https://box.tailnet.ts.net:9443/v1"

    monkeypatch.setattr(
        product_ops.plugin_tools,
        "publish_tailnet",
        lambda **kwargs: {
            "ok": True,
            "dns_name": "box.tailnet.ts.net",
            "provider_base_url": tailnet_url,
            "dashboard_url": "https://box.tailnet.ts.net:9444/",
        },
    )
    # Emulate the old path's load of the active profile when reproducing the regression.
    hermes_cli = ModuleType("hermes_cli")
    config = ModuleType("hermes_cli.config")
    config.load_config = lambda: yaml.safe_load((work / "config.yaml").read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)

    result = product_ops.serve_tailnet()

    assert result["served"] is True
    assert result["provider_base_url"] == tailnet_url
    for home, original in originals.items():
        payload = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert payload["providers"].pop("turbofit")["api"] == tailnet_url
        assert payload == original
    for directory in non_profiles:
        assert not (directory / "config.yaml").exists()

