from __future__ import annotations

import json
from pathlib import Path

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
                {"profile": "unleashed-lane", "main": "qwen38-unleashed", "main_name": "Qwen 3.8 Unleashed", "fit": True},
            ]
        },
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "recommendation_snapshot",
        lambda preference=None: {"recommendations": {"intelligence": [{"profile": "unleashed-lane"}, {"profile": "bonsai-lane"}]}},
    )
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "select_profile",
        lambda profile: {"configured": True, "profile_id": profile},
    )

    result = product_ops.shift_configuration("bonsai")

    assert result["profile"] == "bonsai-lane"


def test_update_products_updates_plugin_desktop_and_sirvir(monkeypatch, tmp_path: Path) -> None:
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
        "install_sirvir_profile",
        lambda **kwargs: {"installed": True, "profile": "sirvir"},
    )

    result = product_ops.update_products(hermes_home=tmp_path)

    assert result["ok"] is True
    assert calls[0][:3] == ["/usr/bin/hermes", "plugins", "update"]
    assert result["sirvir"]["profile"] == "sirvir"


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


def test_serve_tailnet_publishes_provider_url(monkeypatch) -> None:
    monkeypatch.setattr(
        product_ops.plugin_tools,
        "publish_tailnet",
        lambda **kwargs: {
            "ok": True,
            "dns_name": "box.tailnet.ts.net",
            "provider_base_url": "https://box.tailnet.ts.net:9443/v1",
            "dashboard_url": "https://box.tailnet.ts.net:9444/",
        },
    )
    saved = {}

    class Lock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(product_ops.plugin_tools, "_CONFIG_LOCK", Lock())
    monkeypatch.setattr(product_ops.plugin_tools, "configure_hermes", lambda config, **kwargs: saved.setdefault("url", kwargs.get("base_url")) or config)
    import types
    hermes_cli = types.ModuleType("hermes_cli")
    config = types.ModuleType("hermes_cli.config")
    config.load_config = lambda: {}
    config.save_config = lambda *a, **k: saved.setdefault("saved", True)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.config", config)

    result = product_ops.serve_tailnet()

    assert result["served"] is True
    assert result["provider_base_url"].endswith(":9443/v1")
    assert saved["url"] == "https://box.tailnet.ts.net:9443/v1"
    assert saved["saved"] is True

