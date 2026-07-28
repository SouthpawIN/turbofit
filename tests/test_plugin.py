from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "turbofit_plugin",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_configure_hermes_registers_named_provider_and_primary_model() -> None:
    from plugin_tools import configure_hermes

    original = {
        "custom_providers": [
            {
                "name": "other",
                "base_url": "http://example.test/v1",
                "api_key": "secret",
                "models": {"model-a": {}},
            }
        ],
        "model": {"provider": "nous", "default": "some-model"},
    }

    configured = configure_hermes(
        original,
        primary=True,
        fallback=False,
        base_url="http://127.0.0.1:8091/v1",
    )

    assert configured is not original
    assert configured["custom_providers"][0] == original["custom_providers"][0]
    assert configured["custom_providers"][1] == {
        "name": "turbofit",
        "base_url": "http://127.0.0.1:8091/v1",
        "api_key": "not-needed",
        "api_mode": "chat_completions",
        "models": {"auto": {}, "active:main": {}, "active:aux": {}},
    }
    assert configured["model"]["provider"] == "custom:turbofit"
    assert configured["model"]["default"] == "auto"
    assert original["custom_providers"][-1]["name"] == "other"


def test_configure_hermes_fallback_is_idempotent_and_removable() -> None:
    from plugin_tools import configure_hermes

    config = {
        "fallback_providers": [
            {"provider": "nous", "model": "some-cloud-model"},
            {"provider": "custom:turbofit", "model": "old"},
        ]
    }
    enabled = configure_hermes(config, primary=False, fallback=True)
    enabled_again = configure_hermes(enabled, primary=False, fallback=True)

    assert enabled_again["fallback_providers"] == [
        {"provider": "nous", "model": "some-cloud-model"},
        {
            "provider": "custom:turbofit",
            "model": "auto",
            "base_url": "http://127.0.0.1:8091/v1",
            "api_mode": "chat_completions",
        },
    ]

    disabled = configure_hermes(enabled_again, primary=False, fallback=False)
    assert disabled["fallback_providers"] == [
        {"provider": "nous", "model": "some-cloud-model"}
    ]


def test_plain_http_provider_url_is_limited_to_loopback_or_tailnet() -> None:
    import pytest

    from plugin_tools import configure_hermes

    with pytest.raises(ValueError, match="Tailscale"):
        configure_hermes({}, base_url="http://example.com/v1")
    with pytest.raises(ValueError, match="Tailscale"):
        configure_hermes({}, base_url="http://172.20.0.1:8091/v1")

    assert configure_hermes({}, base_url="http://100.100.10.20:8091/v1")[
        "custom_providers"
    ][0]["base_url"] == "http://100.100.10.20:8091/v1"


def test_status_includes_tailnet_publication_state(monkeypatch) -> None:
    import plugin_tools

    monkeypatch.setattr(plugin_tools, "tailnet_status", lambda: {
        "available": True,
        "connected": True,
        "dns_name": "host.example.ts.net",
        "serve": {},
        "error": None,
    })

    status = plugin_tools.status_snapshot({}, probe=False)

    assert status["tailnet"]["connected"] is True
    assert status["tailnet"]["dns_name"] == "host.example.ts.net"


def test_apply_configuration_can_publish_tailnet_and_use_remote_provider_url(monkeypatch) -> None:
    import types
    import plugin_tools

    saved: list[dict] = []
    hermes_package = types.ModuleType("hermes_cli")
    hermes_config = types.ModuleType("hermes_cli.config")
    setattr(hermes_config, "load_config", lambda: {})
    setattr(hermes_config, "save_config", lambda config, merge_existing=False: saved.append(config))
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setattr(plugin_tools, "publish_tailnet", lambda **_: {
        "ok": True,
        "dashboard_url": "https://host.example.ts.net:9444/",
        "provider_base_url": "https://host.example.ts.net:9443/v1",
    })

    result = plugin_tools.apply_configuration(
        primary=True,
        fallback=True,
        profile=None,
        base_url=None,
        publish_tailnet_routes=True,
    )

    assert result["tailnet"]["dashboard_url"] == "https://host.example.ts.net:9444/"
    assert saved[0]["custom_providers"][0]["base_url"] == "https://host.example.ts.net:9443/v1"
    assert saved[0]["fallback_providers"][0]["base_url"] == "https://host.example.ts.net:9443/v1"


def test_install_sirvir_profile_copies_bundled_customer_service_profile(tmp_path: Path) -> None:
    from plugin_tools import install_sirvir_profile

    result = install_sirvir_profile(hermes_home=tmp_path)
    profile = tmp_path / "profiles" / "sirvir"

    assert result == {
        "installed": True,
        "updated": False,
        "profile": "sirvir",
        "path": str(profile),
    }
    assert "customer service" in (profile / "SOUL.md").read_text().lower()
    assert "pull request" in (profile / "AGENTS.md").read_text().lower()
    assert (profile / "config.yaml").is_file()


def test_install_sirvir_profile_updates_only_distribution_owned_files(tmp_path: Path) -> None:
    from plugin_tools import install_sirvir_profile

    install_sirvir_profile(hermes_home=tmp_path)
    profile = tmp_path / "profiles" / "sirvir"
    user_memory = profile / "memories" / "USER.md"
    user_memory.parent.mkdir()
    user_memory.write_text("keep me")

    result = install_sirvir_profile(hermes_home=tmp_path)

    assert result["installed"] is True
    assert result["updated"] is True
    assert user_memory.read_text() == "keep me"


def test_apply_configuration_can_install_bundled_sirvir(monkeypatch) -> None:
    import types
    import plugin_tools

    hermes_package = types.ModuleType("hermes_cli")
    hermes_config = types.ModuleType("hermes_cli.config")
    setattr(hermes_config, "load_config", lambda: {})
    setattr(hermes_config, "save_config", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setattr(plugin_tools, "install_sirvir_profile", lambda: {
        "installed": True, "updated": False, "profile": "sirvir", "path": "/profiles/sirvir"
    })

    result = plugin_tools.apply_configuration(
        primary=False,
        fallback=None,
        profile=None,
        base_url=None,
        install_sirvir=True,
    )

    assert result["sirvir"]["installed"] is True


def test_handle_configure_rejects_string_booleans_before_side_effects(monkeypatch) -> None:
    import plugin_tools

    called = False

    def apply(**_):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(plugin_tools, "apply_configuration", apply)
    result = json.loads(plugin_tools.handle_configure({"primary": "false", "publish_tailnet": "false"}))

    assert result["ok"] is False
    assert "boolean" in result["error"]
    assert called is False


def test_plugin_registers_status_configure_and_slash_command() -> None:
    plugin = _load_plugin_module()

    class Context:
        def __init__(self) -> None:
            self.tools = []
            self.commands = []
            self.skills = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, name, handler, description):
            self.commands.append((name, handler, description))

        def register_skill(self, name, path):
            self.skills.append((name, Path(path)))

    ctx = Context()
    plugin.register(ctx)

    assert {item["name"] for item in ctx.tools} == {
        "turbofit_status",
        "turbofit_configure",
    }
    assert all(item["toolset"] == "turbofit" for item in ctx.tools)
    assert [item[0] for item in ctx.commands] == ["turbofit"]
    assert ctx.skills == [("turbofit", ROOT / "SKILL.md")]


def test_dashboard_contract_is_installable() -> None:
    manifest = json.loads((ROOT / "dashboard" / "manifest.json").read_text())
    assert manifest["name"] == "turbofit"
    assert manifest["tab"]["path"] == "/turbofit"
    assert manifest["api"] == "plugin_api.py"
    assert (ROOT / "dashboard" / manifest["entry"]).is_file()
    assert (ROOT / "dashboard" / manifest["css"]).is_file()


def test_sirvir_sources_are_customer_service_not_autonomous_manager() -> None:
    for path in (
        ROOT / "profiles" / "sirvir" / "SOUL.md",
        ROOT / "references" / "SOUL.md",
        ROOT / "skills" / "turbofit" / "references" / "SOUL.md",
    ):
        text = path.read_text().lower()
        assert "customer service" in text
        assert "autonomous model lifecycle manager" not in text


def test_dashboard_exposes_bundled_sirvir_install_option() -> None:
    bundle = (ROOT / "dashboard" / "dist" / "index.js").read_text()

    assert 'install_sirvir: installSirvir' in bundle
    assert "Install Sirvir customer service profile" in bundle


def test_plugin_manifest_declares_registered_tools() -> None:
    import yaml

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert manifest["kind"] == "standalone"
    assert set(manifest["provides_tools"]) == {
        "turbofit_status",
        "turbofit_configure",
    }
