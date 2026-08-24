from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_installed_plugin_imports_without_project_pythonpath() -> None:
    code = f"""
import importlib.util
import sys
from pathlib import Path
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location(
    'turbofit', root / '__init__.py', submodule_search_locations=[str(root)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert callable(module.register)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


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
    assert configured["custom_providers"] == original["custom_providers"]
    assert configured["providers"]["turbofit"] == {
        "name": "TurboFit",
        "api": "http://127.0.0.1:8091/v1",
        "api_key": "not-needed",
        "transport": "chat_completions",
        "default_model": "auto",
        "models": {"auto": {}, "active:main": {}, "active:aux": {}},
    }
    assert configured["model"]["provider"] == "custom:turbofit"
    assert configured["model"]["default"] == "auto"
    assert original["custom_providers"][-1]["name"] == "other"


def test_configure_hermes_writes_matching_main_aux_context() -> None:
    from plugin_tools import configure_hermes

    configured = configure_hermes(
        {},
        primary=True,
        fallback=False,
        base_url="http://127.0.0.1:8091/v1",
        context_length=131072,
    )

    models = configured["providers"]["turbofit"]["models"]
    assert models["auto"]["context_length"] == 131072
    assert models["active:main"]["context_length"] == 131072
    assert models["active:aux"]["context_length"] == 131072
    assert configured["model"]["context_length"] == 131072
    assert configured["auxiliary"]["compression"] == {
        "provider": "custom:turbofit",
        "model": "active:aux",
        "context_length": 131072,
    }


def test_configure_hermes_rebinds_turbofit_compression_to_main_when_not_primary() -> None:
    from plugin_tools import configure_hermes

    configured = configure_hermes(
        {
            "model": {"provider": "xai-oauth", "default": "grok-4.6", "context_length": 500000},
            "auxiliary": {
                "compression": {
                    "provider": "custom:turbofit",
                    "model": "active:aux",
                    "base_url": "http://127.0.0.1:8091/v1",
                }
            },
        },
        primary=False,
        fallback=True,
        base_url="http://127.0.0.1:8091/v1",
    )

    assert configured["auxiliary"]["compression"]["provider"] == "xai-oauth"
    assert configured["auxiliary"]["compression"]["model"] == "grok-4.6"
    assert configured["auxiliary"]["compression"]["context_length"] == 500000
    assert "base_url" not in configured["auxiliary"]["compression"]


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
        },
    ]

    disabled = configure_hermes(enabled_again, primary=False, fallback=False)
    assert disabled["fallback_providers"] == [
        {"provider": "nous", "model": "some-cloud-model"}
    ]


def test_configure_hermes_accepts_an_explicit_user_fallback_chain() -> None:
    from plugin_tools import configure_hermes

    chain = [
        {"provider": "custom:turbofit", "model": "auto"},
        {"provider": "nous", "model": "deepseek-ai/DeepSeek-V3"},
    ]

    configured = configure_hermes({}, fallback_chain=chain)

    assert configured["fallback_providers"] == chain
    assert configured["fallback_providers"] is not chain


def test_configure_hermes_rejects_secrets_in_user_fallback_chain() -> None:
    import pytest
    from plugin_tools import configure_hermes

    with pytest.raises(ValueError, match="provider and model"):
        configure_hermes({}, fallback_chain=[{
            "provider": "nous", "model": "model", "api_key": "secret",
        }])


def test_plain_http_provider_url_is_limited_to_loopback_or_tailnet() -> None:
    import pytest

    from plugin_tools import configure_hermes

    with pytest.raises(ValueError, match="Tailscale"):
        configure_hermes({}, base_url="http://example.com/v1")
    with pytest.raises(ValueError, match="Tailscale"):
        configure_hermes({}, base_url="http://8.8.8.8:8091/v1")

    assert configure_hermes({}, base_url="http://192.168.1.101:8091/v1")[
        "providers"
    ]["turbofit"]["api"] == "http://192.168.1.101:8091/v1"
    assert configure_hermes({}, base_url="http://100.100.10.20:8091/v1")[
        "providers"
    ]["turbofit"]["api"] == "http://100.100.10.20:8091/v1"


def test_status_includes_tailnet_publication_state(monkeypatch) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

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
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

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
    assert saved[0]["providers"]["turbofit"]["api"] == "https://host.example.ts.net:9443/v1"
    chain = saved[0]["fallback_providers"]
    assert {"provider": "custom:turbofit", "model": "auto"} in chain
    assert {"provider": "nous", "model": "stealth/ox-alpha"} in chain
    assert {"provider": "nous", "model": "stepfun/step-3.7-flash:free"} in chain
    assert {"provider": "nous", "model": "tencent/hy3:free"} in chain
    assert {"provider": "nous", "model": "poolside/laguna-s-2.1:free"} in chain
    assert {"provider": "nous", "model": "poolside/laguna-xs-2.1:free"} in chain
    assert not any("hermes" in str(item.get("model", "")).lower() for item in chain)
    assert not any("nvidia" in str(item).lower() or "nim" in str(item).lower() for item in chain)


def test_install_sirvir_profile_uses_current_github_distribution(monkeypatch, tmp_path: Path) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    calls = []
    profile = tmp_path / "profiles" / "sirvir"
    monkeypatch.setattr(plugin_tools.shutil, "which", lambda name: "/usr/bin/hermes" if name == "hermes" else None)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        profile.mkdir(parents=True)
        (profile / "distribution.yaml").write_text("name: sirvir\nversion: 2.1.0\n")
        return types.SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr(plugin_tools.subprocess, "run", fake_run)

    result = plugin_tools.install_sirvir_profile(hermes_home=tmp_path)

    assert calls[0][0] == [
        "/usr/bin/hermes", "profile", "install", "https://github.com/SouthpawIN/sirvir.git",
        "--name", "sirvir", "--yes",
    ]
    assert calls[0][1]["env"]["HERMES_HOME"] == str(tmp_path)
    assert result["source"] == "https://github.com/SouthpawIN/sirvir.git"
    assert result["updated"] is False
    assert result["version"] == "2.1.0"


def test_install_sirvir_profile_updates_github_distribution_and_preserves_user_state(monkeypatch, tmp_path: Path) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    calls = []
    profile = tmp_path / "profiles" / "sirvir"
    profile.mkdir(parents=True)
    (profile / "distribution.yaml").write_text("name: sirvir\nversion: 2.0.0\n")
    user_memory = profile / "memories" / "USER.md"
    user_memory.parent.mkdir()
    user_memory.write_text("keep me")
    monkeypatch.setattr(plugin_tools.shutil, "which", lambda name: "/usr/bin/hermes" if name == "hermes" else None)

    def fake_run(command, **kwargs):
        calls.append(command)
        (profile / "distribution.yaml").write_text("name: sirvir\nversion: 2.1.0\n")
        return types.SimpleNamespace(returncode=0, stdout="updated", stderr="")

    monkeypatch.setattr(plugin_tools.subprocess, "run", fake_run)

    result = plugin_tools.install_sirvir_profile(hermes_home=tmp_path)

    assert calls == [["/usr/bin/hermes", "profile", "update", "sirvir", "--yes"]]
    assert result["updated"] is True
    assert result["version"] == "2.1.0"
    assert user_memory.read_text() == "keep me"


def test_readme_prominently_includes_sirvir_and_reciprocal_install() -> None:
    text = (ROOT / "README.md").read_text()

    assert "## Sirvir" in text
    assert "https://github.com/SouthpawIN/sirvir" in text
    assert "Install Sirvir" in text
    assert "Sirvir installs Turbofit when it is missing" in text


def test_install_desktop_plugin_copies_native_desktop_surface(tmp_path: Path) -> None:
    from plugin_tools import install_desktop_plugin

    result = install_desktop_plugin(hermes_home=tmp_path)
    installed = tmp_path / "desktop-plugins" / "turbofit" / "plugin.js"

    assert result["installed"] is True
    assert installed.is_file()
    text = installed.read_text()
    assert "ROUTES_AREA" in text
    assert "SIDEBAR_NAV_AREA" in text
    assert "api.rest('/configure'" in text


def test_desktop_plugin_source_has_status_recommendation_and_fallback_controls() -> None:
    text = (ROOT / "desktop" / "plugin.js").read_text()

    assert "api.rest('/status'" in text
    assert "api.rest('/combinations'" in text
    assert "Manual —" in text
    assert "Main model" in text
    assert "Auxiliary model" in text
    assert "Context length" in text
    assert "api.rest(`/recommendations?" in text
    assert "fallback_chain" in text
    assert "publish_tailnet" in text
    assert "install_sirvir" in text
    assert "install_native" in text
    assert "install_freetoken" in text
    assert "install_lemonade" in text
    assert "Dashboard Tailnet HTTPS port" in text
    assert "Provider Tailnet HTTPS port" in text
    assert "ScoreBar" in text
    assert "api.rest('/shift'" in text
    assert "api.rest('/update'" in text
    assert "Shift up" in text
    assert "Serve on Tailscale" in text
    assert "api.rest('/serve'" in text


def test_apply_configuration_can_install_bundled_sirvir(monkeypatch) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

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


def test_apply_configuration_can_install_lemonade_runtime(monkeypatch) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    hermes_package = types.ModuleType("hermes_cli")
    hermes_config = types.ModuleType("hermes_cli.config")
    setattr(hermes_config, "load_config", lambda: {})
    setattr(hermes_config, "save_config", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setattr(plugin_tools, "install_lemonade_runtime", lambda: {
        "status": "verified", "version": "11.5.1"
    })

    result = plugin_tools.apply_configuration(
        primary=False,
        fallback=None,
        profile=None,
        base_url="http://127.0.0.1:13305/api/v1",
        install_lemonade=True,
    )

    assert result["lemonade"] == {"status": "verified", "version": "11.5.1"}


def test_apply_configuration_can_install_native_runtime(monkeypatch) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    hermes_package = types.ModuleType("hermes_cli")
    hermes_config = types.ModuleType("hermes_cli.config")
    setattr(hermes_config, "load_config", lambda: {})
    setattr(hermes_config, "save_config", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setattr(
        plugin_tools, "install_native_runtime",
        lambda: {"status": "verified", "backend": "cuda"},
    )

    result = plugin_tools.apply_configuration(
        primary=False, fallback=None, profile=None, base_url=None, install_native=True,
    )

    assert result["native_runtime"] == {"status": "verified", "backend": "cuda"}


def test_apply_configuration_can_install_freetoken_candidate(monkeypatch) -> None:
    import types
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    hermes_package = types.ModuleType("hermes_cli")
    hermes_config = types.ModuleType("hermes_cli.config")
    setattr(hermes_config, "load_config", lambda: {})
    setattr(hermes_config, "save_config", lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", hermes_config)
    monkeypatch.setattr(
        plugin_tools,
        "install_freetoken_runtime",
        lambda: {
            "status": "verified-candidate",
            "version": "0.1.2",
            "auto_promote": False,
        },
    )

    result = plugin_tools.apply_configuration(
        primary=False,
        fallback=None,
        profile=None,
        base_url=None,
        install_freetoken=True,
    )

    assert result["freetoken_runtime"] == {
        "status": "verified-candidate",
        "version": "0.1.2",
        "auto_promote": False,
    }


def test_handle_configure_rejects_string_booleans_before_side_effects(monkeypatch) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

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

        def register_command(self, name, handler, description, args_hint=""):
            self.commands.append((name, handler, description, args_hint))

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


def test_slash_commands_are_enabled_in_sirvir_and_default_homes(tmp_path: Path) -> None:
    import plugin_tools
    import yaml

    default = tmp_path / ".hermes"
    sirvir = default / "profiles" / "sirvir"
    (default / "plugins" / "turbofit").mkdir(parents=True)
    sirvir.mkdir(parents=True)
    (sirvir / "config.yaml").write_text("model:\n  provider: custom:turbofit\n", encoding="utf-8")
    (default / "plugins" / "turbofit" / "plugin.yaml").write_text("name: turbofit\n", encoding="utf-8")

    result = plugin_tools.activate_slash_commands(hermes_home=sirvir, plugin_root=default / "plugins" / "turbofit")

    default_config = yaml.safe_load((default / "config.yaml").read_text(encoding="utf-8"))
    sirvir_config = yaml.safe_load((sirvir / "config.yaml").read_text(encoding="utf-8"))
    assert "turbofit" in default_config["plugins"]["enabled"]
    assert "turbofit" in sirvir_config["plugins"]["enabled"]
    assert (sirvir / "plugins" / "turbofit" / "plugin.yaml").is_file()
    assert (sirvir / "skills" / "turbofit" / "SKILL.md").is_file()
    assert result["homes"] >= 2


def test_combination_snapshot_requests_portable_fit_lanes(monkeypatch) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    calls = []

    class Result:
        stdout = "[]"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(
        plugin_tools.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or Result(),
    )

    payload = plugin_tools.combination_snapshot()

    assert payload["ok"] is True
    assert "--evidence-scope" in calls[0]
    assert calls[0][calls[0].index("--evidence-scope") + 1] == "portable-fit"


def test_recommendation_snapshot_exposes_unmeasured_compatible_lanes(monkeypatch) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    calls = []

    class Result:
        stderr = ""

        def __init__(self, command):
            self.returncode = 0 if "portable-fit" in command else 1
            self.stdout = (
                '[{"profile":"binary-bonsai-27b-q1-0-auto-64k",'
                '"fit":true,"validation_required":true,"min_tps":0}]'
                if self.returncode == 0 else "[]"
            )

    monkeypatch.setattr(
        plugin_tools.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command) or Result(command),
    )

    payload = plugin_tools.recommendation_snapshot("balanced")

    assert payload["recommendations"]["balanced"] == []
    assert payload["compatible_lanes"][0]["validation_required"] is True
    assert payload["ok"] is True
    assert any("portable-fit" in command for command in calls)


def test_slash_turbofit_rescans_hardware_and_returns_multiple_preferences(monkeypatch) -> None:
    plugin = _load_plugin_module()
    monkeypatch.setattr(plugin, "recommendation_snapshot", lambda preference=None: {
        "ok": True,
        "hardware": {"os": "linux", "system_ram_mb": 128000},
        "recommendations": {
            "intelligence": [{"profile": "quality"}],
            "balanced": [{"profile": "balanced"}],
            "speed": [{"profile": "fast"}],
        },
        "requested_preference": preference,
        "preferences": ["intelligence", "balanced", "speed"],
    })

    payload = json.loads(plugin._slash_turbofit(""))

    assert payload["ok"] is True
    assert payload["preferences"] == ["intelligence", "balanced", "speed"]
    assert payload["hardware"]["system_ram_mb"] == 128000


def test_slash_turbofit_tiers_returns_all_hardware_levels(monkeypatch) -> None:
    plugin = _load_plugin_module()
    monkeypatch.setattr(plugin, "hardware_tier_snapshot", lambda: {
        "current_hardware": {"native_tier_gb": 48},
        "tiers": [{"capacity_gb": value} for value in (8, 16, 24, 48, 64, 96, 200, 300)],
    })

    payload = json.loads(plugin._slash_turbofit("tiers"))

    assert payload["current_hardware"]["native_tier_gb"] == 48
    assert [item["capacity_gb"] for item in payload["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]


def test_slash_turbofit_setup_launches_the_setup_screen(monkeypatch) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "turbofit",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    plugin = importlib.util.module_from_spec(spec)
    sys.modules["turbofit"] = plugin
    spec.loader.exec_module(plugin)
    monkeypatch.setattr(plugin, "launch_setup_screen", lambda: {"launched": True, "url": "http://127.0.0.1:9119/"})

    payload = json.loads(plugin._slash_turbofit("setup"))

    assert payload["setup"]["launched"] is True


def test_launch_setup_screen_refreshes_desktop_surface(monkeypatch, tmp_path: Path) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    monkeypatch.setattr(plugin_tools, "install_desktop_plugin", lambda **kwargs: {"installed": True, "path": str(tmp_path)})
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **kwargs: {"ok": True, "families": ["bonsai-27b"], "downloaded": 0, "verified": 1, "artifacts": []})

    result = plugin_tools.launch_setup_screen()

    assert result["launched"] is True
    assert result["surface"] == "desktop"
    assert result["path"] == "/turbofit"


def test_dashboard_contract_is_installable() -> None:
    manifest = json.loads((ROOT / "dashboard" / "manifest.json").read_text())
    assert manifest["name"] == "turbofit"
    assert manifest["tab"]["path"] == "/turbofit"
    assert manifest["api"] == "plugin_api.py"
    assert (ROOT / "dashboard" / manifest["entry"]).is_file()
    assert (ROOT / "dashboard" / manifest["css"]).is_file()


def test_dashboard_exposes_physical_hardware_tournaments() -> None:
    spec = importlib.util.spec_from_file_location("turbofit_dashboard_api", ROOT / "dashboard/plugin_api.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._tournament_rows()

    assert payload["ok"] is True
    assert [item["vram_gb"] for item in payload["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    assert all(item["physical_evidence_required"] is True for item in payload["tiers"])
    assert any("QWEN3-8" in candidate["configuration"].upper() for item in payload["tiers"] for candidate in item["candidates"])
    assert "/api/plugins/turbofit/tournaments" in (ROOT / "dashboard/dist/index.js").read_text()


def test_dashboard_exposes_speed_and_intelligence_hardware_tiers() -> None:
    spec = importlib.util.spec_from_file_location("turbofit_dashboard_tier_api", ROOT / "dashboard/plugin_api.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module.hardware_tier_snapshot()

    assert [item["capacity_gb"] for item in payload["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    bundle = (ROOT / "dashboard/dist/index.js").read_text()
    assert "/api/plugins/turbofit/hardware-tiers" in bundle
    assert "speed versus intelligence" in bundle


def test_dashboard_exposes_auxiliary_recommendations_by_tier() -> None:
    spec = importlib.util.spec_from_file_location("turbofit_dashboard_aux_api", ROOT / "dashboard/plugin_api.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = module._auxiliary_tiers()

    assert payload["ok"] is True
    assert [item["vram_gb"] for item in payload["tiers"]] == [8, 16, 24, 48, 64, 96, 200, 300]
    assert payload["tiers"][3]["status"] == "requires-current-recipe-validation"
    assert payload["tiers"][3]["best_auxiliary"] is None
    bundle = (ROOT / "dashboard/dist/index.js").read_text()
    assert "/api/plugins/turbofit/auxiliary-tiers" in bundle
    assert "Auxiliary candidates by hardware tier" in bundle


def test_sirvir_references_are_customer_service_not_autonomous_manager() -> None:
    for path in (
        ROOT / "references" / "SOUL.md",
        ROOT / "skills" / "turbofit" / "references" / "SOUL.md",
    ):
        text = path.read_text().lower()
        assert "customer service" in text
        assert "autonomous model lifecycle manager" not in text


def test_dashboard_exposes_bundled_sirvir_install_option() -> None:
    bundle = (ROOT / "dashboard" / "dist" / "index.js").read_text()

    assert 'install_sirvir: installSirvir' in bundle
    assert "Install or update GitHub-current Sirvir customer service" in bundle


def test_plugin_manifest_declares_registered_tools() -> None:
    import yaml

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert manifest["kind"] == "standalone"
    assert set(manifest["provides_tools"]) == {
        "turbofit_status",
        "turbofit_configure",
    }


def test_plugin_loads_without_developer_pythonpath(monkeypatch) -> None:
    """The installed Hermes plugin must bootstrap its own src-layout."""
    for entry in tuple(sys.path):
        if Path(entry or ".").resolve() == (ROOT / "src").resolve():
            monkeypatch.setattr(
                sys,
                "path",
                [item for item in sys.path if Path(item or ".").resolve() != (ROOT / "src").resolve()],
            )
            break
    sys.modules.pop("turbofit_runtime", None)

    plugin = _load_plugin_module()

    assert plugin.check_available() is True


def test_select_profile_uses_running_hermes_python(monkeypatch) -> None:
    import plugin_tools
    monkeypatch.setattr(plugin_tools, "ensure_recommended_models", lambda **_: {"ok": True, "families": ["bonsai-27b"], "artifacts": []})

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = '{"configured": true}'
        stderr = ""

    def run(command, **_kwargs):
        seen.append(command)
        return Result()

    monkeypatch.setattr(plugin_tools.subprocess, "run", run)

    assert plugin_tools.select_profile("auto")["configured"] is True
    assert seen == [[sys.executable, str(ROOT / "scripts" / "turbofit-runtime"), "set", "auto"]]
