from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SIRVIR = ROOT / "profiles" / "sirvir"


def test_sirvir_is_local_only_customer_service_with_pr_suggestions() -> None:
    soul = (SIRVIR / "SOUL.md").read_text()
    guide = (SIRVIR / "AGENTS.md").read_text()
    readme = (SIRVIR / "README.md").read_text()
    combined = "\n".join((soul, guide, readme)).lower()

    assert "install, configure, use, and troubleshoot" in combined
    assert "pull-request suggestion" in combined
    assert "local fallback ladder" in combined
    assert "minimum local floor" in combined
    assert "fail closed" in combined
    assert "cloud" not in combined
    assert "api fallback" not in combined
    assert "remote model" not in combined


def test_sirvir_uses_only_the_local_turbofit_provider() -> None:
    config = yaml.safe_load((SIRVIR / "config.yaml").read_text())

    assert config["model"] == {
        "provider": "custom:turbofit",
        "default": "auto",
    }
    assert set(config["providers"]) == {"turbofit"}
    provider = config["providers"]["turbofit"]
    assert provider["api"] == "http://127.0.0.1:8091/v1"
    assert provider["default_model"] == "auto"
    assert set(provider["models"]) == {"auto", "active:main", "active:aux"}
    assert "fallback_providers" not in config
    assert "custom_providers" not in config


def test_sirvir_distribution_owns_the_support_handbook() -> None:
    manifest = yaml.safe_load((SIRVIR / "distribution.yaml").read_text())

    assert manifest["version"] == "2"
    assert set(manifest["distribution_owned"]) == {
        "README.md",
        "SOUL.md",
        "AGENTS.md",
        "config.yaml",
    }


def test_sirvir_installer_copies_handbook_and_preserves_user_state(tmp_path: Path) -> None:
    from plugin_tools import install_sirvir_profile

    first = install_sirvir_profile(hermes_home=tmp_path)
    installed = tmp_path / "profiles" / "sirvir"
    memory = installed / "memories" / "USER.md"
    memory.parent.mkdir()
    memory.write_text("keep user state")

    second = install_sirvir_profile(hermes_home=tmp_path)

    assert first["updated"] is False
    assert second["updated"] is True
    assert (installed / "README.md").read_text() == (SIRVIR / "README.md").read_text()
    assert memory.read_text() == "keep user state"
