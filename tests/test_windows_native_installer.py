from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "install-windows-native-service.ps1"


def test_windows_installer_is_user_scoped_jinja_and_health_verified() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Register-ScheduledTask" in text
    assert "-RunLevel Limited" in text
    assert '"--jinja"' in text
    assert '"--fit", "on"' in text
    assert "/health" in text
    assert "did not become healthy" in text
    assert "api_key" not in text.lower()
    assert "RunAsAdministrator" not in text


def test_windows_installer_supports_one_step_cuda_cpu_gateway_and_uninstall() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '[ValidateSet("cuda", "vulkan", "cpu")]' in text
    assert "TURBOFIT_GATEWAY_HOST" in text
    assert "Unregister-ScheduledTask" in text
    assert "install-dspark-runtime" in text
    assert "TurbofitGateway" in text
    assert "turbofit-gateway.py" in text
    assert '"--alias", "bonsai-27b-1bit-128k-main"' in text


def test_windows_installer_does_not_treat_native_stderr_as_a_powershell_failure() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.count('$ErrorActionPreference = "Continue"') >= 2
    assert text.count("$exitCode = $LASTEXITCODE") >= 2
    assert text.count("exit $exitCode") >= 2


def test_readme_separates_hermes_gateway_from_turbofit_8091() -> None:
    readme = (ROOT / "README.md").read_text()
    windows = (ROOT / "docs" / "windows-native-install.md").read_text()
    skill = (ROOT / "SKILL.md").read_text()

    assert "Hermes messaging gateway" in readme
    assert "127.0.0.1:8091" in readme
    assert "WinError 10061" in readme
    assert "docs/windows-native-install.md" in readme
    assert "## Tailscale" in readme
    assert "/turbofit serve" in readme
    assert "Funnel is never used" in readme or "never uses Funnel" in readme
    assert "never from Sirvir" not in skill
    assert "Sirvir handles install" in readme or "Sirvir handles install and setup" in readme
    assert "WinError 10061" in windows
    assert "TurbofitGateway" in windows
