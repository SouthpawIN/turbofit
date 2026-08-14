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

    assert '[ValidateSet("cuda", "cpu")]' in text
    assert "Unregister-ScheduledTask" in text
    assert "install-dspark-runtime" in text
    assert "TurbofitGateway" in text
    assert "turbofit-gateway.py" in text
    assert '"--alias", "bonsai-27b-1bit-128k-main"' in text
