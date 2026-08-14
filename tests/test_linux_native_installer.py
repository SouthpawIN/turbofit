from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_linux_service_installer_installs_controller_and_continuous_gateway() -> None:
    text = (ROOT / "scripts/install-controller-service").read_text(encoding="utf-8")

    assert 'GATEWAY_UNIT_PATH = UNIT_PATH.with_name("turbofit-gateway.service")' in text
    assert 'gateway = ROOT / "scripts" / "turbofit-gateway.py"' in text
    assert "TURBOFIT_ALLOW_API=true" in text
    assert "TURBOFIT_CAMPAIGN_LEASE=" in text
    assert '"enable", "turbofit-controller.service", "turbofit-gateway.service"' in text
    assert '"restart", "turbofit-gateway.service", "turbofit-controller.service"' in text
