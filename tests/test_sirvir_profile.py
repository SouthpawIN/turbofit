from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_turbofit_does_not_ship_a_stale_sirvir_snapshot() -> None:
    manifest = yaml.safe_load((ROOT / "distribution.yaml").read_text())

    assert not (ROOT / "profiles" / "sirvir").exists()
    assert "profiles/sirvir/" not in manifest["distribution_owned"]


def test_turbofit_points_to_canonical_github_sirvir() -> None:
    text = "\n".join(
        (ROOT / path).read_text()
        for path in ("README.md", "SKILL.md")
    )

    assert "https://github.com/SouthpawIN/sirvir" in text
    assert "GitHub-current" in text
    assert "tested pull requests" in text


def test_sirvir_install_tool_is_github_current() -> None:
    schema = __import__("schemas").TURBOFIT_CONFIGURE
    description = schema["parameters"]["properties"]["install_sirvir"]["description"]

    assert "SouthpawIN/sirvir" in description
    assert "bundled" not in description.lower()
