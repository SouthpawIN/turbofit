from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_turbofit_does_not_ship_a_stale_turbosouth_snapshot() -> None:
    manifest = yaml.safe_load((ROOT / "distribution.yaml").read_text())

    assert not (ROOT / "profiles" / "turbosouth").exists()
    assert "profiles/turbosouth/" not in manifest["distribution_owned"]


def test_turbofit_points_to_canonical_github_turbosouth() -> None:
    text = "\n".join(
        (ROOT / path).read_text()
        for path in ("README.md", "SKILL.md")
    )

    assert "https://github.com/SouthpawIN/turbosouth" in text
    assert "GitHub-current" in text
    assert "tested pull requests" in text


def test_turbosouth_install_tool_is_github_current() -> None:
    schema = __import__("schemas").TURBOFIT_CONFIGURE
    # canonical property is now install_turbosouth, install_turbosouth remains as legacy alias
    props = schema["parameters"]["properties"]
    key = "install_turbosouth" if "install_turbosouth" in props else "install_turbosouth"
    description = props[key]["description"]

    assert "SouthpawIN/turbosouth" in description
    assert "bundled" not in description.lower()


def test_turbosouth_github_install_uses_full_git_url() -> None:
    source = (ROOT / "plugin_tools.py").read_text()
    assert '"https://github.com/SouthpawIN/turbosouth.git"' in source
    assert '"SouthpawIN/turbosouth"' not in source
