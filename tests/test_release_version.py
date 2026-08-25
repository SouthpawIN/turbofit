from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "2.4.0"


def test_all_release_surfaces_use_release_version() -> None:
    plugin = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    distribution = yaml.safe_load((ROOT / "distribution.yaml").read_text())
    dashboard = json.loads((ROOT / "dashboard/manifest.json").read_text())
    skill_frontmatter = (ROOT / "SKILL.md").read_text().split("---", 2)[1]
    skill = yaml.safe_load(skill_frontmatter)
    bundled_skill_frontmatter = (ROOT / "skills/turbofit/SKILL.md").read_text().split("---", 2)[1]
    bundled_skill = yaml.safe_load(bundled_skill_frontmatter)

    assert plugin["version"] == RELEASE_VERSION
    assert distribution["version"] == RELEASE_VERSION
    assert dashboard["version"] == RELEASE_VERSION
    assert skill["version"] == RELEASE_VERSION
    assert bundled_skill["version"] == RELEASE_VERSION


def test_release_readmes_show_every_new_2_2_model() -> None:
    required = {
        "Maple Preview 20B-A1B",
        "Qwen 3.8 27B Unleashed",
        "Ornith 1.5 35A3B",
        "MiniMax Music 3",
        "NVIDIA Parakeet TDT 0.6B v3",
        "Soprano TTS",
    }
    for relative in ("README.md", "skills/turbofit/README.md"):
        text = (ROOT / relative).read_text()
        assert required <= {name for name in required if name in text}, relative


def test_readme_qwen_variants_match_active_catalog() -> None:
    catalog = json.loads((ROOT / "references/model-catalog.json").read_text())
    qwen = [row for row in catalog["models"] if row["id"].startswith("qwen3-8-27b-")]
    unleashed = [row for row in qwen if "unleashed" in row["id"]]
    assert len(unleashed) == 2
    assert {row["quantization"] for row in unleashed} == {"UD-IQ3_XXS", "UD-Q3_K_XL"}
    readme = (ROOT / "README.md").read_text()
    assert "Qwen 3.8 27B Unleashed" in readme
    assert "Ornith 1.5 35A3B" in readme
