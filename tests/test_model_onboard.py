from __future__ import annotations

import json
import runpy
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_model_onboarding_replaces_variants_without_matrix_count_drift(tmp_path: Path) -> None:
    module = runpy.run_path(str(ROOT / "scripts/turbofit-model-onboard"), run_name="model_onboard_test")
    for name, relative in {
        "CATALOG": "references/model-catalog.json",
        "MATRIX": "references/configuration-matrix.json",
        "RECIPES": "references/model-recipes.json",
        "ARTIFACTS": "references/artifact-manifest.json",
        "POLICY": "references/catalog-campaign-policy.json",
    }.items():
        destination = tmp_path / Path(relative).name
        destination.write_bytes((ROOT / relative).read_bytes())
        module[name] = destination

    catalog = json.loads(module["CATALOG"].read_text())
    recipes = json.loads(module["RECIPES"].read_text())
    artifacts = json.loads(module["ARTIFACTS"].read_text())
    old_id = "qwen3-8-27b-q4-mtp"
    model = deepcopy(next(item for item in catalog["models"] if item["id"] == old_id))
    model["id"] = "qwen3-8-27b-test"
    model["name"] = "Qwen3.8 27B test"
    recipe = deepcopy(recipes["variants"][old_id])
    recipe["alias"] = model["id"]
    artifact = deepcopy(artifacts["artifacts"][0])
    artifact["destination"] = "Qwen3.8-27B/test.gguf"
    artifact["families"] = [model["id"]]
    before = len(json.loads(module["MATRIX"].read_text())["rows"])

    result = module["build"]({
        "schema": "turbofit.model-onboarding/v1",
        "family": "Qwen3.8 27B",
        "replace_model_ids": [old_id],
        "models": [model],
        "recipes": {model["id"]: recipe},
        "artifacts": [artifact],
    })

    ids = {item["id"] for item in result["catalog"]["models"]}
    assert old_id not in ids
    assert model["id"] in ids
    assert len(result["matrix"]["rows"]) == before
    assert old_id in result["policy"]["deferred_models"]
    assert result["policy"]["replacement"]["status"] == "onboarded-awaiting-physical-evidence"
