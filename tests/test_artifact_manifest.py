from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.recipes import RecipeBook

ROOT = Path(__file__).parents[1]


def test_artifact_manifest_covers_every_executable_matrix_file() -> None:
    matrix = json.loads((ROOT / "references/configuration-matrix.json").read_text())["rows"]
    book = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cpu")
    expected: set[str] = set()
    for row in matrix:
        for component in book.resolve_catalog_configuration(row).components:
            expected.add(component.model_path)
            if component.projector_path:
                expected.add(component.projector_path)
            command = list(component.command)
            if "--model-draft" in command:
                expected.add(command[command.index("--model-draft") + 1])

    payload = json.loads((ROOT / "references/artifact-manifest.json").read_text())
    model_root = Path.home() / "Models/storage/gguf"
    expected_destinations = {str(Path(path).relative_to(model_root)) for path in expected}
    actual = {item["destination"] for item in payload["artifacts"]}

    assert payload["schema"] == "turbofit.artifact-manifest/v1"
    assert expected_destinations <= actual
    assert len(actual) == 44
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])
    assert all(len(item["revision"]) == 40 for item in payload["artifacts"])
    assert all(item["size_bytes"] > 0 for item in payload["artifacts"])


def test_model_catalog_pins_every_source_revision() -> None:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")

    assert len(catalog.models) == 47
    assert all(len(model.revision) == 40 for model in catalog.models)


def test_deepseek_artifacts_pin_current_0731_quant_and_official_upstream() -> None:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    deepseek = [model for model in catalog.models if model.id.startswith("deepseek-v4-flash-0731-")]
    manifest = json.loads((ROOT / "references/artifact-manifest.json").read_text())
    artifacts = [item for item in manifest["artifacts"] if item["repo_id"] == "unsloth/DeepSeek-V4-Flash-0731-GGUF"]

    assert len(deepseek) == 5
    assert {model.revision for model in deepseek} == {"fbbb5b93fb787c21338159b0af3318bb3f4d9768"}
    assert {model.upstream_revision for model in deepseek} == {"7872f01b1d1fe23eabc4c98b48bffcef5a386062"}
    assert artifacts
    assert {item["revision"] for item in artifacts} == {"fbbb5b93fb787c21338159b0af3318bb3f4d9768"}
    assert any(item["path"] == "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf" for item in artifacts)
