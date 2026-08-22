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
    assert len(actual) == 35
    assert all(len(item["sha256"]) == 64 for item in payload["artifacts"])
    assert all(len(item["revision"]) == 40 for item in payload["artifacts"])
    assert all(item["size_bytes"] > 0 for item in payload["artifacts"])


def test_model_catalog_pins_every_source_revision() -> None:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")

    assert len(catalog.models) == 45
    assert all(len(model.revision) == 40 for model in catalog.models)


def test_unleashed_and_ornith_artifacts_are_pinned() -> None:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")
    unleashed = [model for model in catalog.models if "unleashed" in model.id]
    ornith = [model for model in catalog.models if model.id.startswith("ornith-1-5-35a3b")]
    manifest = json.loads((ROOT / "references/artifact-manifest.json").read_text())
    unleashed_artifacts = [item for item in manifest["artifacts"] if item["repo_id"] == "outsourc-e/Qwen3.8-27B-Unleashed-GGUF"]
    ornith_artifacts = [item for item in manifest["artifacts"] if item["repo_id"] == "ornith-ai/Ornith-1.5-35B-A3B-GGUF"]

    assert len(unleashed) == 2
    assert {model.revision for model in unleashed} == {"67a999218fd7002f11bf82bc81d6289beea60841"}
    assert ornith and ornith[0].revision == "fbbaed45c2f0e200276ffa51701a24d45dc7f57e"
    assert unleashed_artifacts
    assert ornith_artifacts
    assert any(item["path"] == "Qwen3.8-27B-Unleashed-UD-Q3_K_XL.gguf" for item in unleashed_artifacts)
    assert any(item["path"] == "Ornith-1.5-35B-Q4_K_M.gguf" for item in ornith_artifacts)
