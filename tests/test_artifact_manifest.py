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
    mlx_destinations = {
        item["destination"]
        for item in payload["artifacts"]
        if "qwen3-8-27b-uncensored-mlx-8bit" in item.get("families", [])
    }
    # Catalog entries tracked in the manifest before their recipe rows are
    # automatable: MiniMax M3 has no local matrix row, and Darwin 180B RSI
    # is a PLE/SSD-streamed checkpoint awaiting its dedicated runtime flavor.
    known_non_matrix_repos = {
        "unsloth/MiniMax-M3-GGUF",
        "Baekpica/Darwin-180B-RSI-Mixed-Quant-GGUF",
        # vLLM serves the checkpoint directory as a unit; the manifest tracks it
        # with a snapshot digest rather than one row per safetensors shard.
        "thinkingmachines/Inkling-2-Small",
    }
    known_non_matrix_destinations = {
        item["destination"]
        for item in payload["artifacts"]
        if item.get("repo_id") in known_non_matrix_repos
    }

    assert payload["schema"] == "turbofit.artifact-manifest/v1"
    assert expected_destinations <= actual
    assert actual == expected_destinations | known_non_matrix_destinations | mlx_destinations
    assert mlx_destinations
    assert all(
        len(item["sha256"]) == 64
        or (
            item["sha256"].startswith("snapshot:")
            and len(item["sha256"].removeprefix("snapshot:")) == 40
        )
        for item in payload["artifacts"]
    )
    assert all(len(item["revision"]) == 40 for item in payload["artifacts"])
    assert all(item["size_bytes"] > 0 for item in payload["artifacts"])


def test_model_catalog_pins_every_source_revision() -> None:
    catalog = ModelCatalog.load(ROOT / "references/model-catalog.json")

    assert len(catalog.models) == 86  # + Qwen 3.6/3.5, LFM2.5, LFM3 Omni, Nemotron, 7 uncensored builds, Inkling 2 Small (vLLM)
    assert all(len(model.revision) == 40 for model in catalog.models)



def test_qwen_dflash2_drafter_is_immutable_and_bonsai_specific_drafters_stay_separate() -> None:
    manifest = json.loads((ROOT / "references/artifact-manifest.json").read_text())
    dflash = next(item for item in manifest["artifacts"] if item["path"] == "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
    bonsai = [item for item in manifest["artifacts"] if "Bonsai-27B-dspark" in item["path"]]

    assert dflash["repo_id"] == "incoai/Qwen3.8-27B-DFlash2-GGUF"
    assert dflash["revision"] == "6cb5872e2cee6b4e780a8414922350be8e42d65c"
    assert dflash["sha256"] == "18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594"
    assert dflash["families"] == ["qwen3-8-27b-q4-dflash2"]
    assert bonsai
    assert all("qwen3-8-27b" not in item["families"] for item in bonsai)
