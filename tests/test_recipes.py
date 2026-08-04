from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.schema import MatrixRow


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "references/model-recipes.json"


def row(main: str, aux: str, context: int) -> MatrixRow:
    return MatrixRow(
        id=MatrixRow.make_id(main, aux, context), main=main, aux=aux, context=context,
        status="pending", method_priority=("dspark", "mtp", "nextn"),
    )


def test_auto_profile_launches_only_main_component() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Carwin Nano", "auto", 131_072)
    )

    assert len(resolved.components) == 1
    component = resolved.components[0]
    assert component.role == "main"
    assert component.method == "mtp"
    assert component.gpu == "0"
    assert "--spec-type" in component.command
    assert "draft-mtp" in component.command


def test_small_dedicated_pair_pins_aux_gpu0_and_main_gpu1() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Ternary Bonsai", "1 Bit Bonsai", 65_536)
    )

    assert [(item.role, item.gpu) for item in resolved.components] == [("aux", "0"), ("main", "1")]
    assert all(item.method == "dspark" for item in resolved.components)
    assert all(item.kind == "docker" for item in resolved.components)


def test_bonsai_262k_uses_baseline_not_dspark() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Ternary Bonsai", "1 Bit Bonsai", 262_144)
    )

    assert all(item.method == "baseline" for item in resolved.components)
    assert all("DRAFT_MODEL" not in item.environment for item in resolved.components)


def test_one_bit_bonsai_one_million_uses_four_x_yarn_across_both_gpus() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("1 Bit Bonsai", "auto", 1_048_576)
    )

    component = resolved.components[0]
    assert component.kind == "docker"
    assert component.image == "turbofit-bonsai-1bit:local"
    assert component.gpu == "0,1"
    assert component.method == "baseline"
    assert component.environment["CTX"] == "1048576"
    assert component.command == (
        "--jinja",
        "--split-mode", "layer", "--tensor-split", "1,1",
        "--rope-scaling", "yarn", "--rope-scale", "4",
        "--yarn-orig-ctx", "262144",
    )


def test_large_main_reserves_aux_gpu_then_fits_across_visible_cards() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("GLM 5.2", "Carwin Nano", 65_536)
    )

    aux, main = resolved.components
    assert aux.role == "aux" and aux.gpu == "0"
    assert main.role == "main" and main.gpu == "0,1"
    assert main.method == "baseline"
    assert main.command[main.command.index("-c") + 1] == "65536"
    assert "--fit" in main.command


def test_grm_262k_uses_measured_seven_layer_split_across_both_gpus() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("GRM 2.6 Plus", "Carwin Nano", 262_144)
    )

    aux, main = resolved.components
    assert aux.gpu == "0"
    assert main.gpu == "0,1"
    assert main.command[main.command.index("--split-mode") + 1] == "layer"
    assert main.command[main.command.index("--tensor-split") + 1] == "7,58"
    assert main.command[main.command.index("--main-gpu") + 1] == "1"
    assert main.command[main.command.index("--fit") + 1] == "off"


def test_one_million_context_applies_matching_four_x_yarn_to_both_models() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("GRM 2.6 Plus", "Carwin Nano", 1_048_576)
    )

    assert len(resolved.components) == 2
    aux, main = resolved.components
    for component in resolved.components:
        assert component.command[component.command.index("-c") + 1] == "1048576"
        assert component.command[component.command.index("--rope-scaling") + 1] == "yarn"
        assert component.command[component.command.index("--rope-scale") + 1] == "4"
        assert component.command[component.command.index("--yarn-orig-ctx") + 1] == "262144"

    assert aux.gpu == "0"
    assert aux.command[aux.command.index("--n-cpu-moe") + 1] == "5"
    assert aux.command[aux.command.index("--spec-type") + 1] == "draft-mtp"
    assert main.gpu == "1"
    assert main.command[main.command.index("-ngl") + 1] == "64"
    assert "--no-kv-offload" in main.command
    assert "--spec-type" not in main.command


def test_every_resolved_component_enables_jinja_tool_calling() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    cases = (
        row("Carwin Nano", "auto", 131_072),
        row("GRM 2.6 Plus", "Carwin Nano", 131_072),
        row("Ternary Bonsai", "1 Bit Bonsai", 65_536),
    )

    for case in cases:
        for component in book.resolve(case).components:
            assert "--jinja" in component.command


def test_macos_compiles_docker_only_bonsai_recipe_to_native_metal_process() -> None:
    data = json.loads(RECIPES.read_text())
    component = RecipeBook(data, platform_name="darwin").resolve(
        row("1 Bit Bonsai", "auto", 131_072)
    ).components[0]

    assert component.kind == "process"
    assert component.command[0] == data["atomic_binary"]
    assert "--jinja" in component.command
    assert "--model-draft" in component.command
    assert component.model_path.endswith("Bonsai-27B-Q1_0.gguf")


def test_every_catalog_configuration_compiles_to_an_actual_jinja_launch_recipe() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    matrix = json.loads((ROOT / "references" / "configuration-matrix.json").read_text())

    resolved = [book.resolve_catalog_configuration(item) for item in matrix["rows"]]

    assert len(resolved) == 192
    assert len({item.row_id for item in resolved}) == 192
    assert all(component.command and "--jinja" in component.command for item in resolved for component in item.components)


def test_catalog_variants_compile_distinct_artifacts_and_features() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    fp16 = book.resolve_catalog_configuration({
        "id": "ternary-bonsai-27b-fp16-vision-dspark--auto--64k",
        "main": "ternary-bonsai-27b-fp16-vision-dspark",
        "auxiliary": "auto",
        "context": 65_536,
        "status": "candidate",
    }).components[0]
    baseline = book.resolve_catalog_configuration({
        "id": "ternary-bonsai-27b--auto--64k",
        "main": "ternary-bonsai-27b",
        "auxiliary": "auto",
        "context": 65_536,
        "status": "candidate",
    }).components[0]

    assert fp16.model_path.endswith("Ternary-Bonsai-27B-F16.gguf")
    assert fp16.environment["DRAFT_MODEL"].endswith("Ternary-Bonsai-27B-dspark-bf16.gguf")
    assert fp16.projector_path.endswith("Ternary-Bonsai-27B-mmproj-BF16.gguf")
    assert fp16.method == "dspark"
    assert baseline.model_path.endswith("Ternary-Bonsai-27B-Q2_0.gguf")
    assert baseline.method == "baseline"


def test_requested_fp16_and_q4_variants_resolve_real_artifact_names() -> None:
    book = RecipeBook.load(RECIPES)
    bonsai = book.resolve_catalog_configuration({
        "id": "bonsai-27b-fp16-vision-dspark--auto--64k",
        "main": "bonsai-27b-fp16-vision-dspark",
        "auxiliary": "auto",
        "context": 65_536,
        "status": "candidate",
    }).components[0]
    minimax = book.resolve_catalog_configuration({
        "id": "minimax-m3-q4--auto--64k",
        "main": "minimax-m3-q4",
        "auxiliary": "auto",
        "context": 65_536,
        "status": "candidate",
    }).components[0]
    laguna = book.resolve_catalog_configuration({
        "id": "laguna-s2-1-fp16--auto--64k",
        "main": "laguna-s2-1-fp16",
        "auxiliary": "auto",
        "context": 65_536,
        "status": "candidate",
    }).components[0]

    assert bonsai.model_path.endswith("Bonsai-27B-F16.gguf")
    assert bonsai.environment["DRAFT_MODEL"].endswith("Bonsai-27B-dspark-bf16.gguf")
    assert bonsai.projector_path.endswith("Bonsai-27B-mmproj-BF16.gguf")
    assert minimax.model_path.endswith("MiniMax-M3-UD-Q4_K_M-00001-of-00007.gguf")
    assert laguna.model_path.endswith("laguna-s-2.1-F16.gguf")


def test_recipe_compiles_deterministic_profile_name_and_aliases() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("GRM 2.6 Plus", "Carwin Nano", 131_072)
    )

    assert resolved.profile_name == "grm-2-6-plus-carwin-nano-128k"
    assert resolved.main_alias == "grm-2-6-plus"
    assert resolved.aux_alias == "carwin-nano"
    assert resolved.aux_mode == "dedicated"
