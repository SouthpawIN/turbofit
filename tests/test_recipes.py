from __future__ import annotations

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
    resolved = RecipeBook.load(RECIPES).resolve(row("Carwin Nano", "auto", 131_072))

    assert len(resolved.components) == 1
    component = resolved.components[0]
    assert component.role == "main"
    assert component.method == "mtp"
    assert component.gpu == "0"
    assert "--spec-type" in component.command
    assert "draft-mtp" in component.command


def test_small_dedicated_pair_pins_aux_gpu0_and_main_gpu1() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("Ternary Bonsai", "1 Bit Bonsai", 65_536))

    assert [(item.role, item.gpu) for item in resolved.components] == [("aux", "0"), ("main", "1")]
    assert all(item.method == "dspark" for item in resolved.components)
    assert all(item.kind == "docker" for item in resolved.components)


def test_bonsai_262k_uses_baseline_not_dspark() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("Ternary Bonsai", "1 Bit Bonsai", 262_144))

    assert all(item.method == "baseline" for item in resolved.components)
    assert all("DRAFT_MODEL" not in item.environment for item in resolved.components)


def test_one_bit_bonsai_one_million_uses_four_x_yarn_across_both_gpus() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("1 Bit Bonsai", "auto", 1_048_576))

    component = resolved.components[0]
    assert component.kind == "docker"
    assert component.image == "turbofit-bonsai-1bit:local"
    assert component.gpu == "0,1"
    assert component.method == "baseline"
    assert component.environment["CTX"] == "1048576"
    assert component.command == (
        "--split-mode", "layer", "--tensor-split", "1,1",
        "--rope-scaling", "yarn", "--rope-scale", "4",
        "--yarn-orig-ctx", "262144",
    )


def test_large_main_reserves_aux_gpu_then_fits_across_visible_cards() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("GLM 5.2", "Carwin Nano", 65_536))

    aux, main = resolved.components
    assert aux.role == "aux" and aux.gpu == "0"
    assert main.role == "main" and main.gpu == "0,1"
    assert main.method == "baseline"
    assert main.command[main.command.index("-c") + 1] == "65536"
    assert "--fit" in main.command


def test_grm_262k_uses_measured_seven_layer_split_across_both_gpus() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("GRM 2.6 Plus", "Carwin Nano", 262_144))

    aux, main = resolved.components
    assert aux.gpu == "0"
    assert main.gpu == "0,1"
    assert main.command[main.command.index("--split-mode") + 1] == "layer"
    assert main.command[main.command.index("--tensor-split") + 1] == "7,58"
    assert main.command[main.command.index("--main-gpu") + 1] == "1"
    assert main.command[main.command.index("--fit") + 1] == "off"


def test_one_million_context_applies_matching_four_x_yarn_to_both_models() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("GRM 2.6 Plus", "Carwin Nano", 1_048_576))

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


def test_recipe_compiles_deterministic_profile_name_and_aliases() -> None:
    resolved = RecipeBook.load(RECIPES).resolve(row("GRM 2.6 Plus", "Carwin Nano", 131_072))

    assert resolved.profile_name == "grm-2-6-plus-carwin-nano-128k"
    assert resolved.main_alias == "grm-2-6-plus"
    assert resolved.aux_alias == "carwin-nano"
    assert resolved.aux_mode == "dedicated"
