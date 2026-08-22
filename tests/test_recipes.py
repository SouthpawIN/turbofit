from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.recipes import RecipeBook, resolve_native_backend
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
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


def test_backend_detection_uses_vulkan_before_cpu_when_no_vendor_cli_exists() -> None:
    available = {"vulkaninfo"}

    assert resolve_native_backend(
        platform_name="linux", which=lambda command: command if command in available else None,
    ) == "vulkan"


def test_small_dedicated_pair_pins_aux_gpu0_and_main_gpu1() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Ternary Bonsai", "1 Bit Bonsai", 65_536)
    )

    assert [(item.role, item.gpu) for item in resolved.components] == [("aux", "0"), ("main", "1")]
    assert all(item.method == "dspark" for item in resolved.components)
    assert all(item.kind == "process" for item in resolved.components)


def test_bonsai_262k_uses_baseline_not_dspark() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Ternary Bonsai", "1 Bit Bonsai", 262_144)
    )

    assert all(item.method == "baseline" for item in resolved.components)
    assert all("--model-draft" not in item.command for item in resolved.components)


def test_one_bit_bonsai_one_million_uses_four_x_yarn_across_both_gpus() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("1 Bit Bonsai", "auto", 1_048_576)
    )

    component = resolved.components[0]
    assert component.kind == "process"
    assert component.gpu == "0,1"
    assert component.method == "baseline"
    assert component.command[component.command.index("-c") + 1] == "1048576"
    assert component.command[component.command.index("--split-mode") + 1] == "layer"
    assert component.command[component.command.index("--tensor-split") + 1] == "1,1"
    assert component.command[component.command.index("--rope-scale") + 1] == "4"
    assert component.command[component.command.index("--yarn-orig-ctx") + 1] == "262144"


def test_large_main_reserves_aux_gpu_then_fits_across_visible_cards() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("GLM 5.2", "Carwin Nano", 65_536)
    )

    aux, main = resolved.components
    assert aux.role == "aux" and aux.gpu == "0"
    assert main.role == "main" and main.gpu == "0,1"
    assert main.method == "mtp"
    assert main.command[main.command.index("-c") + 1] == "65536"
    assert "--fit" not in main.command
    assert main.command[0].endswith("ik-llama.cpp-f2328aa0c19954d0ab31a3de60fbf50e47c2429f/build-cuda/bin/llama-server")
    assert main.command[main.command.index("--spec-type") + 1] == "mtp:n_max=4,p_min=0.5"
    assert "--cpu-moe" in main.command
    assert "--no-mmap" in main.command
    assert "-dsa" in main.command and "-fidx" in main.command
    assert "--cache-type-k" not in main.command


def test_ternary_bonsai_uses_pinned_prism_runtime() -> None:
    component = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("Ternary Bonsai", "auto", 65_536)
    ).components[0]

    assert component.command[0].endswith("prism-llama.cpp-9ca265a57f85f2117942490f421f64a226dd9847/build-cuda/bin/llama-server")


def test_qwen_262k_uses_pinned_mtp_sidecar_and_projector() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("qwen3-8-27b-q4-mtp", "Carwin Nano", 262_144)
    )

    aux, main = resolved.components
    assert aux.gpu == "0"
    assert main.gpu == "1"
    assert main.command[main.command.index("--model-draft") + 1].endswith("mtp-Qwen3.8-27B-Q4_0.gguf")
    assert main.command[main.command.index("--mmproj") + 1].endswith("mmproj-Qwen3.8-27B-Q8_0.gguf")
    assert main.command[main.command.index("--spec-type") + 1] == "draft-mtp"


def test_one_million_context_applies_matching_four_x_yarn_to_both_models() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("qwen3-8-27b-q4-mtp", "Carwin Nano", 1_048_576)
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
    assert main.command[main.command.index("--model-draft") + 1].endswith("mtp-Qwen3.8-27B-Q4_0.gguf")
    assert main.command[main.command.index("--spec-type") + 1] == "draft-mtp"


def test_every_resolved_component_enables_jinja_tool_calling() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    cases = (
        row("Carwin Nano", "auto", 131_072),
        row("qwen3-8-27b-q4-mtp", "Carwin Nano", 131_072),
        row("Ternary Bonsai", "1 Bit Bonsai", 65_536),
    )

    for case in cases:
        for component in book.resolve(case).components:
            assert "--jinja" in component.command


def test_every_resolved_component_has_bounded_turbohaul_microbatching() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    cases = (
        row("Carwin Nano", "auto", 131_072),
        row("qwen3-8-27b-q4-mtp", "ornith-1-5-35a3b", 262_144),
        row("Qwen 3.8 27B Unleashed", "auto", 65_536),
    )

    for case in cases:
        for component in book.resolve(case).components:
            command = component.command
            batch = int(command[command.index("-b") + 1])
            ubatch = int(command[command.index("-ub") + 1])
            assert 1 <= ubatch <= batch
            assert ubatch <= 512


def test_every_dedicated_catalog_recipe_uses_distinct_role_ports() -> None:
    book = RecipeBook.load(ROOT / "references/model-recipes.json", platform_name="linux")
    matrix = json.loads((ROOT / "references/configuration-matrix.json").read_text())
    for item in matrix["rows"]:
        if item["auxiliary"] == "auto":
            continue
        recipe = book.resolve_catalog_configuration(item)
        by_role = {component.role: component.port for component in recipe.components}
        assert by_role == {"aux": 11610, "main": 11605}


def test_macos_resolves_bonsai_to_native_metal_process() -> None:
    data = json.loads(RECIPES.read_text())
    component = RecipeBook(data, platform_name="darwin").resolve(
        row("1 Bit Bonsai", "auto", 131_072)
    ).components[0]

    assert component.kind == "process"
    assert component.command[0].endswith(
        "prism-llama.cpp-9ca265a57f85f2117942490f421f64a226dd9847/build-metal/bin/llama-server"
    )
    assert "--jinja" in component.command
    assert "--model-draft" in component.command
    assert component.command[component.command.index("--spec-type") + 1] == "draft-dspark"
    assert component.command[component.command.index("--spec-draft-n-max") + 1] == "4"
    assert component.command[component.command.index("-ngld") + 1] == "auto"
    assert component.model_path.endswith("Bonsai-27B-Q1_0.gguf")


def test_every_catalog_configuration_compiles_to_an_actual_jinja_launch_recipe() -> None:
    book = RecipeBook.load(RECIPES, platform_name="linux")
    matrix = json.loads((ROOT / "references" / "configuration-matrix.json").read_text())

    resolved = [book.resolve_catalog_configuration(item) for item in matrix["rows"]]

    assert len(resolved) == 336
    assert len({item.row_id for item in resolved}) == 336
    assert all(component.command and "--jinja" in component.command for item in resolved for component in item.components)


def test_unleashed_q3_compiles_with_vision_projector() -> None:
    component = RecipeBook.load(RECIPES).resolve_catalog_configuration({
        "id": "qwen3-8-27b-unleashed-ud-q3-k-xl--auto--262k",
        "main": "qwen3-8-27b-unleashed-ud-q3-k-xl",
        "auxiliary": "auto",
        "context": 262_144,
        "status": "candidate",
    }).components[0]

    assert component.model_path.endswith("Qwen3.8-27B-Unleashed-UD-Q3_K_XL.gguf")
    assert component.projector_path.endswith("mmproj-Unleashed-f16.gguf")
    assert component.method == "baseline"
    assert "--jinja" in component.command
    assert "--host" in component.command


def test_ornith_aux_compiles_with_expert_offload() -> None:
    recipe = RecipeBook.load(RECIPES).resolve_catalog_configuration({
        "id": "qwen3-8-27b-unleashed-ud-q3-k-xl--ornith-1-5-35a3b--64k",
        "main": "qwen3-8-27b-unleashed-ud-q3-k-xl",
        "auxiliary": "ornith-1-5-35a3b",
        "context": 65_536,
        "status": "candidate",
    })
    aux = next(item for item in recipe.components if item.role == "aux")
    assert aux.model_path.endswith("Ornith-1.5-35B-Q4_K_M.gguf")
    assert "--cpu-moe" in aux.command


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
    assert fp16.command[fp16.command.index("--model-draft") + 1].endswith("Ternary-Bonsai-27B-dspark-bf16.gguf")
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
    assert bonsai.command[bonsai.command.index("--model-draft") + 1].endswith("Bonsai-27B-dspark-bf16.gguf")
    assert bonsai.projector_path.endswith("Bonsai-27B-mmproj-BF16.gguf")
    assert minimax.model_path.endswith("MiniMax-M3-UD-Q4_K_M-00001-of-00007.gguf")
    assert laguna.model_path.endswith("laguna-s-2.1-F16.gguf")


def test_recipe_compiles_deterministic_profile_name_and_aliases() -> None:
    resolved = RecipeBook.load(RECIPES, platform_name="linux").resolve(
        row("qwen3-8-27b-q4-mtp", "Carwin Nano", 131_072)
    )

    assert resolved.profile_name == "qwen3-8-27b-q4-mtp-carwin-nano-128k"
    assert resolved.main_alias == "qwen3-8-27b-q4-mtp"
    assert resolved.aux_alias == "carwin-nano"
    assert resolved.aux_mode == "dedicated"


def test_cpu_only_qwen_1m_uses_host_memory_and_cpu_binary() -> None:
    hardware = HardwareFingerprint("linux", "x86_64", 393216)
    component = RecipeBook.load(
        RECIPES, platform_name="linux", backend_name="cpu", hardware=hardware,
    ).resolve(row("qwen3-8-27b-q4-mtp", "auto", 1_048_576)).components[0]

    assert component.command[0].endswith("build-cpu/bin/llama-server")
    assert component.command[component.command.index("-ngl") + 1] == "0"
    assert "--split-mode" not in component.command
    assert "--tensor-split" not in component.command
    assert "--main-gpu" not in component.command


def test_cpu_only_dspark_uses_the_pinned_prism_cpu_runtime() -> None:
    hardware = HardwareFingerprint("linux", "x86_64", 131072)
    component = RecipeBook.load(
        RECIPES, platform_name="linux", backend_name="cpu", hardware=hardware,
    ).resolve(row("1 Bit Bonsai", "auto", 65_536)).components[0]

    assert component.command[0].endswith(
        "prism-llama.cpp-9ca265a57f85f2117942490f421f64a226dd9847/build-cpu/bin/llama-server"
    )
    assert component.command[component.command.index("-ngl") + 1] == "0"
    assert component.command[component.command.index("-ngld") + 1] == "0"


def test_dedicated_gpu_qwen_1m_places_large_kv_cache_in_detected_host_ram() -> None:
    hardware = HardwareFingerprint(
        "linux", "x86_64", 393216,
        devices=(
            AcceleratorDevice(0, "a", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "01"),
            AcceleratorDevice(1, "b", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "02"),
        ),
    )
    component = RecipeBook.load(
        RECIPES, platform_name="linux", backend_name="cuda", hardware=hardware,
    ).resolve(row("qwen3-8-27b-q4-mtp", "auto", 1_048_576)).components[0]

    assert component.gpu == "0,1"
    assert "--no-kv-offload" in component.command
    assert component.command[component.command.index("-ngl") + 1] == "auto"


def test_unified_memory_qwen_1m_uses_one_shared_pool_without_host_gpu_double_counting() -> None:
    hardware = HardwareFingerprint(
        "linux", "aarch64", 131072,
        devices=(AcceleratorDevice(0, "u", "NVIDIA GB10", "nvidia", "cuda", 131072, "12.1", "01"),),
    )
    component = RecipeBook.load(
        RECIPES, platform_name="linux", backend_name="cuda", hardware=hardware,
    ).resolve(row("qwen3-8-27b-q4-mtp", "auto", 1_048_576)).components[0]

    assert component.gpu == "0"
    assert "--no-kv-offload" not in component.command
    assert "--tensor-split" not in component.command

    direct = RecipeBook.load(
        RECIPES, platform_name="linux", backend_name="cuda", hardware=hardware,
    ).resolve_component(
        "qwen3-8-27b-q4-mtp", role="main", gpu="0,1", port=11605,
        context=1_048_576,
    )
    assert direct.gpu == "0"
