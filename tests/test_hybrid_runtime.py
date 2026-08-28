from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from turbofit_runtime.hybrid_runtime import HybridCatalog


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "runtime-profiles" / "hybrid-models.json"


def test_catalog_defines_large_local_models_without_host_paths() -> None:
    catalog = HybridCatalog.load(CATALOG)

    assert set(catalog.models) == {
        "laguna-s2-1-q4-k-m",
        "minimax-m3-q4-k-m",
        "glm-5-2-2-788bpw",
        "qwen3-8-flash-next-ud-q4-k-xl",
    }
    minimax = catalog.models["minimax-m3-q4-k-m"]
    assert minimax.name == "MiniMax M3 UD-Q4_K_M"
    assert minimax.source.files[0].filename.startswith("UD-Q4_K_M/")
    assert all(model.source.revision for model in catalog.models.values())
    assert all(model.source.files for model in catalog.models.values())
    assert all(not item.filename.startswith("/") for model in catalog.models.values() for item in model.source.files)
    configurations = {
        model.id: model.configurations[0]
        for model in catalog.models.values()
    }
    assert configurations["laguna-s2-1-q4-k-m"].status == "validated"
    assert configurations["minimax-m3-q4-k-m"].status == "validated"
    assert configurations["glm-5-2-2-788bpw"].status == "validated"
    assert configurations["laguna-s2-1-q4-k-m"].evidence
    assert configurations["minimax-m3-q4-k-m"].evidence
    assert configurations["glm-5-2-2-788bpw"].evidence


def test_dual_24gb_configuration_uses_system_ram_and_both_gpus() -> None:
    catalog = HybridCatalog.load(CATALOG)

    for model in catalog.models.values():
        config = model.configuration("dual-24gb-64k")
        assert config.context == 65_536
        assert config.min_system_ram_mib > sum(config.min_vram_mb_per_card)
        assert config.launch.split_mode in {"layer", "graph"}
        if model.id == "qwen3-8-flash-next-ud-q4-k-xl":
            assert config.launch.tensor_split == ()
        else:
            assert config.launch.tensor_split == (1.0, 1.0)


def test_each_system_ram_model_has_64k_128k_262k_and_1m_configurations() -> None:
    catalog = HybridCatalog.load(CATALOG)

    for model in (
        catalog.models["laguna-s2-1-q4-k-m"],
        catalog.models["minimax-m3-q4-k-m"],
        catalog.models["glm-5-2-2-788bpw"],
    ):
        assert {config.context for config in model.configurations} == {
            65_536, 131_072, 262_144, 1_048_576,
        }
        for config in model.configurations:
            command = config.command(binary="/opt/llama-server", model_path="/models/model.gguf", port=8080)
            assert "--jinja" in command
            assert config.status == "validated"
            assert config.evidence
            assert "--no-kv-offload" in command

    laguna_1m = catalog.models["laguna-s2-1-q4-k-m"].configuration("dual-24gb-1m")
    laguna_command = laguna_1m.command(binary="/opt/llama-server", model_path="/models/model.gguf", port=8080)
    assert laguna_command[laguna_command.index("--rope-scaling") + 1] == "yarn"
    assert laguna_command[laguna_command.index("--rope-scale") + 1] == "4"


def test_qwen_flash_next_uses_measured_context_dependent_expert_placement() -> None:
    model = HybridCatalog.load(CATALOG).models["qwen3-8-flash-next-ud-q4-k-xl"]

    assert [config.launch.cpu_moe_layers for config in model.configurations] == [36, 37, 39, 38]
    assert [config.status for config in model.configurations] == [
        "validated", "configured-unmeasured", "validated", "configured-unmeasured",
    ]
    for config in model.configurations:
        command = config.command(binary="/opt/llama-server", model_path="/models/qwen.gguf", port=8080)
        assert "--tensor-split" not in command
        assert command[command.index("--numa") + 1] == "distribute"
    one_million = model.configuration("dual-24gb-1m").command(
        binary="/opt/llama-server", model_path="/models/qwen.gguf", port=8080
    )
    assert "--no-kv-offload" in one_million
    assert one_million[one_million.index("--rope-scaling") + 1] == "yarn"


def test_laguna_host_kv_profiles_use_measured_maximal_gpu_offload() -> None:
    laguna = HybridCatalog.load(CATALOG).models["laguna-s2-1-q4-k-m"]

    expected_cpu_moe = {
        "dual-24gb-64k": 34,
        "dual-24gb-128k": 34,
        "dual-24gb-262k": 40,
        "dual-24gb-1m": 40,
    }
    for configuration_id, cpu_moe_layers in expected_cpu_moe.items():
        configuration = laguna.configuration(configuration_id)
        assert configuration.launch.gpu_layers == 999
        assert configuration.launch.cpu_moe_layers == cpu_moe_layers
        assert "--no-kv-offload" in configuration.launch.extra_args
    assert laguna.configuration("dual-24gb-64k").launch.extra_args[-2:] == ("--main-gpu", "1")


def test_promoted_minimax_and_glm_rungs_encode_measured_placement() -> None:
    catalog = HybridCatalog.load(CATALOG)
    minimax = catalog.models["minimax-m3-q4-k-m"]
    assert [c.launch.cpu_moe_layers for c in minimax.configurations] == [56, 56, 58, 58]
    assert minimax.configuration("dual-24gb-1m").launch.extra_args[-2:] == ("--main-gpu", "1")
    glm = catalog.models["glm-5-2-2-788bpw"]
    for config in glm.configurations[:-1]:
        assert config.launch.cpu_moe_layers == "all"
        assert config.launch.extra_args[-2:] == ("--numa", "distribute")
    assert glm.configuration("dual-24gb-1m").launch.cpu_moe_layers == 72


def test_fit_requires_ram_and_every_gpu_budget() -> None:
    config = HybridCatalog.load(CATALOG).models["laguna-s2-1-q4-k-m"].configuration("dual-24gb-64k")

    assert config.fits(total_system_ram_mib=386_000, vram_mb_per_card=(24_576, 24_576))
    assert not config.fits(total_system_ram_mib=config.min_system_ram_mib - 1, vram_mb_per_card=(24_576, 24_576))
    assert not config.fits(
        total_system_ram_mib=386_000,
        available_system_ram_mib=config.required_available_ram_mib - 1,
        vram_mb_per_card=(24_576, 24_576),
    )
    assert not config.fits(total_system_ram_mib=386_000, vram_mb_per_card=(24_576, 16_000))


def test_command_builder_emits_cpu_moe_and_memory_policy() -> None:
    config = HybridCatalog.load(CATALOG).models["minimax-m3-q4-k-m"].configuration("dual-24gb-64k")

    command = config.command(
        binary="/opt/llama-server",
        model_path="/models/MiniMax-M3-00001-of-00007.gguf",
        port=11601,
    )

    assert command[:4] == ("/opt/llama-server", "-m", "/models/MiniMax-M3-00001-of-00007.gguf", "--port")
    assert command[command.index("--n-cpu-moe") + 1] == "56"
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("--threads") + 1] == "12"
    assert command[command.index("--tensor-split") + 1] == "1,1"
    assert "--no-mmap" not in command
    assert "--mlock" not in command


def test_glm_command_contains_required_dsa_cpu_moe_flags() -> None:
    catalog = HybridCatalog.load(CATALOG)
    glm = catalog.models["glm-5-2-2-788bpw"].configuration("dual-24gb-64k")
    command = glm.command(
        binary="/engines/ik/llama-server", model_path="/models/glm.gguf", port=11600
    )

    assert command[command.index("--threads") + 1] == "14"
    assert "--cpu-moe" in command
    assert "-dsa" in command
    assert "-fidx" in command
    assert command[command.index("--split-mode") + 1] == "layer"
    assert "--fit" not in command
    assert command[:3] == ("env", "GGML_CUDA_NO_PINNED=1", "/engines/ik/llama-server")
    assert command[command.index("-ngl") + 1] == "79"
    assert "--no-mmap" not in command


def test_hybrid_config_cli_runs_without_external_pythonpath() -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [str(ROOT / "scripts" / "turbofit-hybrid-config"), "list"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "laguna-s2-1-q4-k-m" in result.stdout
    assert "minimax-m3-q4-k-m" in result.stdout
    assert "glm-5-2-2-788bpw" in result.stdout


def test_catalog_rejects_validated_configuration_without_evidence(tmp_path: Path) -> None:
    text = CATALOG.read_text().replace(
        '"sha256:f09ff73a098ced577cc4d973ec0fd01d09a7e853d4e5c1e966c24df10701c08b"',
        "null",
        1,
    )
    path = tmp_path / "catalog.json"
    path.write_text(text)

    with pytest.raises(ValueError, match="validated configuration requires evidence"):
        HybridCatalog.load(path)
