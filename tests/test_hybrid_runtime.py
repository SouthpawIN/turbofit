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
        "minimax-m3-mxfp4-moe",
        "glm-5-2-2-788bpw",
    }
    assert all(model.source.revision for model in catalog.models.values())
    assert all(model.source.files for model in catalog.models.values())
    assert all(not item.filename.startswith("/") for model in catalog.models.values() for item in model.source.files)
    configurations = {
        model.id: model.configurations[0]
        for model in catalog.models.values()
    }
    assert configurations["laguna-s2-1-q4-k-m"].status == "validated"
    assert configurations["minimax-m3-mxfp4-moe"].status == "validated"
    assert configurations["glm-5-2-2-788bpw"].status == "validated"
    assert configurations["laguna-s2-1-q4-k-m"].evidence
    assert configurations["minimax-m3-mxfp4-moe"].evidence
    assert configurations["glm-5-2-2-788bpw"].evidence


def test_dual_24gb_configuration_uses_system_ram_and_both_gpus() -> None:
    catalog = HybridCatalog.load(CATALOG)

    for model in catalog.models.values():
        config = model.configuration("dual-24gb-64k")
        assert config.context == 65_536
        assert config.min_system_ram_mib > sum(config.min_vram_mb_per_card)
        assert config.launch.split_mode in {"layer", "graph"}
        assert config.launch.tensor_split == (1.0, 1.0)


def test_each_system_ram_model_has_64k_128k_262k_and_1m_configurations() -> None:
    catalog = HybridCatalog.load(CATALOG)

    for model in catalog.models.values():
        assert {config.context for config in model.configurations} == {
            65_536, 131_072, 262_144, 1_048_576,
        }
        for config in model.configurations:
            command = config.command(binary="/opt/llama-server", model_path="/models/model.gguf", port=8080)
            assert "--jinja" in command
            if config.context > 65_536:
                assert config.status == "configured-unmeasured"
                assert config.evidence is None
                assert "--no-kv-offload" in command

    laguna_1m = catalog.models["laguna-s2-1-q4-k-m"].configuration("dual-24gb-1m")
    laguna_command = laguna_1m.command(binary="/opt/llama-server", model_path="/models/model.gguf", port=8080)
    assert laguna_command[laguna_command.index("--rope-scaling") + 1] == "yarn"
    assert laguna_command[laguna_command.index("--rope-scale") + 1] == "4"


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
    config = HybridCatalog.load(CATALOG).models["minimax-m3-mxfp4-moe"].configuration("dual-24gb-64k")

    command = config.command(
        binary="/opt/llama-server",
        model_path="/models/MiniMax-M3-00001-of-00007.gguf",
        port=11601,
    )

    assert command[:4] == ("/opt/llama-server", "-m", "/models/MiniMax-M3-00001-of-00007.gguf", "--port")
    assert command[command.index("--n-cpu-moe") + 1] == "56"
    assert command[command.index("--tensor-split") + 1] == "1,1"
    assert "--no-mmap" not in command
    assert "--mlock" not in command


def test_glm_command_contains_required_dsa_cpu_moe_flags() -> None:
    catalog = HybridCatalog.load(CATALOG)
    glm = catalog.models["glm-5-2-2-788bpw"].configuration("dual-24gb-64k")
    command = glm.command(
        binary="/engines/ik/llama-server", model_path="/models/glm.gguf", port=11600
    )

    assert "--cpu-moe" in command
    assert "-dsa" in command
    assert "-fidx" in command
    assert command[command.index("--split-mode") + 1] == "layer"
    assert "--fit" not in command


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
    assert "minimax-m3-mxfp4-moe" in result.stdout
    assert "glm-5-2-2-788bpw" in result.stdout


def test_catalog_rejects_validated_configuration_without_evidence(tmp_path: Path) -> None:
    text = CATALOG.read_text().replace(
        '"sha256:a332e44a601c129b90262c877ad1f62e0d8fd54780dba5f566ee010d51225ec5"',
        "null",
        1,
    )
    path = tmp_path / "catalog.json"
    path.write_text(text)

    with pytest.raises(ValueError, match="validated configuration requires evidence"):
        HybridCatalog.load(path)
