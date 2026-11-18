"""Flint and LFM2.5 reach low tiers by streaming experts, not by needing a big card.

The tier report already places both families on 8-16 GB bands through measured
expert offload. The recipes did not ask for it, so the launcher would have put
every expert on the GPU and failed on exactly the machines the report promised.
These tests pin the two halves together: the placement arithmetic and the flag
the launcher actually emits.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime import tier_fit
from turbofit_runtime.gguf_footprint import ExpertFootprint
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.recipes import RecipeBook

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
RECIPES = ROOT / "references" / "model-recipes.json"
FOOTPRINTS = ROOT / "references" / "model-footprints.json"

FAMILIES = ("LFM2.5 24B-A2B", "Flint 40A2.5B")


def _footprints() -> dict:
    data = json.loads(FOOTPRINTS.read_text())
    return data.get("models") or data


@pytest.mark.parametrize("family", FAMILIES)
def test_recipe_asks_for_automatic_expert_offload(family: str) -> None:
    spec = json.loads(RECIPES.read_text())["models"][family]
    assert spec.get("auto_cpu_moe") is True, (
        f"{family} is placed by offload in the tier report, so its recipe must "
        "ask the launcher to offload too"
    )


@pytest.mark.parametrize("model_id,vram", [
    ("lfm2-5-24b-a2b-q4", 12),
    ("flint-40a2-5b-mix12g", 12),
])
def test_low_tier_placement_is_offload_not_refusal(model_id: str, vram: int) -> None:
    record = _footprints()[model_id]
    placement = tier_fit.placement_for(model_id, record, 65_536, vram, 64 * GIB)
    assert placement is not None
    assert placement.placement == "offload"
    assert placement.n_cpu_moe > 0


def _book_with_probes(family: str, *, vram_gib: int, expert_gib: float,
                      total_gib: float, layers: int) -> RecipeBook:
    total = int(total_gib * GIB)
    expert = int(expert_gib * GIB)
    fp = ExpertFootprint(
        total_bytes=total, expert_bytes=expert, expert_layers=layers,
        dense_bytes=total - expert, tensor_count=layers * 3,
        expert_tensor_count=layers * 3,
        expert_tensor_names=tuple(f"blk.{i}.ffn_gate_exps.weight" for i in range(layers)),
        shards=("model.gguf",),
    )
    data = json.loads(RECIPES.read_text())
    fingerprint = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=65_536,
        devices=(AcceleratorDevice(
            index=0, uuid="GPU-0", name="Test", vendor="nvidia", backend="cuda",
            memory_total_mb=vram_gib * 1024, compute_capability="8.6", bus_id=None,
        ),),
    )
    return RecipeBook(
        data, backend_name="cuda", platform_name="linux", hardware=fingerprint,
        vram_probe=lambda: vram_gib * GIB,
        ram_probe=lambda: 64 * GIB,
        footprint_probe=lambda path: fp,
    )


@pytest.mark.parametrize("family,total,expert,layers", [
    ("LFM2.5 24B-A2B", 13.5, 12.9, 38),
    ("Flint 40A2.5B", 22.5, 21.3, 47),
])
def test_launcher_emits_n_cpu_moe_on_a_12gb_card(
    family: str, total: float, expert: float, layers: int,
) -> None:
    book = _book_with_probes(family, vram_gib=12, expert_gib=expert,
                             total_gib=total, layers=layers)
    component = book.resolve_component(
        family, role="main", gpu="0", port=11605, context=65_536,
    )
    command = list(component.command)
    assert "--n-cpu-moe" in command, f"{family} did not offload on a 12 GB card"
    assert int(command[command.index("--n-cpu-moe") + 1]) > 0
