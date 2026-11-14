"""Plan CPU+VRAM expert offload so a model too big for VRAM still runs.

llama.cpp can keep a subset of MoE expert layers in system RAM (`--n-cpu-moe N`)
and leave attention/KV/dense weights on the GPU. Which N to use is a budgeting
question: how much VRAM is actually free, and how much RAM can we afford to
hold, given the model's per-layer expert footprint.

This is deliberately engine-neutral arithmetic so any host can be planned for,
not just the machine it was written on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

MIB = 1024 * 1024


def available_ram_bytes(meminfo: Path | str | None = None) -> int | None:
    """MemAvailable in bytes, or None where the host exposes no /proc."""
    path = Path(meminfo) if meminfo is not None else Path("/proc/meminfo")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


@dataclass(frozen=True)
class OffloadPlan:
    feasible: bool
    n_cpu_moe: int
    expert_layers: int
    vram_bytes: int
    ram_bytes: int
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def plan_expert_offload(
    *,
    total_bytes: int,
    expert_bytes: int,
    expert_layers: int,
    vram_bytes: int,
    ram_bytes: int | None,
    vram_reserve_bytes: int = 1536 * MIB,
    ram_reserve_bytes: int = 2048 * MIB,
) -> OffloadPlan:
    """Choose how many expert layers must live in RAM for this model to fit.

    `expert_bytes` is the total size of all expert tensors; `total_bytes` the
    whole model. Anything that is not experts (attention, KV, embeddings, dense
    FFN, mmproj) is assumed to be VRAM-resident, which is what `--n-cpu-moe`
    implies.
    """
    if expert_layers <= 0 or expert_bytes <= 0:
        raise ValueError("expert_layers and expert_bytes must be positive")
    if vram_bytes <= 0:
        raise ValueError("vram_bytes must be positive")

    usable_vram = max(0, vram_bytes - vram_reserve_bytes)
    non_expert = max(0, total_bytes - expert_bytes)
    per_layer = math.ceil(expert_bytes / expert_layers)

    if non_expert >= usable_vram:
        return OffloadPlan(False, 0, expert_layers, 0, 0,
                           f"non-expert weights alone need {non_expert / 2**30:.1f} GiB "
                           f"but only {usable_vram / 2**30:.1f} GiB of VRAM is usable")

    layers_on_gpu = (usable_vram - non_expert) // per_layer
    # llama.cpp's --n-cpu-moe counts the expert layers held in RAM from the start.
    n_cpu_moe = max(0, expert_layers - int(layers_on_gpu))
    offloaded = n_cpu_moe * per_layer
    gpu_held = non_expert + (expert_layers - n_cpu_moe) * per_layer

    if ram_bytes is None:
        return OffloadPlan(False, n_cpu_moe, expert_layers, gpu_held, offloaded,
                           "system RAM could not be measured, so offloading cannot be planned")
    usable_ram = max(0, ram_bytes - ram_reserve_bytes)
    if offloaded > usable_ram:
        return OffloadPlan(False, n_cpu_moe, expert_layers, gpu_held, offloaded,
                           f"offloading {offloaded / 2**30:.1f} GiB of experts needs more than the "
                           f"{usable_ram / 2**30:.1f} GiB of usable RAM")
    return OffloadPlan(True, n_cpu_moe, expert_layers, gpu_held, offloaded,
                       f"{n_cpu_moe}/{expert_layers} expert layers in RAM "
                       f"({offloaded / 2**30:.1f} GiB), {gpu_held / 2**30:.1f} GiB in VRAM")
