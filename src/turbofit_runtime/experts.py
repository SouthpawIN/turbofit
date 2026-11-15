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

from .gguf_footprint import ExpertFootprint, GGUFFormatError, measure_expert_footprint

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


@dataclass(frozen=True)
class ModelOffloadDecision:
    """A plan plus the measurements it was made from, for evidence and logs."""

    plan: OffloadPlan
    footprint: ExpertFootprint | None
    vram_bytes: int
    ram_bytes: int | None

    def as_dict(self) -> dict:
        return {
            "feasible": self.plan.feasible,
            "n_cpu_moe": self.plan.n_cpu_moe,
            "expert_layers": self.plan.expert_layers,
            "vram_bytes": self.vram_bytes,
            "ram_bytes": self.ram_bytes,
            "reason": self.plan.reason,
            "footprint": self.footprint.as_dict() if self.footprint is not None else None,
        }


def plan_tensor_split(
    footprint: ExpertFootprint, *, n_cpu_moe: int, device_count: int
) -> str | None:
    """Balance a llama.cpp layer split by measured bytes instead of layer count.

    With `--n-cpu-moe N` the first N layers' experts move to RAM, so the
    GPU-resident weights pile up in the later layers. An even layer split then
    leaves one card holding far more than its share (and OOMing while the other
    card idles).

    The cut points are chosen to minimise the heaviest device's share, over
    contiguous layer ranges (llama.cpp's layer split is contiguous), with the
    non-layer tensors counted against the first device, which is where llama.cpp
    places the output/embedding weights.
    """
    if device_count < 2 or not footprint.layers:
        return None
    weights = [
        layer.other_bytes + (layer.expert_bytes if layer.index >= n_cpu_moe else 0)
        for layer in footprint.layers
    ]
    non_layer = max(0, footprint.total_bytes - sum(layer.total_bytes for layer in footprint.layers))
    if sum(weights) <= 0:
        return None

    count = len(weights)
    devices = min(device_count, count)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    # dp[k][i] = best (max segment, cut list) for the first i layers over k devices.
    def segment(start: int, end: int) -> int:
        load = prefix[end] - prefix[start]
        return load + non_layer if start == 0 else load

    best: list[list[tuple[int, tuple[int, ...]]]] = [
        [(2**62, ()) for _ in range(count + 1)] for _ in range(devices + 1)
    ]
    best[0][0] = (0, ())
    for k in range(1, devices + 1):
        for end in range(1, count + 1):
            for start in range(k - 1, end):
                previous, cuts = best[k - 1][start]
                if previous >= 2**62:
                    continue
                load = max(previous, segment(start, end))
                if load < best[k][end][0]:
                    best[k][end] = (load, cuts + (end,))
    load, cuts = best[devices][count]
    if load >= 2**62:
        return None
    sizes = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
    if any(size <= 0 for size in sizes):
        return None
    return ",".join(str(size) for size in sizes)


def plan_model_offload(
    model_path: str | Path,
    *,
    vram_bytes: int,
    ram_bytes: int | None,
    footprint: ExpertFootprint | None = None,
    cache_dir: str | Path | None = None,
    vram_reserve_bytes: int = 1536 * MIB,
    ram_reserve_bytes: int = 2048 * MIB,
) -> ModelOffloadDecision:
    """Plan offload for `model_path` from its own GGUF header and host capacity.

    Raises `GGUFFormatError`/`OSError` when the artifact cannot be measured; the
    caller decides whether that is fatal (it is not, for a model that has not
    been downloaded yet) or must be surfaced.
    """
    if footprint is None:
        footprint = measure_expert_footprint(model_path, cache_dir=cache_dir)
    plan = plan_expert_offload(
        total_bytes=footprint.total_bytes,
        expert_bytes=footprint.expert_bytes,
        expert_layers=footprint.expert_layers,
        vram_bytes=vram_bytes,
        ram_bytes=ram_bytes,
        vram_reserve_bytes=vram_reserve_bytes,
        ram_reserve_bytes=ram_reserve_bytes,
    )
    return ModelOffloadDecision(plan=plan, footprint=footprint, vram_bytes=vram_bytes,
                                ram_bytes=ram_bytes)
