"""Utilities for measured multi-GPU layer-offload sweeps."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def tensor_split_for_layers(*, offload_layers: int, total_layers: int) -> str:
    """Return a llama.cpp layer split with the offloaded share first."""
    if total_layers < 2:
        raise ValueError("total_layers must be at least 2")
    if not 0 < offload_layers < total_layers:
        raise ValueError("offload_layers must be between 1 and total_layers - 1")
    return f"{offload_layers},{total_layers - offload_layers}"


def choose_optimal_offload(
    runs: Iterable[dict[str, Any]], *, safety_margin_mb: int
) -> dict[str, Any]:
    """Pick the fastest passing run that preserves the requested VRAM margin."""
    eligible = [
        run
        for run in runs
        if run.get("passed")
        and int(run.get("min_free_mb", 0)) >= safety_margin_mb
    ]
    if not eligible:
        raise ValueError("no passing offload run meets the VRAM safety margin")
    return max(
        eligible,
        key=lambda run: (
            float(run.get("main_tps", 0.0)),
            -int(run["offload_layers"]),
        ),
    )


def moe_expert_offload_layers(
    *,
    total_layers: int,
    expert_bytes_per_layer: int,
    dense_bytes: int,
    usable_vram_mb: int,
    usable_host_mb: int,
    compute_reserve_mb: int = 2048,
) -> int:
    """Return how many layers' routed experts must live on the host (``-ncmoe N``).

    Sparse-MoE checkpoints (Qwen4Exp, GLM5Next, ...) are dominated by routed
    expert weights that are only sparsely read per token, so they are the
    cheapest thing to keep in system RAM while attention, shared experts and
    embeddings stay resident on the accelerator.

    This is deliberately hardware-derived rather than recipe-baked: the same
    checkpoint yields a different answer on a 16 GB laptop, a dual-24 GB
    workstation and a 192 GB unified-memory box. Returns 0 when everything
    fits in VRAM, and ``total_layers`` when no expert tensors can be resident.

    Raises ValueError when even the fully-offloaded split cannot fit in host RAM.
    """
    if total_layers < 1:
        raise ValueError("total_layers must be at least 1")
    if expert_bytes_per_layer < 0 or dense_bytes < 0:
        raise ValueError("tensor byte sizes must be non-negative")

    mib = 1024 * 1024
    vram_budget = (usable_vram_mb - compute_reserve_mb) * mib
    host_budget = usable_host_mb * mib

    if vram_budget <= 0:
        # No usable accelerator budget: everything routed goes to the host.
        if total_layers * expert_bytes_per_layer + dense_bytes > host_budget:
            raise ValueError("model does not fit in host memory")
        return total_layers

    # Dense (non-expert) weights are the mandatory accelerator residency cost.
    remaining = vram_budget - dense_bytes
    if remaining < 0:
        if total_layers * expert_bytes_per_layer > host_budget:
            raise ValueError("model does not fit in host memory")
        return total_layers

    if expert_bytes_per_layer == 0:
        return 0

    resident = min(total_layers, remaining // expert_bytes_per_layer)
    offloaded = total_layers - int(resident)
    if offloaded * expert_bytes_per_layer > host_budget:
        raise ValueError("model does not fit in host memory")
    return int(offloaded)
