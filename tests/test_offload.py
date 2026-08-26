from __future__ import annotations

import pytest

from turbofit_runtime.offload import (
    choose_optimal_offload,
    moe_expert_offload_layers,
    tensor_split_for_layers,
)


def test_tensor_split_maps_exact_layer_count_to_secondary_gpu() -> None:
    assert tensor_split_for_layers(offload_layers=4, total_layers=65) == "4,61"
    assert tensor_split_for_layers(offload_layers=8, total_layers=65) == "8,57"


def test_tensor_split_rejects_invalid_layer_counts() -> None:
    with pytest.raises(ValueError):
        tensor_split_for_layers(offload_layers=0, total_layers=65)
    with pytest.raises(ValueError):
        tensor_split_for_layers(offload_layers=65, total_layers=65)


def test_choose_optimal_requires_success_and_safety_margin_then_prefers_throughput() -> None:
    runs = [
        {"offload_layers": 3, "passed": False, "main_tps": 0.0, "min_free_mb": 0},
        {"offload_layers": 4, "passed": True, "main_tps": 58.0, "min_free_mb": 300},
        {"offload_layers": 6, "passed": True, "main_tps": 55.0, "min_free_mb": 1100},
        {"offload_layers": 8, "passed": True, "main_tps": 57.0, "min_free_mb": 1600},
    ]

    best = choose_optimal_offload(runs, safety_margin_mb=1024)

    assert best["offload_layers"] == 8


def test_choose_optimal_breaks_equal_throughput_ties_with_fewer_offloaded_layers() -> None:
    runs = [
        {"offload_layers": 6, "passed": True, "main_tps": 57.0, "min_free_mb": 1200},
        {"offload_layers": 8, "passed": True, "main_tps": 57.0, "min_free_mb": 1800},
    ]

    best = choose_optimal_offload(runs, safety_margin_mb=1024)

    assert best["offload_layers"] == 6


# --- MoE expert offload: hardware-derived, never recipe-baked -----------------

# Qwen3.8-Flash-Next UD-Q4_K_XL shape, measured: 48 layers, ~6 GiB dense,
# ~2.04 GiB of routed expert weight per layer.
_GIB = 1024**3
_Q4_LAYERS = 48
_Q4_DENSE = 6 * _GIB
_Q4_EXPERT_PER_LAYER = int(2.04 * _GIB)


def _q4(vram_mb: int, host_mb: int) -> int:
    return moe_expert_offload_layers(
        total_layers=_Q4_LAYERS,
        expert_bytes_per_layer=_Q4_EXPERT_PER_LAYER,
        dense_bytes=_Q4_DENSE,
        usable_vram_mb=vram_mb,
        usable_host_mb=host_mb,
    )


def test_moe_offload_scales_with_vram_so_configs_are_not_host_specific() -> None:
    # More VRAM must never mean more host offload.
    small = _q4(24 * 1024, 128 * 1024)
    medium = _q4(48 * 1024, 377 * 1024)
    large = _q4(96 * 1024, 377 * 1024)

    assert small > medium > large
    assert all(0 <= n <= _Q4_LAYERS for n in (small, medium, large))


def test_moe_offload_returns_zero_when_everything_fits_in_vram() -> None:
    assert _q4(192 * 1024, 192 * 1024) == 0


def test_moe_offload_sends_everything_to_host_without_usable_vram() -> None:
    assert (
        moe_expert_offload_layers(
            total_layers=_Q4_LAYERS,
            expert_bytes_per_layer=_Q4_EXPERT_PER_LAYER,
            dense_bytes=_Q4_DENSE,
            usable_vram_mb=0,
            usable_host_mb=377 * 1024,
        )
        == _Q4_LAYERS
    )


def test_moe_offload_rejects_hosts_that_cannot_hold_the_model() -> None:
    with pytest.raises(ValueError):
        _q4(8 * 1024, 32 * 1024)


def test_moe_offload_reserves_headroom_for_compute_buffers() -> None:
    # A reserve large enough to consume the whole card must fall back to host.
    assert (
        moe_expert_offload_layers(
            total_layers=_Q4_LAYERS,
            expert_bytes_per_layer=_Q4_EXPERT_PER_LAYER,
            dense_bytes=_Q4_DENSE,
            usable_vram_mb=24 * 1024,
            usable_host_mb=377 * 1024,
            compute_reserve_mb=24 * 1024,
        )
        == _Q4_LAYERS
    )


def test_moe_offload_validates_inputs() -> None:
    with pytest.raises(ValueError):
        moe_expert_offload_layers(
            total_layers=0,
            expert_bytes_per_layer=1,
            dense_bytes=0,
            usable_vram_mb=1024,
            usable_host_mb=1024,
        )
    with pytest.raises(ValueError):
        moe_expert_offload_layers(
            total_layers=4,
            expert_bytes_per_layer=-1,
            dense_bytes=0,
            usable_vram_mb=1024,
            usable_host_mb=1024,
        )
