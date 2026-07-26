from __future__ import annotations

import pytest

from turbofit_runtime.offload import choose_optimal_offload, tensor_split_for_layers


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
