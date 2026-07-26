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
