"""Maple-Preview's sliding-window pattern, derived instead of demanded.

Maple is a 3:1 hybrid: three sliding-window-512 attention layers followed by one
full-attention layer, repeating (the MLX reference's ``layer_types`` is
``SSSF`` x6 for 24 layers). The first published GGUF conversions
(``maple-tq2_0.gguf``, ``maple-q2_0.gguf``) shipped **without**
``maple.attention.sliding_window_pattern``: the converter relied on the engine
knowing the ratio. Newer upstream ``src/models/maple.cpp`` reads the array with
``required=true``, so those files die at load time with::

    error loading model: error loading model hyperparameters:
    array key not found in model: maple.attention.sliding_window_pattern

The engine patch in ``runtime-patches/maple-llama.cpp/`` derives the pattern in
C++ (``hparams.set_swa_pattern(4, false)``) and only lets an explicit array
override it. This module is the Python mirror of that decision, so the rule can
be tested and used by doctor/verification paths without a 5 GiB checkpoint.

Semantics match llama.cpp: ``1`` = the layer runs sliding-window attention,
``0`` = the layer runs full attention.
"""

from __future__ import annotations

ARCH = "maple"

#: Full attention every Nth layer, i.e. ``layer_types == SSSF`` repeated.
FULL_ATTENTION_PERIOD = 4

#: Layers in Maple-Preview 20B-A1B.
DEFAULT_LAYERS = 24


def sliding_window_pattern(
    n_layers: int = DEFAULT_LAYERS,
    period: int = FULL_ATTENTION_PERIOD,
) -> list[int]:
    """Return the derived per-layer pattern: ``1`` for SWA layers, ``0`` for full.

    ``period=4`` gives ``[1, 1, 1, 0, 1, 1, 1, 0, ...]``, which is the
    ``SSSF`` layout in the published ``config.json``.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if period <= 0:
        raise ValueError("period must be positive")
    return [0 if (i % period) == period - 1 else 1 for i in range(n_layers)]


def resolve_pattern(
    arch: str,
    present_keys: set[str] | tuple[str, ...],
    *,
    n_layers: int = DEFAULT_LAYERS,
    explicit_pattern: list[int] | None = None,
) -> list[int] | None:
    """Decide the pattern to use for a given GGUF.

    * ``None`` for any architecture other than Maple -- the caller does not own
      that decision.
    * the explicit array when the GGUF carries one (loader value wins),
    * the derived 3:1 pattern when ``maple.attention.sliding_window_pattern``
      is missing, instead of hard-failing.
    """
    if arch != ARCH:
        return None
    if explicit_pattern:
        return list(explicit_pattern)
    if f"{ARCH}.attention.sliding_window_pattern" in present_keys:
        # Present but not decoded here: the loader parses it, we stay out of the way.
        return None
    return sliding_window_pattern(n_layers)


def describe(pattern: list[int]) -> str:
    """``[1,1,1,0,...]`` -> ``SSSFSSSF...`` for logs and doctor output."""
    return "".join("S" if value else "F" for value in pattern)
