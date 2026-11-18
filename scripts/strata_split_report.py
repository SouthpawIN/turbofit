"""Print the Strata split for a 48 GB VRAM / 377 GiB RAM machine."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from turbofit_runtime.strata_planner import (  # noqa: E402
    GIB,
    Refusal,
    recommend_for_machine,
)

VRAM = 48 * 1024 ** 3
RAM = 377 * GIB

results = recommend_for_machine(VRAM, RAM, None, ["avx2", "avx512f"])
print(f"machine: {VRAM/GIB:.0f} GB VRAM, {RAM/GIB:.0f} GiB RAM, SSD unbounded\n")
header = (
    f"{'family':24} {'quant':22} {'ctx':>7} {'cache':>7} {'kv':>5} {'ple':>5} "
    f"{'vram':>6} {'ram':>6} {'ssd':>6} {'262K':>5} {'speed':>6}"
)
print(header)
print("-" * len(header))
for key, item in results.items():
    if isinstance(item, Refusal):
        print(f"{key:24} REFUSED: {item.reason}")
        continue
    print(
        f"{key[:24]:24} {item.model_id[:22]:22} {item.context_k:>6}K "
        f"{item.expert_cache_slots:>7} {item.kv_mode:>5} {item.ple_io:>5} "
        f"{item.vram_bytes/GIB:>6.1f} {item.ram_bytes/GIB:>6.1f} "
        f"{item.ssd_bytes/GIB:>6.1f} "
        f"{'yes' if item.fits_preferred_context else 'NO':>5} "
        f"{item.predicted_speed:>6.2f}"
    )
print()
for key, item in results.items():
    if isinstance(item, Refusal):
        continue
    print(f"{key}: {' '.join(item.flags)}")
