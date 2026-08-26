# Turbofit Fit List — Maple small-device evidence

Dedicated VRAM:
- 8 GB: Maple Preview TQ2_0; Ornith 1.5 alternative when host RAM can hold offloaded experts.
- 16 GB: Qwen 3.8 27B Unleashed UD-IQ3_XXS; SuperQwen may replace it only after winning the same physical campaign.
- 24–95 GB: Qwen 3.8 27B Unleashed UD-Q3_K_XL.
- 96 GB+: Qwen 3.8 27B 16-bit until Unleashed FP16 is published.

Integrated / RAM-only total memory:
- 8–15 GB: Maple Preview TQ2_0.
- 16–23 GB: Ornith 1.5 35A3B.
- 24 GB+: Qwen 3.8 27B Unleashed UD-Q3_K_XL.

Measured Maple evidence:
- Artifact: 5,454,482,432 bytes; SHA-256 09d219202562dbd17722dc8e3273527a021182ab7f892c2a06aac459a8f3a090.
- CPU-only 128K inside 8 GiB hard cap, no swap: 5.581 GiB peak; 11.918 tok/s decode.
- CPU-only quality/tool suite: 5.623 GiB peak; 3/3 tool calls; arithmetic pass; 255 cached prompt tokens observed.
- CUDA 128K: 6,115 MiB residency delta; 82.988 tok/s decode; 256.06 tok/s prefill.
- Exact physical 8 GB GPU test is still pending; the CUDA result is a measured-residency surrogate on RTX 3090.
- Maple uses a fork-only runtime and is restricted to native 64K/128K contexts.

Dynamic selection:
- Pressure-aware contraction and healing stay enabled.
- Fit List defines eligible model families; live free capacity and verified runtime evidence choose the active rung.
- No candidate is Auto-promoted from estimates or larger-card evidence.
