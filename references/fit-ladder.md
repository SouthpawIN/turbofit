# Fit Ladder — the TurboFit List (canonical)

> **The Fit Ladder is the source of truth for the UI.** The canonical Fit lives here (mirrored from `wiki/index.md` § TurboFit List): **two ladders that converge on Flash Next**, ordered by capability. `serve auto` builds the fit chain from these ladders, picks the largest sustainable rung, and walks down the chain on OOM/timeout — **the Fit List IS the fallback chain** (the adaptive fit chain).

## Fit calculation

```
Required: model_size + KV_cache ≤ tier_size
KV_cache ≈ 2-4 GB at 262K context (q8_0 KV)
Comfortable: model_size + 2 GB ≤ tier_size
```

## The TurboFit List — 5 models, ordered by capability

| # | Model | Type | Params | Optimal Quant | Size | AIME26 | GPQA-D | DeepSWE | LCBv6 | TB4.0 |
|---|-------|------|--------|---------------|------|--------|--------|---------|-------|-------|
| 1 | **Maple Preview 20B-A1B** | Ternary MoE | 20B-A1B | TQ2_0-head-Q4_K | 5.5 GiB | 87.5% | 73.5% | — | 75.1% | — |
| 2 | **Cyber-Tiel Coder 35B-A3B** | Code LLM (MoE) | 35B-A3B | UD-Q4_K_XL | 22.4 GiB | — | — | — | — | — |
| 3 | **Qwen 3.8 27B GSQ-RCO** | LLM (quant) | 27B | IQ3_S | 11.8 GiB | — | 89.2% | 42.2% | 90.3% | — |
| 4 | **Qwen 3.8 Flash Next** | MoE (Qwen4 Preview) | 180B / 6B | IQ3_XXS | 47 GiB | — | 91.7% | 58.7% | 91.9% | — |
| 5 | **DeepSeek V4.1 Flash** | MoE | 763B / 16B | FP8 / UD-Q4_K_XL | 160 / ~40 GiB | — | 90.9% | 74.2% | — | 31.2% |

> GSQ-RCO IQ3_S benchmarks (AIME25 100%, GPQA-D 89.2%, DeepSWE 42.2%, LCBv6 90.3%) are from the model card; Maple LCBv6 is LiveCodeBench v6.

## The two ladders (converge on Flash Next)

Dense does not run well on CPU / Unified memory, so the tiers split into two ladders:
- **Ladder 1 — Dedicated (GPU / VRAM):** dense is fine — `Ternary Bonsai 2 27B → Underdog 27B → Qwen 3.8 27B GSQ-RCO → Flash Next`.
- **Ladder 2 — Unified (CPU / Apple Silicon):** MoE only, no dense — `Maple Preview 20B-A1B → Cyber-Tiel Coder 35B-A3B (in place of Qwen 3.8 27B GSQ-RCO) → Flash Next → DeepSeek V4.1` (DeepSeek V4.1 is the top rung, **above** Flash Next).

## Cross-cutting rules (every rung)
- **MTP on whenever we can** — every one of these models has MTP (multi-token prediction) available in some form; turn it on at each rung.
- **Mind the power bill (kWh / hour).** IQ3_S is lossless, so there's no reason to push a heavier quant (and a higher power draw) past it — 32 GB and 48 GB both stay on Qwen 3.8 27B GSQ-RCO @ IQ3_S.

## GPU (NVIDIA/AMD VRAM — ≤16 GB assumes system RAM available for MoE offload + KV)

| Tier | Model | Quant | Size | VRAM Used | Context | Notes |
|------|-------|-------|------|-----------|---------|-------|
| 8-11 GB | Ternary Bonsai 2 27B | PTQ1_0 / PQ2_0 | ~8-9 GiB | ~9 GiB | 131K | Dense ternary GGUF; prism-llama.cpp; MTP on |
| 12-15 GB | Underdog 27B | PQ2_0 | ~12-13 GiB | ~13 GiB | 131K | Dense ternary GGUF; prism-llama.cpp; MTP on |
| 16 GB | Qwen 3.8 27B GSQ-RCO | IQ3_S | 11.8 GiB | ~12 GiB | 131K | Near-lossless; MTP on; Cyber-Tiel Coder 35B-A3B UD-IQ3_XXS also fits |
| 24 GB | Qwen 3.8 27B GSQ-RCO | IQ3_S | 11.8 GiB | ~12 GiB | 262K | More KV headroom; MTP on |
| 32 GB | Qwen 3.8 27B GSQ-RCO | IQ3_S (lossless) | 11.8 GiB | ~12 GiB | 262K+ | Lossless — don't up the power bill past it; MTP on |
| 48 GB | Qwen 3.8 27B GSQ-RCO | IQ3_S (lossless) + MTP | 11.8 GiB | ~12 GiB | 262K+ | Same IQ3_S as 32 GB, MTP on |
| 64 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native | MoE; both ladders converge (Strata on 2×3090) |
| 96 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native, 1M w/ YaRN | Qwen4 preview, beats Claude Opus 4.6 |
| 128 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 1M | Max context for Flash Next |

> IQ3_S = near-lossless, recommended where it fits. Only move to FP16 at 64 GB+. Flash Next QSA gives ~4.9× decode speedup vs dense attention at 262 K; IQ3_XXS is the sweet spot for 64 GB+.

## CPU (system RAM — model + KV must fit in RAM)

| Tier | Model | Quant | Size | RAM Used | Context | Notes |
|------|-------|-------|------|----------|---------|-------|
| 8-15 GB | Maple Preview 20B-A1B | TQ2_0 | 5.5 GiB | ~6 GiB | 131K native | MoE; stands in for Ternary Bonsai 2 27B / Underdog (dense) |
| 16-23 GB | Cyber-Tiel Coder 35B-A3B | UD-IQ3_XXS | — | — | 262K native | MoE, 3B active; faster than Qwen 3.8 27B GSQ-RCO on CPU |
| 24-31 GB | Cyber-Tiel Coder 35B-A3B | UD-Q4_K_XL | 22.4 GiB | ~23 GiB | 262K native | Fast on CPU |
| 32-47 GB | Cyber-Tiel Coder 35B-A3B | UD-Q5_K_XL | — | — | 262K native | Fast on CPU |
| 48-63 GB | Cyber-Tiel Coder 35B-A3B | UD-Q6_K_XL | — | — | 262K+ | More KV headroom |
| 64 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native | Both ladders converge (Cyber-Tiel Coder 35B-A3B stood in for Qwen 3.8 27B GSQ-RCO) |
| 96 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native | Flash Next entry tier |
| 128 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 1M w/ YaRN | Max context |
| top | DeepSeek V4.1 Flash | FP8 / UD-Q4_K_XL | large (MoE) | — | 181K | 763B MoE — top rung, **above** Flash Next; 1017 update, Strata engine; pending: does it fit the target tier? |

## Unified memory (Apple Silicon — shared CPU/GPU RAM)

| Tier | Model | Quant | Size | RAM Used | Context | Notes |
|------|-------|-------|------|----------|---------|-------|
| 8-15 GB | Maple Preview 20B-A1B | TQ2_0 | 5.5 GiB | ~6 GiB | 131K native | MoE; 218 tok/s on M4; stands in for Ternary Bonsai 2 27B / Underdog |
| 16-23 GB | Cyber-Tiel Coder 35B-A3B | UD-IQ3_XXS | — | — | 262K native | MoE, 3B active; faster than Qwen 3.8 27B GSQ-RCO |
| 24-31 GB | Cyber-Tiel Coder 35B-A3B | UD-Q4_K_XL | 22.4 GiB | ~23 GiB | 262K native | 22+4 = 26 GB ✓ |
| 32-47 GB | Cyber-Tiel Coder 35B-A3B | UD-Q5_K_XL | — | — | 262K native | Fast on CPU |
| 48-63 GB | Cyber-Tiel Coder 35B-A3B | UD-Q6_K_XL | — | — | 262K+ | M2 Max/Ultra territory |
| 64 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native | Candidate-only on Apple — no macOS engine yet |
| 96 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 262K native | M4 Ultra |
| 128 GB | Qwen 3.8 Flash Next | IQ3_XXS | 47 GiB | ~48 GiB | 1M w/ YaRN | Max Flash Next |
| top | DeepSeek V4.1 Flash | FP8 / UD-Q4_K_XL | large (MoE) | — | 181K | 763B MoE — top rung, **above** Flash Next; 1017 update, Strata engine |

> ★ = on the TurboFit List. Models without ★ are Model Zoo only.

## Live catalog (86 models, 31 families)

The currently-automated, downloadable set (`model-catalog.json`) — **86 models in 31 families**: `Ternary Bonsai 2 27B` (3 variants), `Underdog 27B`, `Qwen 3.8 27B` (11, incl. GSQ-RCO), `Qwen 3.8 Flash Next`, `Maple Preview 20B-A1B`, `Cyber-Tiel Coder 35B-A3B` (5), `Ternary Bonsai 27B` (12), `Bonsai 27B` (18), `Carwin Nano` (2), `GLM 5.2`, `Laguna S2.1` (2), `MiniMax M3`, `Darwin 180B RSI`. The two ladders above are the curated rungs `serve auto` promotes; the rest of the catalog is Model Zoo download-only.

## Recovery order (when VRAM/RAM pressure rises)

The main model is protected until the last step · MoE expert offload is the first pressure valve · context drops only when absolutely necessary · **aux hard-drive streaming is available at every tier** · API aux kicks in only when even streamed aux can't serve.

1. **Shrink context** — last-resort, least-preferred.
2. **MoE expert offload to RAM** — the first pressure valve.
3. **Aux → hard drive, at every tier** — when the auxiliary model no longer fits in VRAM/RAM, it streams from the hard drive (SSD-PLE / disk tiering). This is **not** limited to the bottom rung: any tier whose pressure budget can't hold the aux moves the aux to disk streaming. The aux stays live and role-separated — it just reads from the drive.
4. **API aux** — only when a streamed local aux can't keep up (or no aux artifact exists at the tier).
5. **Down the main ladder** — the main model is protected until every step above is exhausted.

At the bottom rung (**Ternary Bonsai 2 27B** — the "kick-a-bite" tier), the squeeze is maximum: **both** main and auxiliary run squeezed, with the main on hard-drive SSD-PLE streaming and the auxiliary streaming from disk alongside it. That rung exists to prove owned-process drain, pressure contraction, recovery, and rollback under full squeeze.

> Full model detail: [`wiki/models/`](../wiki/models/) (one page per model) · Fit UI: [`wiki/models/hardware-tiers.html`](../wiki/models/hardware-tiers.html) · [`wiki/models/leaderboard.html`](../wiki/models/leaderboard.html) · Canonical: [`fit-lists.json`](https://github.com/SouthpawIN/turbofit/blob/main/fit-lists.json) · [`model-catalog.json`](https://github.com/SouthpawIN/turbofit/blob/main/model-catalog.json)
