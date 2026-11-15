# Model Lineup

Three layers of model, ranked by **measured evidence, not heuristics**.

> Grounded in the product repo: [`curated-lineup.md`](../../../../../turbofit/references/curated-lineup.md), [`model-intel.md`](../../../../../turbofit/references/model-intel.md), [`model-catalog.json`](../../../../../turbofit/references/model-catalog.json). API picks below are June 2026 data.

## 1. Local fit-ladder models
The canonical TurboFit List (the source of truth; see [fit-ladder.md](fit-ladder.md)) is **two ladders that converge on Qwen 3.8 Flash Next** — the only rung where the dense ladder switches to MoE: **GPU-dedicated (dense)** (`Ternary Bonsai 2 27B → Underdog 27B → Qwen 3.8 27B GSQ-RCO → Qwen 3.8 Flash Next`) + **Unified / CPU / Apple Silicon (MoE only, no dense)** (`Maple Preview 20B-A1B → Cyber-Tiel Coder 35B-A3B → Qwen 3.8 Flash Next`, with DeepSeek V4.1 Flash (Strata 1017) on top, engine pending). **MTP on for every rung when available**, and fit selection weighs the **power bill (kWh/hour)** — IQ3_S is lossless, so no heavier quant past it. The live catalog `model-catalog.json` is the **61 models in 13 families** `serve auto` downloads today (full list in the repo README → "Live catalog"); the two ladders above are the curated rungs `serve auto` promotes, the rest of the catalog is Model Zoo download-only.

## 2. API lineup (for Modest / Thin tiers)
Ranked by composite (GPQA + SWE-bench + MMLU + AIME):

| Pick | Model | Why |
|---|---|---|
| Best free main | DeepSeek V4 Pro (NIM) | 90.5% GPQA, 80.6% SWE-bench, 96.4% AIME — free |
| Best paid main | Qwen 3.7 MAX | 93.7% MMLU (#1), 92.4% GPQA |
| Best value main | DeepSeek V4 Flash | 88.1% GPQA, free on NIM |
| Best coding main | DeepSeek V4 Pro | highest open SWE-bench |
| Best free vision aux | MiniMax M3 (NIM) | 69.3% SWE-bench, vision, 1M ctx, free |
| Best paid vision aux | Qwen 3.7 Plus | via Nous for Tool Gateway |

## 3. Named local picks (MRI-Trust merges / MTP grafts)
Opinionated best-pairings from the local catalog:

| Role | Model | GPQA | tok/s | VRAM |
|---|---|---|---|---|
| Smartest dense main | Darwin 28B Reason | 89.39% | 38 | 17GB |
| Fastest main | Prism Eagle 27B (Mamba2+GDN) | ~86% | 121 | 14GB |
| Best MoE main | Darwin Apex 36B | ~87% | 107 | 16GB |
| Smallest aux | Carnice 35A3B | ~83% | 110 | 11GB |

## Pairing rules
1. **Text-only mains MUST pair with a vision aux** — never two blind models.
2. The **aux offsets 40–85% of token cost** (cheap/free aux + paid main = affordable).
3. The **aux needs 262K+ context** to compress the main without losing info.
4. **Free NIM pairs beat everything on cost**: DeepSeek V4 Pro + MiniMax M3.
5. **Route paid API through Nous** for Tool Gateway + 10% credit bonus.
