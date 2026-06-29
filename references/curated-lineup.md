# Curated Model Lineup

Opinionated picks for main + aux roles, local + API. **Ranked by actual benchmark scores, not heuristics.**

## API Model Leaderboard (by benchmark scores — June 2026)

### Main API — text-only mains (must pair with vision aux)

Ranked by composite score (GPQA + SWE-bench + MMLU + AIME weighted equally):

| Rank | Model | MMLU | GPQA Diamond | SWE-bench Verified | SWE-bench Pro | AIME 2026 | Cost (in/out) | Context | Through Nous? |
|------|-------|------|-------------|-------------------|---------------|-----------|---------------|---------|---------------|
| 1 | **Qwen 3.7 MAX** | 93.7% (#1) | 92.4% | — | 60.6% | — | $1.25/$3.75 | 1M | ✅ `qwen/qwen3.7-max` |
| 2 | **DeepSeek V4 Pro** | 90.2% | 90.5% | 80.6% | 58.6% | 96.4% | FREE (NIM) / $0.435/$0.87 | 1M | ✅ `deepseek/deepseek-v4-pro` |
| 3 | **GLM 5.2** | — | ~87% | 72.8% | — | — | $0.95/$3.00 | 1M | ✅ `z-ai/glm-5.2` |
| 4 | **DeepSeek V4 Flash** | 87.8% | 88.1% | 79.0% | — | — | FREE (NIM) / $0.09/$0.18 | 1M | ✅ `deepseek/deepseek-v4-flash` |
| 5 | **Mimo V2.5 Pro** | — | — | — | — | — | ~$0.43/$0.87 | 1M | ✅ `xiaomi/mimo-v2.5-pro` |

**Pick logic:**
- **Best free main**: DeepSeek V4 Pro (GPQA 90.5%, SWE-bench 80.6%, AIME 96.4% — all FREE on NIM)
- **Best paid main**: Qwen 3.7 MAX (MMLU #1 globally at 93.7%, GPQA 92.4%, HMMT 97.1%)
- **Best value main**: DeepSeek V4 Flash (88.1% GPQA, 79.0% SWE-bench — free on NIM, cheapest paid at $0.09/$0.18)
- **Best coding main**: DeepSeek V4 Pro (80.6% SWE-bench Verified, highest open model)
- **Best agentic main**: Qwen 3.7 MAX (agent-focused, SWE-bench Pro 60.6%, SWE-bench Multilingual 78.3%)

### Vision-capable models (main or aux)

| Rank | Model | GPQA | SWE-bench | Vision | Cost | Context | Through Nous? |
|------|-------|------|-----------|--------|------|---------|---------------|
| 1 | **Kimi K2.6** | 90.5% | — | ✅ | $0.60/$2.50 | 1M | ✅ |
| 2 | **MiniMax M3** | 62.5% | 69.3% (Verified) / 59.0% (Pro) | ✅ | FREE (NIM) / ~$0.30/$1.20 | 1M | ✅ |
| 3 | **Qwen 3.7 Plus** | — | — | ✅ | $0.32/$1.28 | 1M | ✅ |
| 4 | **Mimo V2.5** | — | — | ✅ | $0.105/$0.28 | 1M | ✅ |
| 5 | **Qwen 3.6 Plus** | — | — | ✅ | FREE (OR) | 1M | ❌ OR only |
| 6 | **Qwen 3.5 Flash** | — | — | ✅ | $0.065/$0.26 | 1M | ✅ |

**Pick logic:**
- **Best free vision aux**: MiniMax M3 (69.3% SWE-bench, vision, 1M ctx — FREE on NIM)
- **Best free vision (OR)**: Qwen 3.6 Plus (FREE on OR preview, 1M ctx, vision — but NOT through Nous, no Tool Gateway)
- **Best paid vision aux**: Qwen 3.7 Plus ($0.32/$1.28, through Nous for Tool Gateway + 10% bonus)
- **Best coding vision**: Kimi K2.6 (90.5% GPQA, vision, 1M ctx)

### API Aux — the aux role is three things, not just "the other model"

The auxiliary model handles **three critical jobs** that make or break the fleet:

1. **Cost offset (40-85% of all tokens)**: Hermes routes vision, web extraction, compression, skills hub, and other auxiliary tasks to the aux model. A paid main at $1.25/$3.75 becomes affordable when 40-85% of the traffic hits a free or cheap aux instead. Without this offset, a paid main alone would burn through a monthly budget in days.

2. **Vision injection**: Text-only mains (DeepSeek V4 Pro/Flash, Qwen 3.7 MAX, GLM 5.2, Mimo V2.5 Pro) have zero image understanding. The aux provides vision — screenshot analysis, image generation review, OCR, browser automation screenshots. Without a vision-capable aux, the fleet is blind.

3. **Context compression**: When the main's context fills up, the aux compresses it — summarizing prior turns so the main can keep going with full intelligence. The aux needs enough context window to hold the compressed conversation (262K minimum, 1M ideal) so compression doesn't lose information.

Ranked by all three roles (vision required, cost minimized, context ≥262K):

| Rank | Model | Vision | Cost | Context | SWE-bench | Why |
|------|-------|--------|------|---------|-----------|-----|
| 1 | **MiniMax M3** | ✅ | FREE (NIM) | 1M | 69.3% | Free, vision, 1M ctx, strong reasoning. Best all-around aux. Offsets 40-85% of main cost to $0. |
| 2 | **Qwen 3.6 Plus** | ✅ | FREE (OR) | 1M | — | Free on OR, vision, 1M ctx. No Nous = no Tool Gateway. Good for compression + vision, but misses Tool Gateway routing. |
| 3 | **Qwen 3.7 Plus** | ✅ | $0.32/$1.28 | 1M | — | Through Nous for Tool Gateway + 10% credit bonus. Best paid aux — cheap enough to offset a premium main. |
| 4 | **Mimo V2.5** | ✅ | $0.105/$0.28 | 1M | — | Cheapest paid vision through Nous. Good for budget-constrained setups that need Tool Gateway. |
| 5 | **Qwen 3.5 Flash** | ✅ | $0.065/$0.26 | 1M | — | Cheapest overall. Weakest reasoning — acceptable for vision + compression but don't expect quality aux tasks. |

## Hardware Tiers

| Tier | VRAM | Setup | Default Main | Default Aux | Gateway |
|------|------|-------|-------------|-------------|---------|
| **Beefy** | ≥24GB | Local main + local aux | 27-28B dense (Q4) | 35B MoE 3B-active | N/A (local) |
| **Modest** | 8-24GB | API main + free/cheap aux | DeepSeek V4 Pro (API, FREE) | MiniMax M3 (NIM, FREE) | ⚪ NIM |
| **Thin** | <8GB or no GPU | API main + API aux | DeepSeek V4 Pro (API, FREE) | MiniMax M3 (NIM, FREE) | ⚪ NIM |

`serve auto` detects your tier. `--api` forces API mode. `--free` restricts to free endpoints.

## Local Model Recommendations (by archetype + benchmark)

### Main candidates (Tier S + SF only)

| Archetype | Model | GPQA | tok/s | VRAM | Why It's Recommended |
|-----------|-------|------|-------|------|---------------------|
| 27-28B dense (reasoning) | Darwin 28B Reason | 89.39% | 38 | 17 GB | Smartest local dense model. MRI-Trust merge. Best for deep reasoning tasks. |
| 27B hybrid/Mamba | Prism Eagle 27B | ~86% | 121 | 14 GB | Fastest local model. Mamba2+GDN. Best when speed matters more than peak reasoning. |
| 36B MoE (3B active) | Darwin Apex 36B | ~87% | 107 | 16 GB | Best MoE main. NextN speculative decoding. 3B active params = fast inference. |
| 27B dense + MTP | Carwin 28B MTP | ~84% | 100 | 17 GB | Multimodal MTP. Good balance of speed + reasoning. |
| 27B dense (coder) | Qwopus 27B Coder MTP | ~84% | 100 | 17 GB | Coding-optimized MTP. Best for SWE-bench-style tasks locally. |
| 27B dense (speed) | Qwopus 27B v2 MTP | ~84% | 100 | 17 GB | Speed-optimized. Native MTP graft for 1.6x speedup. |
| 35B MoE (3B active) | Carnice 35A3B | ~83% | 110 | 11 GB | Smallest VRAM in tier. Always-on aux + vision. 1M ctx. |

### Aux candidates (Tier F + C — auxiliary only)

| Archetype | Model | tok/s | VRAM | Role |
|-----------|-------|-------|------|------|
| 27B fine-tune (MTP) | Qwable 27B MTP | 100 | 17 GB | Vision aux, compression |
| 27B uncensored (MTP) | Qwable/Qwopus Abliterated | 100 | 17-20 GB | Uncensored aux |
| 35B stock MoE | Qwen3.5-35B-A3B | 40 | 22 GB | Fallback MoE |
| Small multimodal | Qwen2.5-Omni-3B | 30 | 3 GB | Ultra-light aux for Modest tier |

## Recommended Pairings (based on benchmarks)

### Beefy Tier (≥24GB VRAM) — Best Local Setup

| Main | Aux | Main GPQA | Aux Vision | Total VRAM | Notes |
|------|-----|-----------|------------|------------|-------|
| **Darwin 28B Reason** | **Carnice 35A3B** | 89.39% | ✅ | 28 GB | Smartest main + smallest aux. Best overall pairing. |
| Prism Eagle 27B | Carnice 35A3B | ~86% | ✅ | 25 GB | Fastest main (121 tok/s) + smallest aux. |
| Darwin Apex 36B | Carnice 35A3B | ~87% | ✅ | 27 GB | Best MoE main + best MoE aux. NextN on both. |

### Modest Tier (8-24GB) — API + Free Aux

| Main | Aux | Main GPQA | Cost | Gateway | Notes |
|------|-----|-----------|------|---------|-------|
| **DeepSeek V4 Pro (NIM)** | **MiniMax M3 (NIM)** | 90.5% | FREE | ⚪ NIM | Zero cost. Best reasoning free. Vision aux free. |
| DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | 88.1% | FREE | ⚪ NIM | Zero cost. Faster main, slightly less reasoning. |
| DeepSeek V4 Pro (Nous) | Qwen 3.7 Plus (Nous) | 90.5% | Paid | 🟢 NOUS+TG | Tool Gateway active. 10% credit bonus. Best quality. |

### Thin Tier (<8GB or no GPU) — Zero Cost

| Main | Aux | Main GPQA | Cost | Gateway | Notes |
|------|-----|-----------|------|---------|-------|
| **DeepSeek V4 Pro (NIM)** | **MiniMax M3 (NIM)** | 90.5% | FREE | ⚪ NIM | Same as Modest. Zero cost, 1M ctx. |
| DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | 88.1% | FREE | ⚪ NIM | Slightly faster, still free. |

## Pairing Rules

1. **Text-only mains MUST pair with vision aux.** DeepSeek V4 Pro/Flash, Qwen 3.7 MAX, GLM 5.2, Mimo V2.5 Pro → need vision aux. Without it, the fleet can't see images, screenshots, or browser output.
2. **The aux offsets 40-85% of token cost.** A paid main at $1.25/$3.75 is only affordable when most traffic routes to a free or cheap aux. Always pick the cheapest aux that has vision + 262K+ context.
3. **The aux must have 262K+ context for compression.** When the main's context fills, the aux compresses it. If the aux has less than 262K context, compression loses information and the main degrades.
4. **Free NIM pairings beat everything on cost.** DeepSeek V4 Pro + MiniMax M3 = zero cost, 90.5% GPQA main, 69.3% SWE-bench aux, vision, 1M ctx on both. Unbeatable for free.
5. **Route through Nous for Tool Gateway.** When using paid API models, route through Nous (not direct OpenRouter) for Firecrawl + FAL + OpenAI TTS + Browser Use + 10% credit bonus.
6. **Qwen 3.7 MAX is the MMLU king.** 93.7% MMLU — #1 globally. Best for knowledge-heavy tasks. Pair with MiniMax M3 (free vision + compression).
7. **DeepSeek V4 Pro is the SWE-bench king.** 80.6% SWE-bench Verified — highest open model. Best for coding. Pair with MiniMax M3 (free vision + compression).
8. **Qwen 3.7 MAX is the agentic king.** SWE-bench Pro 60.6%, SWE-bench Multilingual 78.3%, HMMT 97.1%. Best for long-horizon agent tasks. Pair with Qwen 3.7 Plus (same family, Tool Gateway, vision).
9. **Never pair two text-only models.** If the main has no vision, the aux MUST have vision. DeepSeek V4 Pro + DeepSeek V4 Flash = both blind = broken fleet. Always prioritize abilities and cost over family matching.

## Key Principles

- Main is always protected until Step 5 (Beefy scaling ladder)
- MoE expert offload is the first pressure valve
- Context drops only when absolutely necessary
- API aux kicks in when local aux can't fit
- Each step preserves maximum intelligence while respecting VRAM
- **Zero-cost full Hermes-Agent capability is available to anyone regardless of hardware**
