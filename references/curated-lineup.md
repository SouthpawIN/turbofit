# Curated Model Lineup

Opinionated picks for main + aux roles, local + API. **Good intelligence is always available no matter your hardware.**

## Hardware Tiers

| Tier | VRAM | Setup | Default Main | Default Aux |
|------|------|-------|-------------|-------------|
| **Beefy** | ≥24GB | Local main + local aux | Darwin Reason 28B | Carnice 35A3B |
| **Modest** | 8-24GB | API main + local aux | DeepSeek V4 Pro (API) | Carnice (cpu-moe) |
| **Thin** | <8GB or no GPU | API main + API aux | DeepSeek V4 Pro (free) | Kimi K2.6 (free) |

`serve auto` detects your tier. `--api` forces API mode. `--free` restricts to free endpoints.

## Featured Local Models (7 picks)

| # | Alias | Size | Tier | Speed | Role | Key Feature |
|---|-------|------|------|-------|------|-------------|
| 1 | `darwin-28b-reason` | 28B dense | S | 38 tok/s | Main | Smartest 27B, 1M ctx |
| 2 | `darwin-apex-36b-i-compact` | 36B/3B MoE | S | 107 tok/s | Main | Fastest MoE, NextN |
| 3 | `prism-eagle-27b` | 27B hybrid | S | 121 tok/s | Main | Mamba2+GDN, smallest 27B |
| 4 | `carwin-28b-mtp` | 27B MTP | SF | ~100 tok/s | Main | MTP + multimodal |
| 5 | `qwopus-27b-v2-mtp` | 27B MTP | SF | ~100 tok/s | Main | Speed-optimized MTP |
| 6 | `qwopus-27b-coder-mtp` | 27B MTP | SF | ~100 tok/s | Main | Claude Fable-5 + Kimi traces |
| 7 | `carnice-apex-35a3b-compact` | 35A3B MoE | SD | 30 tok/s | Aux | Always-on, vision, 1M ctx |

## Auxiliary Local

### `carnice-apex-35a3b-compact` ⭐ Aux Choice in All Cases
- **Why:** Always-on 35A3B, vision, MTP, 1M ctx capable
- **VRAM:** ~17 GB (down to ~11 GB with cpu-moe)
- **Spec decoding:** nextn + turbo4-kv
- **tok/s target:** 30 (10 with cpu-moe)
- **Context:** 1M

## API Main (ranked by volume performance — free first)

| Tier | Model | Vision | Cost | Context | Why |
|------|-------|--------|------|---------|-----|
| S | DeepSeek V4 Pro | No | FREE (NIM) | 1M | Best open reasoning + coding |
| S | Kimi K2.6 | Yes | FREE (OpenRouter) | 1M | Strong coding + vision |
| S | GLM 5.1 | No | $0.44/$0.87 | 1M | Agentic + long-horizon |
| S | Qwen 3.7 Max | No | $1.25/$3.75 | 1M | Qwen frontier |
| SF | DeepSeek V4 Flash | No | FREE (NIM) | 1M | Fast reasoning, best value |
| SF | MiniMax M3 | Yes | FREE (NIM) | 1M | Vision + 1M ctx |
| SF | Nemotron Ultra | No | FREE (NIM) | 1M | 550B MoE, reasoning budget |
| F | DeepSeek V4 Flash | No | $0.10/$0.20 | 1M | Cheapest paid main |
| F | Qwen 3.5 Flash | Yes | $0.065/$0.26 | 1M | Cheapest vision main |

## API Aux (ranked by vision > speed > cost — free first)

| Tier | Model | Vision | Cost | Context | Why |
|------|-------|--------|------|---------|-----|
| SD | Kimi K2.6 | Yes | FREE | 1M | Best free aux — vision + reasoning |
| SD | MiniMax M3 | Yes | FREE (NIM) | 1M | Vision + 1M ctx |
| SD | Step 3.7 Flash | Yes | FREE | 1M | Fast, vision |
| SD | DeepSeek V4 Flash | No | FREE (NIM) | 1M | Fast reasoning (pair with vision main) |
| F | Qwen 3.5 Flash | Yes | $0.065/$0.26 | 1M | Cheapest paid vision aux |

## Cascading VRAM Scaling Ladder (Beefy tier)

| Step | State | Main | Aux | Context | Action |
|------|-------|------|-----|---------|--------|
| 1 | Ideal | Darwin 28B @ 1M | Carnice 35A3B @ 1M | 1M | Nothing |
| 2 | Mild pressure | Darwin 28B | Carnice (CPU-moe) | 1M | Offload aux experts to CPU |
| 3 | Moderate | Darwin 28B | Carnice 35A3B | 128K | Drop context of both |
| 4 | High | Darwin 28B | API auto | 128K | Drop Carnice, route aux to API |
| 5 | Critical | Prism Eagle 27B | API auto | 128K | Swap main to small dense |
| 6 | Extreme | Darwin Apex 35A3B | API auto | 64K | Swap main to MoE (3B active) |
| 7 | API-only | DeepSeek V4 Pro (API) | Kimi K2.6 (API) | API | No local serving viable — zero cost fallback |

**Key principles:**
- Main is always protected until Step 5
- MoE expert offload is the first pressure valve (Steps 2, 6)
- Context drops only when absolutely necessary
- API aux kicks in when local aux can't fit
- Step 7 = fully API, zero cost, still strong performance
- Each step preserves maximum intelligence while respecting VRAM

## Benchmark Group: `compare_27b`

The following models should be benchmarked head-to-head at Q4 quantization:

- `qwopus-27b-v2-mtp`
- `qwable-27b-mtp`
- `carwin-28b-mtp`
- `darwin-apex-36b-i-compact`

Run: `serve bench compare_27b` to rank by tok/s and quality.
