# API Model Rankings (2026-06-25)

Curated API models ranked by quality. Prices in $/M tokens (input/output).

## Main API Models (Text-Only — must pair with vision aux)

| Rank | Model | Provider | Input $ | Output $ | Vision | Context | Tier | Through Nous? |
|------|-------|----------|---------|----------|--------|---------|------|---------------|
| 1 | GLM 5.2 | Z.AI / OpenRouter / Nous | $0.95 (OR) / $1.40 (Z.AI) | $3.00 (OR) / $4.40 (Z.AI) | ❌ | 1M | S | ✅ `z-ai/glm-5.2` |
| 2 | Qwen 3.7 MAX | DashScope / OpenRouter / Nous | $1.25 | $3.75 | ❌ | 1M | S | ✅ `qwen/qwen3.7-max` |
| 3 | DeepSeek V4 Pro | DeepSeek / OpenRouter / NIM (free) / Nous | $0.435 (DS) / $1.74 (OR) / FREE (NIM) | $0.87 (DS) / $3.48 (OR) / FREE (NIM) | ❌ | 1M | S | ✅ `deepseek/deepseek-v4-pro` |
| 4 | DeepSeek V4 Flash | DeepSeek / OpenRouter / NIM (free) / Nous | $0.09 (OR) / $0.14 (DS) / FREE (NIM) | $0.18 (OR) / $0.28 (DS) / FREE (NIM) | ❌ | 1M | SF | ✅ `deepseek/deepseek-v4-flash` |
| 5 | Mimo V2.5 Pro | Xiaomi / OpenRouter / Nous | ~$1.00 | ~$3.00 | ❌ | 1M | SF | ✅ `xiaomi/mimo-v2.5-pro` |

## Main API Models (Vision-capable — can pair with any aux)

| Rank | Model | Provider | Input $ | Output $ | Vision | Context | Tier | Through Nous? |
|------|-------|----------|---------|----------|--------|---------|------|---------------|
| 1 | MiniMax M3 | NIM (free) / OpenRouter / Nous | ~$0.30 / FREE (NIM) | ~$1.20 / FREE (NIM) | ✅ | 1M | SF | ✅ `minimaxai/minimax-m3` |
| 2 | Qwen 3.7 Plus | DashScope / OpenRouter / Fireworks / Nous | $0.32 (OR) / $0.50 (FW) | $1.28 (OR) / $3.00 (FW) | ✅ | 1M | SF | ✅ `qwen/qwen3.7-plus` |
| 3 | Mimo V2.5 | Xiaomi / OpenRouter / Nous | $0.105 | $0.28 | ✅ | 1M | F | ✅ `xiaomi/mimo-v2.5` |
| 4 | Kimi K2.7 Code | OpenRouter / Moonshot / Nous | $0.74 (OR) | $3.50 (OR) | ✅ | 256K | SF | ✅ `moonshotai/kimi-k2.7-code` |
| 5 | Kimi K2.6 | OpenRouter / Moonshot / Nous | $0.60 | $2.50 | ✅ | 1M | SF | ✅ `moonshotai/kimi-k2.6` |
| 6 | Qwen 3.6 Plus | OpenRouter (free preview) | FREE | FREE | ✅ | 1M | F | ❌ Not through Nous |
| 7 | Qwen 3.6 Flash | OpenRouter / Nous | $0.1875 | $1.125 | ✅ | 262K | F | ✅ `qwen/qwen3.6-flash` |
| 8 | Qwen 3.5 Flash | OpenRouter / Nous | $0.065 | $0.26 | ✅ | 1M | SD | ✅ `qwen/qwen3.5-flash-02-23` |

## Aux API Models (ranked by vision > speed > cost — free first)

| Rank | Model | Vision | Cost | Context | Why |
|------|-------|--------|------|---------|-----|
| 1 | Qwen 3.6 Plus | ✅ | FREE (OR) | 1M | Best free aux — vision + 1M ctx. Not through Nous. |
| 2 | MiniMax M3 | ✅ | FREE (NIM) | 1M | Vision + 1M ctx via NIM. |
| 3 | DeepSeek V4 Flash | ❌ | FREE (NIM) | 1M | Fast reasoning, no vision — pair with vision main. |
| 4 | Mimo V2.5 | ✅ | $0.105/$0.28 (Nous) | 1M | Cheapest vision aux through Nous. |
| 5 | Qwen 3.5 Flash | ✅ | $0.065/$0.26 (Nous) | 1M | Cheapest paid vision aux, 1M ctx. |
| 6 | Kimi K2.6 | ✅ | $0.60/$2.50 | 1M | Strong reasoning aux, 1M ctx. |
| 7 | Kimi K2.7 Code | ✅ | $0.74/$3.50 | 256K | Strong coding aux, limited ctx. |
| 8 | Qwen 3.6 Flash | ✅ | $0.1875/$1.125 | 262K | Mid-tier, 262K ctx only. |

## Free Endpoints (outside Nous gateway)

| Model | Via | Ctx | Vision | Key Required | Rate Limit |
|-------|-----|-----|--------|-------------|------------|
| DeepSeek V4 Pro | NIM | 1M | ❌ | `NVIDIA_API_KEY` | ~1000 RPM |
| DeepSeek V4 Flash | NIM | 1M | ❌ | `NVIDIA_API_KEY` | ~1000 RPM |
| MiniMax M3 | NIM | 1M | ✅ | `NVIDIA_API_KEY` | ~1000 RPM |
| Qwen 3.6 Plus | OpenRouter (`:free`) | 1M | ✅ | `OPENROUTER_API_KEY` | Rate-limited |

## Hardware Tier → Default Pairings

| Hardware Tier | Default Main | Default Aux | Gateway | Blended Cost |
|---------------|-------------|-------------|---------|-------------|
| **Beefy** (≥24GB) | Darwin 28B (local) | Carnice 35A3B (local) | N/A | $0 |
| **Beefy API fallback** | GLM 5.2 (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | ~$0.44/$1.42 |
| **Modest** (8-24GB) | DeepSeek V4 Pro (Nous) | Qwen 3.6 Plus (OR free) | 🟡 NOUS+OR | ~$0.17/$0.35 |
| **Thin** (<8GB) | DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | ⚪ NIM | FREE |

**Zero-cost full Hermes-Agent capability is available to anyone regardless of hardware.**

## Provider Routing Notes

- **Nous provider** (`inference-api.nousresearch.com/v1`): OpenRouter-style slugs, $22/mo credits, Tool Gateway included. Best for users who want one bill.
- **OpenRouter direct**: 10% credit bonus when adding credits. Free models (`:free`) available but NOT through Nous gateway. Use for Qwen 3.6 Plus free aux.
- **NVIDIA NIM** (`integrate.api.nvidia.com/v1`): Free endpoints, ~1000 RPM, separate `NVIDIA_API_KEY`. No Tool Gateway. Best for zero-cost setups.
- **DeepSeek direct API**: Cheapest DeepSeek pricing ($0.435/$0.87 for Pro, $0.14/$0.28 for Flash). Cache hit: $0.0028/$0.0036 per M input.
- **Z.AI direct**: GLM 5.2 at $1.40/$4.40 (most expensive). OpenRouter routing is cheaper ($0.95/$3.00).

## Tier Legend

- **S** = Best-in-class reasoning
- **SF** = Strong reasoning + speed/value
- **F** = Strong performance
- **SD** = Decent, good price/performance
