# API Model Rankings by Volume Performance

Ranked by reasoning/coding quality for main, and vision+cost for aux. Free endpoints listed first.

## Main API Rankings

| Tier | Model | Vision | Provider | Cost | 1M ctx | Why |
|------|-------|--------|----------|------|--------|-----|
| **S** | DeepSeek V4 Pro | No | NIM (free) / OpenRouter (paid) | FREE / $0.60/$1.50 | No | Best open reasoning + coding |
| **S** | Kimi K2.6 | Yes | OpenRouter (free tier) | FREE | Yes | Strong coding + vision + 1M ctx |
| **S** | GLM 5.1 | No | Z.AI / OpenRouter | $0.10/$0.30 | No | Agentic + long-horizon reasoning |
| **S** | Qwen 3.7 Max | No | DashScope / OpenRouter | Paid | Yes | Qwen frontier, 1M ctx |
| **SF** | DeepSeek V4 Flash | No | NIM (free) | FREE | No | Fast reasoning, best value per token |
| **SF** | MiniMax M3 | Yes | NIM (free) | FREE | Yes | Vision + 1M ctx |
| **SF** | Nemotron Ultra | No | NIM (free) | FREE | No | 550B MoE, reasoning budget |
| **F** | Qwen 3.5 Flash | Yes | OpenRouter | $0.065/$0.26 | Yes | Cheapest paid vision main |

## Aux API Rankings

Ranked by vision > speed > cost. Free first.

| Tier | Model | Vision | Provider | Cost | 1M ctx | Why |
|------|-------|--------|----------|------|--------|-----|
| **SD** | Kimi K2.6 | Yes | OpenRouter (free) | FREE | Yes | Best free aux — vision + reasoning + 1M ctx |
| **SD** | MiniMax M3 | Yes | NIM (free) | FREE | Yes | Vision + 1M ctx |
| **SD** | Step 3.7 Flash | Yes | OpenRouter (free) | FREE | No | Fast vision aux |
| **SD** | DeepSeek V4 Flash | No | NIM (free) | FREE | No | Fast reasoning (no vision — pair with vision main) |
| **F** | Qwen 3.5 Flash | Yes | OpenRouter | $0.065/$0.26 | Yes | Cheapest paid vision aux |

## Free Tier Details

- **NVIDIA NIM** — `https://integrate.api.nvidia.com/v1`, `NVIDIA_API_KEY` from `build.nvidia.com` signup. ~1000 RPM, no credit card.
- **OpenRouter free tier** — models with `:free` suffix (e.g. `deepseek/deepseek-chat-v3-0324:free`). Rate-limited but zero cost.
- **Kimi K2.6** — available free via OpenRouter's `moonshotai/kimi-k2.6:free` endpoint.

## Hardware Tier → API Recommendations

| Hardware Tier | Main | Aux | Total Cost |
|---------------|------|-----|-----------|
| **Beefy** (≥24GB) | Local (Darwin Reason) | Local (Carnice Apex) | $0 |
| **Modest** (8-24GB) | API (DeepSeek V4 Pro free) | Local (small model, cpu-moe) | $0 |
| **Thin** (<8GB / no GPU) | API (DeepSeek V4 Pro free) | API (Kimi K2.6 free) | $0 |

**Zero-cost full Hermes-Agent capability is available to anyone regardless of hardware.**
