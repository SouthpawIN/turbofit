# API Tier Rankings (2026-06-25)

Curated API models ranked by quality. Prices in $/M tokens (input/output).

## Main API Models (Text-Only — must pair with vision aux)

| Rank | Model | Provider | Input $ | Output $ | Vision | Context | Tier | Through Nous? |
|------|-------|----------|---------|----------|--------|---------|------|---------------|
| 1 | GLM 5.2 | Z.AI / OR / Nous | $0.95 (OR) / $1.40 (Z.AI) | $3.00 (OR) / $4.40 (Z.AI) | ❌ | 1M | S | ✅ `z-ai/glm-5.2` |
| 2 | Qwen 3.7 MAX | DashScope / OR / Nous | $1.25 | $3.75 | ❌ | 1M | S | ✅ `qwen/qwen3.7-max` |
| 3 | DeepSeek V4 Pro | DS / OR / NIM (free) / Nous | $0.435 (DS) / FREE (NIM) | $0.87 (DS) / FREE (NIM) | ❌ | 1M | S | ✅ `deepseek/deepseek-v4-pro` |
| 4 | DeepSeek V4 Flash | DS / OR / NIM (free) / Nous | $0.09 (OR) / FREE (NIM) | $0.18 (OR) / FREE (NIM) | ❌ | 1M | SF | ✅ `deepseek/deepseek-v4-flash` |
| 5 | Mimo V2.5 Pro | Xiaomi / OR / Nous | ~$1.00 | ~$3.00 | ❌ | 1M | SF | ✅ `xiaomi/mimo-v2.5-pro` |

## Vision-Capable Models (main or aux)

| Rank | Model | Input $ | Output $ | Context | Tier | Through Nous? |
|------|-------|---------|----------|---------|------|---------------|
| 1 | MiniMax M3 | ~$0.30 / FREE (NIM) | ~$1.20 / FREE (NIM) | 1M | SF | ✅ |
| 2 | Qwen 3.7 Plus | $0.32 (OR) | $1.28 (OR) | 1M | SF | ✅ |
| 3 | Mimo V2.5 | $0.105 | $0.28 | 1M | F | ✅ |
| 4 | Kimi K2.7 Code | $0.74 | $3.50 | 256K | SF | ✅ |
| 5 | Kimi K2.6 | $0.60 | $2.50 | 1M | SF | ✅ |
| 6 | Qwen 3.6 Plus | FREE (OR) | FREE (OR) | 1M | F | ❌ OR only |
| 7 | Qwen 3.5 Flash | $0.065 | $0.26 | 1M | SD | ✅ |
| 8 | Qwen 3.6 Flash | $0.1875 | $1.125 | 262K | F | ✅ |

## Free API Models (zero cost)

| Model | Vision | Context | Via | Notes |
|-------|--------|---------|-----|-------|
| DeepSeek V4 Pro | ❌ | 1M | NIM | Free, ~1000 RPM |
| DeepSeek V4 Flash | ❌ | 1M | NIM | Free, ~1000 RPM |
| MiniMax M3 | ✅ | 1M | NIM | Free, vision, ~1000 RPM |
| Qwen 3.6 Plus | ✅ | 1M | OpenRouter (`:free`) | Free preview, rate-limited |

## Hardware Tier Defaults

| Tier | VRAM | Default Main | Default Aux | Gateway | Blended Cost |
|------|------|-------------|-------------|---------|-------------|
| Beefy | ≥24GB | Darwin 28B (local) | Carnice 35A3B (local) | N/A | $0 |
| Beefy API | fallback | GLM 5.2 (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | ~$0.44/$1.42 |
| Modest | 8-24GB | DeepSeek V4 Pro (Nous) | Qwen 3.6 Plus (OR free) | 🟡 NOUS+OR | ~$0.17/$0.35 |
| Thin | <8GB | DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | ⚪ NIM | FREE |

## Provider & Gateway Indicators

| Indicator | Meaning |
|-----------|---------|
| 🟢 NOUS+TG | Both through Nous → Tool Gateway active (Firecrawl, FAL, OpenAI TTS, Browser Use) |
| 🟡 NOUS+OR | Main through Nous (TG), aux through OpenRouter (10% credit bonus) |
| 🟠 NOUS+NIM | Main through Nous (TG), aux through NIM (free) |
| 🔵 OR | Both through OpenRouter (10% bonus, no TG) |
| ⚪ NIM | Both through NIM (free, no TG) |

## Tier Legend

- **S** = Best-in-class reasoning
- **SF** = Strong reasoning + speed/value
- **F** = Strong performance
- **SD** = Decent, good price/performance
