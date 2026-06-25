# API Tier Rankings (2026-06-23)

Curated API models ranked by quality. Prices in $/M tokens (input/output).

## Main API Models

| Rank | Model | Provider | Input $ | Output $ | Vision | Context | Tier |
|------|-------|----------|---------|----------|--------|---------|------|
| 1 | deepseek-v4-pro | nous | 0.44 | 0.87 | ❌ | 1M | S |
| 2 | qwen-3.7-max | nous | 1.25 | 3.75 | ❌ | 1M | S |
| 3 | minimax-m3 | nous | 0.30 | 1.20 | ✅ | 1M | SF |
| 4 | qwen-3.7-plus | nous | 0.80 | 2.40 | ✅ | 1M | SF |
| 5 | qwen-3.6-plus | nous | 0.40 | 1.20 | ✅ | 262K | F |
| 6 | kimi-k2.6 | nous | 0.00 | 0.00 | ✅ | 1M | SF |
| 7 | step-3.7-flash | nous | 0.20 | 1.15 | ✅ | 262K | F |
| 8 | mimo-v2.5-flash | nous | 0.00 | 0.00 | ✅ | 262K | F |
| 9 | qwen-3.6-flash | nous | 0.08 | 0.24 | ✅ | 262K | SD |
| 10 | qwen-3.5-flash | nous | 0.065 | 0.26 | ✅ | 1M | SD |

## Free API Models (zero cost)

| Model | Vision | Context | Notes |
|-------|--------|---------|-------|
| kimi-k2.6 | ✅ | 1M | Best free model — reasoning + coding + vision |
| deepseek-v4-pro | ❌ | 1M | Tier-S reasoning, free |
| mimo-v2.5-flash | ✅ | 262K | Small but capable |
| step-3.7-flash-free | ✅ | 262K | StepFun free tier |

## Hardware Tier Defaults

| Tier | VRAM | Setup | Default Main | Default Aux |
|------|------|-------|-------------|-------------|
| Beefy | ≥24GB | Local main + local aux | Darwin Reason 28B | Carnice 35A3B Nano |
| Modest | 8-24GB | API main + local aux | DeepSeek V4 Pro (free) | Small local model |
| Thin | <8GB / no GPU | API main + API aux | DeepSeek V4 Pro | Kimi K2.6 (free, vision) |

## Tier Legend

- **S** = Best-in-class reasoning
- **SF** = Best-in-class with vision
- **F** = Strong performance
- **SD** = Decent, good price/performance