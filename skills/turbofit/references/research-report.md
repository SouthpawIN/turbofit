# Turbofit Model Research Report — 2026-06-27

## Data Sources
- Model pricing: OpenRouter API (`/api/v1/models`) — live
- Usage data: Hermes state.db (`/home/sovthpaw/.hermes/profiles/senter/state.db`) — real user data

## Your Real Usage (from Hermes Insights)

- Active days: 7
- Total sessions: 175
- Total messages: 10,452
- Total tool calls: 5,364
- Total input tokens: 60,105,269
- Total output tokens: 1,867,554
- Total cache read tokens: 391,759,964
- Effective cache hit rate: 86.7%
- Avg input/session: 343,459
- Avg output/session: 10,672
- Avg sessions/day: 25.0
- Total estimated cost: $148.21

### Models You've Actually Used

| Model | Sessions | Input Tokens | Output Tokens | Cache Read | Est. Cost |
|-------|----------|-------------|--------------|------------|-----------|
| z-ai/glm-5.2 | 140 | 28,598,885 | 1,360,076 | 218,182,792 | $70.88 |
| qwen/qwen3.7-max | 4 | 23,468,801 | 241,614 | 98,799,228 | $55.49 |
| deepseek/deepseek-v4-pro | 22 | 7,015,433 | 255,729 | 74,777,944 | $21.57 |
| qwen/qwen3.6-plus | 2 | 719,357 | 8,455 | 0 | $0.25 |
| qwen/qwen3.5-flash-02-23 | 7 | 302,793 | 1,680 | 0 | $0.02 |

### By Platform

| Platform | Sessions | Input | Output | Cost |
|----------|----------|-------|--------|------|
| discord | 17 | 46,340,599 | 938,295 | $112.20 |
| cli | 10 | 6,266,625 | 242,616 | $20.13 |
| subagent | 94 | 4,587,635 | 419,262 | $9.45 |
| cron | 49 | 2,383,442 | 262,429 | $6.32 |
| tui | 5 | 526,968 | 4,952 | $0.11 |

## Live Model Pricing (from OpenRouter API)

| Slug | Vision | Input $/M | Cache Read $/M | Output $/M | Context | Free |
|------|--------|----------|---------------|-----------|---------|------|
| deepseek/deepseek-v3.2 | no | $0.2288 | $0.0229 | $0.3432 | 131K |  |
| deepseek/deepseek-v3.2-exp | no | $0.2700 | - | $0.4100 | 163K |  |
| deepseek/deepseek-v4-flash | no | $0.0900 | $0.0200 | $0.1800 | 1048K |  |
| deepseek/deepseek-v4-pro | no | $0.4350 | $0.0036 | $0.8700 | 1048K |  |
| minimax/minimax-m3 | YES | $0.3000 | $0.0600 | $1.2000 | 1048K |  |
| moonshotai/kimi-k2.6 | YES | $0.6600 | $0.1440 | $3.4100 | 262K |  |
| moonshotai/kimi-k2.7-code | YES | $0.7400 | $0.1500 | $3.5000 | 262K |  |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | no | $0.4000 | - | $0.4000 | 131K |  |
| nvidia/nemotron-3-nano-30b-a3b | no | $0.0500 | - | $0.2000 | 262K |  |
| nvidia/nemotron-3-nano-30b-a3b:free | no | $0.0000 | - | $0.0000 | 256K | FREE |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | YES | $0.0000 | - | $0.0000 | 256K | FREE |
| nvidia/nemotron-3-super-120b-a12b | no | $0.0850 | - | $0.4000 | 1000K |  |
| nvidia/nemotron-3-super-120b-a12b:free | no | $0.0000 | - | $0.0000 | 1000K | FREE |
| nvidia/nemotron-3-ultra-550b-a55b | no | $0.5000 | $0.1000 | $2.2000 | 1000K |  |
| nvidia/nemotron-3-ultra-550b-a55b:free | no | $0.0000 | - | $0.0000 | 1000K | FREE |
| nvidia/nemotron-3.5-content-safety:free | YES | $0.0000 | - | $0.0000 | 128K | FREE |
| nvidia/nemotron-nano-12b-v2-vl:free | YES | $0.0000 | - | $0.0000 | 128K | FREE |
| nvidia/nemotron-nano-9b-v2:free | no | $0.0000 | - | $0.0000 | 128K | FREE |
| qwen/qwen3.5-flash-02-23 | YES | $0.0650 | - | $0.2600 | 1000K |  |
| qwen/qwen3.6-flash | YES | $0.1875 | - | $1.1250 | 1000K |  |
| qwen/qwen3.6-plus | YES | $0.3250 | - | $1.9500 | 1000K |  |
| qwen/qwen3.7-max | no | $1.2500 | $0.2500 | $3.7500 | 1000K |  |
| qwen/qwen3.7-plus | YES | $0.3200 | $0.0640 | $1.2800 | 1000K |  |
| xiaomi/mimo-v2.5 | YES | $0.1050 | - | $0.2800 | 1048K |  |
| xiaomi/mimo-v2.5-pro | no | $0.4350 | $0.0036 | $0.8700 | 1048K |  |
| z-ai/glm-5.1 | no | $0.9800 | $0.1820 | $3.0800 | 202K |  |
| z-ai/glm-5.2 | no | $0.9500 | $0.1800 | $3.0000 | 1048K |  |

## Monthly Cost Projections (based on YOUR real usage)

Projected using your actual daily input/output averages and cache hit rate.
Single model = all tokens go to this model. Pairing = 60% aux offset.

| Model | Single (monthly) | Notes |
|-------|-------------------|-------|
| deepseek/deepseek-v3.2 | $15.70 | 90% cache savings |
| deepseek/deepseek-v3.2-exp | $72.83 |  |
| deepseek/deepseek-v4-flash | $8.99 | 78% cache savings |
| deepseek/deepseek-v4-pro | $22.68 | 99% cache savings |
| minimax/minimax-m3 | $33.28 | 80% cache savings; vision |
| moonshotai/kimi-k2.6 | $82.07 | 78% cache savings; vision |
| moonshotai/kimi-k2.7-code | $86.87 | 80% cache savings; vision |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | $106.24 |  |
| nvidia/nemotron-3-nano-30b-a3b | $14.48 |  |
| nvidia/nemotron-3-nano-30b-a3b:free | FREE |  |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | FREE | vision |
| nvidia/nemotron-3-super-120b-a12b | $25.10 |  |
| nvidia/nemotron-3-super-120b-a12b:free | FREE |  |
| nvidia/nemotron-3-ultra-550b-a55b | $57.07 | 80% cache savings |
| nvidia/nemotron-3-ultra-550b-a55b:free | FREE |  |
| nvidia/nemotron-3.5-content-safety:free | FREE | vision |
| nvidia/nemotron-nano-12b-v2-vl:free | FREE | vision |
| nvidia/nemotron-nano-9b-v2:free | FREE |  |
| qwen/qwen3.5-flash-02-23 | $18.82 | vision |
| qwen/qwen3.6-flash | $57.30 | vision |
| qwen/qwen3.6-plus | $99.33 | vision |
| qwen/qwen3.7-max | $128.68 | 80% cache savings |
| qwen/qwen3.7-plus | $35.50 | 80% cache savings; vision |
| xiaomi/mimo-v2.5 | $29.29 | vision |
| xiaomi/mimo-v2.5-pro | $22.67 | 99% cache savings |
| z-ai/glm-5.1 | $98.88 | 81% cache savings |
| z-ai/glm-5.2 | $96.76 | 81% cache savings |

### Recommended Pairings (projected on YOUR usage)

| Pairing | Monthly Cost | Notes |
|---------|-------------|-------|
| GLM 5.2 + Qwen 3.5 Flash | $50.00 | |
| GLM 5.2 + Mimo V2.5 | $56.28 | |
| GLM 5.2 + MiniMax M3 | $58.67 | |
| GLM 5.2 + Kimi K2.7 Code | $90.83 | |
| Qwen 3.7 MAX + Qwen 3.5 Flash | $62.77 | |
| Qwen 3.7 MAX + Mimo V2.5 | $69.04 | |
| Qwen 3.7 MAX + MiniMax M3 | $71.44 | |
| Qwen 3.7 MAX + Kimi K2.7 Code | $103.59 | |
| DeepSeek V4 Pro + Qwen 3.5 Flash | $20.37 | |
| DeepSeek V4 Pro + Mimo V2.5 | $26.64 | |
| DeepSeek V4 Pro + MiniMax M3 | $29.04 | |
| DeepSeek V4 Pro + Kimi K2.7 Code | $61.19 | |
| DeepSeek V4 Flash + Qwen 3.5 Flash | $14.89 | |
| DeepSeek V4 Flash + Mimo V2.5 | $21.17 | |
| DeepSeek V4 Flash + MiniMax M3 | $23.57 | |
| DeepSeek V4 Flash + Kimi K2.7 Code | $55.72 | |
| Mimo V2.5 Pro + Qwen 3.5 Flash | $20.36 | |
| Mimo V2.5 Pro + Mimo V2.5 | $26.64 | |
| Mimo V2.5 Pro + MiniMax M3 | $29.04 | |
| Mimo V2.5 Pro + Kimi K2.7 Code | $61.19 | |

## Cache Pricing Analysis

Models with cache read pricing offer significant savings for repeated context.
Your effective cache hit rate: 86.7%

- **deepseek/deepseek-v3.2**: cache read $0.0229/M vs input $0.2288/M — 90% savings on cache hits
- **deepseek/deepseek-v4-flash**: cache read $0.0200/M vs input $0.0900/M — 78% savings on cache hits
- **deepseek/deepseek-v4-pro**: cache read $0.0036/M vs input $0.4350/M — 99% savings on cache hits
- **minimax/minimax-m3**: cache read $0.0600/M vs input $0.3000/M — 80% savings on cache hits
- **moonshotai/kimi-k2.6**: cache read $0.1440/M vs input $0.6600/M — 78% savings on cache hits
- **moonshotai/kimi-k2.7-code**: cache read $0.1500/M vs input $0.7400/M — 80% savings on cache hits
- **nvidia/nemotron-3-ultra-550b-a55b**: cache read $0.1000/M vs input $0.5000/M — 80% savings on cache hits
- **qwen/qwen3.7-max**: cache read $0.2500/M vs input $1.2500/M — 80% savings on cache hits
- **qwen/qwen3.7-plus**: cache read $0.0640/M vs input $0.3200/M — 80% savings on cache hits
- **xiaomi/mimo-v2.5-pro**: cache read $0.0036/M vs input $0.4350/M — 99% savings on cache hits
- **z-ai/glm-5.1**: cache read $0.1820/M vs input $0.9800/M — 81% savings on cache hits
- **z-ai/glm-5.2**: cache read $0.1800/M vs input $0.9500/M — 81% savings on cache hits

## Free Models (zero cost on OpenRouter)

- **nvidia/nemotron-3-nano-30b-a3b:free** — 256K ctx, text-only
- **nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free** — 256K ctx, vision
- **nvidia/nemotron-3-super-120b-a12b:free** — 1000K ctx, text-only
- **nvidia/nemotron-3-ultra-550b-a55b:free** — 1000K ctx, text-only
- **nvidia/nemotron-3.5-content-safety:free** — 128K ctx, vision
- **nvidia/nemotron-nano-12b-v2-vl:free** — 128K ctx, vision
- **nvidia/nemotron-nano-9b-v2:free** — 128K ctx, text-only

## About the Nous Tool Gateway

The Nous Tool Gateway (Firecrawl, FAL, OpenAI TTS, Browser Use) is active
whenever the user has a Nous Portal subscription — regardless of which
models are selected for main or aux. It is a subscription feature, not
a per-model feature.

## Action Items

1. Compare prices above with model-database.yaml — update any that changed
2. Add any NEW models not yet in the database (only real, available models)
3. Check if any free models changed status
4. Run `bash scripts/sync-github.sh` after updates
