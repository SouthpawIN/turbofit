# Turbofit Model Research Report — 2026-07-04

## Data Sources
- Model pricing: OpenRouter API (`/api/v1/models`) — live
- Usage data: Hermes state.db (`/home/sovthpaw/.hermes/profiles/senter/state.db`) — real user data

## Your Real Usage (from Hermes Insights)

- Active days: 14
- Total sessions: 899
- Total messages: 22,081
- Total tool calls: 11,546
- Total input tokens: 98,193,591
- Total output tokens: 3,819,760
- Total cache read tokens: 508,111,923
- Effective cache hit rate: 83.8%
- Avg input/session: 109,225
- Avg output/session: 4,249
- Avg sessions/day: 64.2
- Total estimated cost: $197.68

### Models You've Actually Used

| Model | Sessions | Input Tokens | Output Tokens | Cache Read | Est. Cost |
|-------|----------|-------------|--------------|------------|-----------|
| z-ai/glm-5.2 | 180 | 32,597,521 | 1,732,552 | 300,617,443 | $90.59 |
| qwen/qwen3.7-max | 107 | 29,001,956 | 573,215 | 105,640,586 | $67.50 |
| Darwin-28B-REASON.Q4_K_M.gguf | 358 | 26,559,713 | 971,396 | 20,793,778 | $15.17 |
| deepseek/deepseek-v4-pro | 231 | 8,599,799 | 501,118 | 79,658,712 | $24.15 |
| qwen/qwen3.6-plus | 2 | 719,357 | 8,455 | 0 | $0.25 |
| darwin-28b-reason | 14 | 412,452 | 31,344 | 1,401,404 | $0.00 |
| qwen/qwen3.5-flash-02-23 | 7 | 302,793 | 1,680 | 0 | $0.02 |

### By Platform

| Platform | Sessions | Input | Output | Cost |
|----------|----------|-------|--------|------|
| discord | 27 | 56,049,049 | 1,145,423 | $119.29 |
| cron | 586 | 21,421,512 | 1,354,231 | $22.21 |
| subagent | 268 | 11,734,626 | 970,266 | $23.18 |
| cli | 13 | 8,461,436 | 344,888 | $32.89 |
| tui | 5 | 526,968 | 4,952 | $0.11 |

## Live Model Pricing (from OpenRouter API)

| Slug | Vision | Input $/M | Cache Read $/M | Output $/M | Context | Free |
|------|--------|----------|---------------|-----------|---------|------|
| deepseek/deepseek-v3.2 | no | $0.2288 | $0.0229 | $0.3432 | 131K |  |
| deepseek/deepseek-v3.2-exp | no | $0.2700 | - | $0.4100 | 163K |  |
| deepseek/deepseek-v4-flash | no | $0.0900 | $0.0180 | $0.1800 | 1048K |  |
| deepseek/deepseek-v4-pro | no | $0.4350 | $0.0036 | $0.8700 | 1048K |  |
| minimax/minimax-m3 | YES | $0.3000 | $0.0600 | $1.2000 | 1048K |  |
| moonshotai/kimi-k2.6 | YES | $0.6600 | $0.1400 | $3.4100 | 262K |  |
| moonshotai/kimi-k2.7-code | YES | $0.7400 | $0.1500 | $3.5000 | 262K |  |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | no | $0.4000 | - | $0.4000 | 131K |  |
| nvidia/nemotron-3-nano-30b-a3b | no | $0.0500 | - | $0.2000 | 262K |  |
| nvidia/nemotron-3-nano-30b-a3b:free | no | $0.0000 | - | $0.0000 | 256K | FREE |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | YES | $0.0000 | - | $0.0000 | 256K | FREE |
| nvidia/nemotron-3-super-120b-a12b | no | $0.0800 | - | $0.4500 | 1000K |  |
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
| z-ai/glm-5.1 | no | $0.9660 | $0.1794 | $3.0360 | 202K |  |
| z-ai/glm-5.2 | no | $0.8400 | $0.1560 | $2.6400 | 1048K |  |

## Monthly Cost Projections (based on YOUR real usage)

Projected using your actual daily input/output averages and cache hit rate.
Single model = all tokens go to this model. Pairing = 60% aux offset.

| Model | Single (monthly) | Notes |
|-------|-------------------|-------|
| deepseek/deepseek-v3.2 | $14.64 | 90% cache savings |
| deepseek/deepseek-v3.2-exp | $60.17 |  |
| deepseek/deepseek-v4-flash | $7.71 | 80% cache savings |
| deepseek/deepseek-v4-pro | $22.58 | 99% cache savings |
| minimax/minimax-m3 | $30.63 | 80% cache savings; vision |
| moonshotai/kimi-k2.6 | $75.09 | 79% cache savings; vision |
| moonshotai/kimi-k2.7-code | $80.32 | 80% cache savings; vision |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | $87.44 |  |
| nvidia/nemotron-3-nano-30b-a3b | $12.16 |  |
| nvidia/nemotron-3-nano-30b-a3b:free | FREE |  |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | FREE | vision |
| nvidia/nemotron-3-super-120b-a12b | $20.52 |  |
| nvidia/nemotron-3-super-120b-a12b:free | FREE |  |
| nvidia/nemotron-3-ultra-550b-a55b | $52.68 | 80% cache savings |
| nvidia/nemotron-3-ultra-550b-a55b:free | FREE |  |
| nvidia/nemotron-3.5-content-safety:free | FREE | vision |
| nvidia/nemotron-nano-12b-v2-vl:free | FREE | vision |
| nvidia/nemotron-nano-9b-v2:free | FREE |  |
| qwen/qwen3.5-flash-02-23 | $15.81 | vision |
| qwen/qwen3.6-flash | $48.66 | vision |
| qwen/qwen3.6-plus | $84.35 | vision |
| qwen/qwen3.7-max | $117.38 | 80% cache savings |
| qwen/qwen3.7-plus | $32.67 | 80% cache savings; vision |
| xiaomi/mimo-v2.5 | $24.39 | vision |
| xiaomi/mimo-v2.5-pro | $22.58 | 99% cache savings |
| z-ai/glm-5.1 | $89.40 | 81% cache savings |
| z-ai/glm-5.2 | $77.74 | 81% cache savings |

### Recommended Pairings (projected on YOUR usage)

| Pairing | Monthly Cost | Notes |
|---------|-------------|-------|
| GLM 5.2 + Qwen 3.5 Flash | $40.58 | |
| GLM 5.2 + Mimo V2.5 | $45.73 | |
| GLM 5.2 + MiniMax M3 | $49.47 | |
| GLM 5.2 + Kimi K2.7 Code | $79.29 | |
| Qwen 3.7 MAX + Qwen 3.5 Flash | $56.43 | |
| Qwen 3.7 MAX + Mimo V2.5 | $61.58 | |
| Qwen 3.7 MAX + MiniMax M3 | $65.33 | |
| Qwen 3.7 MAX + Kimi K2.7 Code | $95.14 | |
| DeepSeek V4 Pro + Qwen 3.5 Flash | $18.52 | |
| DeepSeek V4 Pro + Mimo V2.5 | $23.66 | |
| DeepSeek V4 Pro + MiniMax M3 | $27.41 | |
| DeepSeek V4 Pro + Kimi K2.7 Code | $57.22 | |
| DeepSeek V4 Flash + Qwen 3.5 Flash | $12.57 | |
| DeepSeek V4 Flash + Mimo V2.5 | $17.72 | |
| DeepSeek V4 Flash + MiniMax M3 | $21.46 | |
| DeepSeek V4 Flash + Kimi K2.7 Code | $51.28 | |
| Mimo V2.5 Pro + Qwen 3.5 Flash | $18.51 | |
| Mimo V2.5 Pro + Mimo V2.5 | $23.66 | |
| Mimo V2.5 Pro + MiniMax M3 | $27.41 | |
| Mimo V2.5 Pro + Kimi K2.7 Code | $57.22 | |

## Cache Pricing Analysis

Models with cache read pricing offer significant savings for repeated context.
Your effective cache hit rate: 83.8%

- **deepseek/deepseek-v3.2**: cache read $0.0229/M vs input $0.2288/M — 90% savings on cache hits
- **deepseek/deepseek-v4-flash**: cache read $0.0180/M vs input $0.0900/M — 80% savings on cache hits
- **deepseek/deepseek-v4-pro**: cache read $0.0036/M vs input $0.4350/M — 99% savings on cache hits
- **minimax/minimax-m3**: cache read $0.0600/M vs input $0.3000/M — 80% savings on cache hits
- **moonshotai/kimi-k2.6**: cache read $0.1400/M vs input $0.6600/M — 79% savings on cache hits
- **moonshotai/kimi-k2.7-code**: cache read $0.1500/M vs input $0.7400/M — 80% savings on cache hits
- **nvidia/nemotron-3-ultra-550b-a55b**: cache read $0.1000/M vs input $0.5000/M — 80% savings on cache hits
- **qwen/qwen3.7-max**: cache read $0.2500/M vs input $1.2500/M — 80% savings on cache hits
- **qwen/qwen3.7-plus**: cache read $0.0640/M vs input $0.3200/M — 80% savings on cache hits
- **xiaomi/mimo-v2.5-pro**: cache read $0.0036/M vs input $0.4350/M — 99% savings on cache hits
- **z-ai/glm-5.1**: cache read $0.1794/M vs input $0.9660/M — 81% savings on cache hits
- **z-ai/glm-5.2**: cache read $0.1560/M vs input $0.8400/M — 81% savings on cache hits

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
