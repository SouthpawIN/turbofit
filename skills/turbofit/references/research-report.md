# Turbofit Model Research Report — 2026-06-25

## Data Sources
- Model pricing: OpenRouter API (`/api/v1/models`) — live
- Usage data: Hermes state.db (`/home/sovthpaw/.hermes/profiles/senter/state.db`) — real user data

## Your Real Usage (from Hermes Insights)

- Active days: 5
- Total sessions: 67
- Total messages: 2,943
- Total tool calls: 1,585
- Total input tokens: 17,427,559
- Total output tokens: 561,384
- Total cache read tokens: 88,038,531
- Effective cache hit rate: 83.5%
- Avg input/session: 260,113
- Avg output/session: 8,379
- Avg sessions/day: 13.4
- Total estimated cost: $33.36

### Models You've Actually Used

| Model | Sessions | Input Tokens | Output Tokens | Cache Read | Est. Cost |
|-------|----------|-------------|--------------|------------|-----------|
| z-ai/glm-5.2 | 58 | 16,405,409 | 551,249 | 88,038,531 | $33.09 |
| qwen/qwen3.6-plus | 2 | 719,357 | 8,455 | 0 | $0.25 |
| qwen/qwen3.5-flash-02-23 | 7 | 302,793 | 1,680 | 0 | $0.02 |

### By Platform

| Platform | Sessions | Input | Output | Cost |
|----------|----------|-------|--------|------|
| discord | 5 | 13,827,274 | 338,508 | $28.37 |
| subagent | 37 | 1,484,104 | 138,509 | $2.61 |
| cron | 15 | 1,086,068 | 73,613 | $2.10 |
| tui | 5 | 526,968 | 4,952 | $0.11 |
| cli | 5 | 503,145 | 5,802 | $0.17 |

## Live Model Pricing (from OpenRouter API)

| Slug | Vision | Input $/M | Cache Read $/M | Output $/M | Context | Free |
|------|--------|----------|---------------|-----------|---------|------|
| deepseek/deepseek-v3.2 | no | $0.2288 | - | $0.3432 | 131K |  |
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
| deepseek/deepseek-v3.2 | $25.08 |  |
| deepseek/deepseek-v3.2-exp | $29.61 |  |
| deepseek/deepseek-v4-flash | $3.91 | 78% cache savings |
| deepseek/deepseek-v4-pro | $10.76 | 99% cache savings |
| minimax/minimax-m3 | $14.46 | 80% cache savings; vision |
| moonshotai/kimi-k2.6 | $35.46 | 78% cache savings; vision |
| moonshotai/kimi-k2.7-code | $37.67 | 80% cache savings; vision |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | $43.17 |  |
| nvidia/nemotron-3-nano-30b-a3b | $5.90 |  |
| nvidia/nemotron-3-nano-30b-a3b:free | FREE |  |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | FREE | vision |
| nvidia/nemotron-3-super-120b-a12b | $10.24 |  |
| nvidia/nemotron-3-super-120b-a12b:free | FREE |  |
| nvidia/nemotron-3-ultra-550b-a55b | $24.78 | 80% cache savings |
| nvidia/nemotron-3-ultra-550b-a55b:free | FREE |  |
| nvidia/nemotron-3.5-content-safety:free | FREE | vision |
| nvidia/nemotron-nano-12b-v2-vl:free | FREE | vision |
| nvidia/nemotron-nano-9b-v2:free | FREE |  |
| qwen/qwen3.5-flash-02-23 | $7.67 | vision |
| qwen/qwen3.6-flash | $23.40 | vision |
| qwen/qwen3.6-plus | $40.55 | vision |
| qwen/qwen3.7-max | $56.05 | 80% cache savings |
| qwen/qwen3.7-plus | $15.43 | 80% cache savings; vision |
| xiaomi/mimo-v2.5 | $11.92 | vision |
| xiaomi/mimo-v2.5-pro | $10.76 | 99% cache savings |
| z-ai/glm-5.1 | $43.19 | 81% cache savings |
| z-ai/glm-5.2 | $42.23 | 81% cache savings |

### Recommended Pairings (projected on YOUR usage)

| Pairing | Monthly Cost | Notes |
|---------|-------------|-------|
| GLM 5.2 + Qwen 3.5 Flash | $21.50 | |
| GLM 5.2 + Mimo V2.5 | $24.05 | |
| GLM 5.2 + MiniMax M3 | $25.57 | |
| GLM 5.2 + Kimi K2.7 Code | $39.49 | |
| Qwen 3.7 MAX + Qwen 3.5 Flash | $27.02 | |
| Qwen 3.7 MAX + Mimo V2.5 | $29.57 | |
| Qwen 3.7 MAX + MiniMax M3 | $31.10 | |
| Qwen 3.7 MAX + Kimi K2.7 Code | $45.02 | |
| DeepSeek V4 Pro + Qwen 3.5 Flash | $8.91 | |
| DeepSeek V4 Pro + Mimo V2.5 | $11.46 | |
| DeepSeek V4 Pro + MiniMax M3 | $12.98 | |
| DeepSeek V4 Pro + Kimi K2.7 Code | $26.91 | |
| DeepSeek V4 Flash + Qwen 3.5 Flash | $6.17 | |
| DeepSeek V4 Flash + Mimo V2.5 | $8.72 | |
| DeepSeek V4 Flash + MiniMax M3 | $10.24 | |
| DeepSeek V4 Flash + Kimi K2.7 Code | $24.16 | |
| Mimo V2.5 Pro + Qwen 3.5 Flash | $8.91 | |
| Mimo V2.5 Pro + Mimo V2.5 | $11.46 | |
| Mimo V2.5 Pro + MiniMax M3 | $12.98 | |
| Mimo V2.5 Pro + Kimi K2.7 Code | $26.91 | |

## Cache Pricing Analysis

Models with cache read pricing offer significant savings for repeated context.
Your effective cache hit rate: 83.5%

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
