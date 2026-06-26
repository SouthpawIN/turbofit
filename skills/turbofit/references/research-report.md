# Turbofit Model Research Report — 2026-06-26

## Data Sources
- Model pricing: OpenRouter API (`/api/v1/models`) — live
- Usage data: Hermes state.db (`/home/sovthpaw/.hermes/profiles/senter/state.db`) — real user data

## Your Real Usage (from Hermes Insights)

- Active days: 6
- Total sessions: 96
- Total messages: 4,549
- Total tool calls: 2,460
- Total input tokens: 23,405,773
- Total output tokens: 859,443
- Total cache read tokens: 182,675,694
- Effective cache hit rate: 88.6%
- Avg input/session: 243,810
- Avg output/session: 8,953
- Avg sessions/day: 16.0
- Total estimated cost: $56.96

### Models You've Actually Used

| Model | Sessions | Input Tokens | Output Tokens | Cache Read | Est. Cost |
|-------|----------|-------------|--------------|------------|-----------|
| z-ai/glm-5.2 | 87 | 22,383,623 | 849,308 | 182,675,694 | $56.69 |
| qwen/qwen3.6-plus | 2 | 719,357 | 8,455 | 0 | $0.25 |
| qwen/qwen3.5-flash-02-23 | 7 | 302,793 | 1,680 | 0 | $0.02 |

### By Platform

| Platform | Sessions | Input | Output | Cost |
|----------|----------|-------|--------|------|
| discord | 8 | 18,976,567 | 511,565 | $50.02 |
| subagent | 52 | 2,046,479 | 223,480 | $3.91 |
| cron | 26 | 1,352,614 | 113,644 | $2.76 |
| tui | 5 | 526,968 | 4,952 | $0.11 |
| cli | 5 | 503,145 | 5,802 | $0.17 |

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
| deepseek/deepseek-v3.2 | $6.89 | 90% cache savings |
| deepseek/deepseek-v3.2-exp | $33.36 |  |
| deepseek/deepseek-v4-flash | $4.04 | 78% cache savings |
| deepseek/deepseek-v4-pro | $9.90 | 99% cache savings |
| minimax/minimax-m3 | $15.37 | 80% cache savings; vision |
| moonshotai/kimi-k2.6 | $38.36 | 78% cache savings; vision |
| moonshotai/kimi-k2.7-code | $40.44 | 80% cache savings; vision |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | $48.53 |  |
| nvidia/nemotron-3-nano-30b-a3b | $6.71 |  |
| nvidia/nemotron-3-nano-30b-a3b:free | FREE |  |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | FREE | vision |
| nvidia/nemotron-3-super-120b-a12b | $11.67 |  |
| nvidia/nemotron-3-super-120b-a12b:free | FREE |  |
| nvidia/nemotron-3-ultra-550b-a55b | $26.47 | 80% cache savings |
| nvidia/nemotron-3-ultra-550b-a55b:free | FREE |  |
| nvidia/nemotron-3.5-content-safety:free | FREE | vision |
| nvidia/nemotron-nano-12b-v2-vl:free | FREE | vision |
| nvidia/nemotron-nano-9b-v2:free | FREE |  |
| qwen/qwen3.5-flash-02-23 | $8.72 | vision |
| qwen/qwen3.6-flash | $26.78 | vision |
| qwen/qwen3.6-plus | $46.41 | vision |
| qwen/qwen3.7-max | $58.66 | 80% cache savings |
| qwen/qwen3.7-plus | $16.39 | 80% cache savings; vision |
| xiaomi/mimo-v2.5 | $13.49 | vision |
| xiaomi/mimo-v2.5-pro | $9.89 | 99% cache savings |
| z-ai/glm-5.1 | $45.14 | 81% cache savings |
| z-ai/glm-5.2 | $44.19 | 81% cache savings |

### Recommended Pairings (projected on YOUR usage)

| Pairing | Monthly Cost | Notes |
|---------|-------------|-------|
| GLM 5.2 + Qwen 3.5 Flash | $22.91 | |
| GLM 5.2 + Mimo V2.5 | $25.77 | |
| GLM 5.2 + MiniMax M3 | $26.90 | |
| GLM 5.2 + Kimi K2.7 Code | $41.94 | |
| Qwen 3.7 MAX + Qwen 3.5 Flash | $28.70 | |
| Qwen 3.7 MAX + Mimo V2.5 | $31.56 | |
| Qwen 3.7 MAX + MiniMax M3 | $32.69 | |
| Qwen 3.7 MAX + Kimi K2.7 Code | $47.73 | |
| DeepSeek V4 Pro + Qwen 3.5 Flash | $9.19 | |
| DeepSeek V4 Pro + Mimo V2.5 | $12.05 | |
| DeepSeek V4 Pro + MiniMax M3 | $13.18 | |
| DeepSeek V4 Pro + Kimi K2.7 Code | $28.22 | |
| DeepSeek V4 Flash + Qwen 3.5 Flash | $6.85 | |
| DeepSeek V4 Flash + Mimo V2.5 | $9.71 | |
| DeepSeek V4 Flash + MiniMax M3 | $10.84 | |
| DeepSeek V4 Flash + Kimi K2.7 Code | $25.88 | |
| Mimo V2.5 Pro + Qwen 3.5 Flash | $9.19 | |
| Mimo V2.5 Pro + Mimo V2.5 | $12.05 | |
| Mimo V2.5 Pro + MiniMax M3 | $13.18 | |
| Mimo V2.5 Pro + Kimi K2.7 Code | $28.22 | |

## Cache Pricing Analysis

Models with cache read pricing offer significant savings for repeated context.
Your effective cache hit rate: 88.6%

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
