# Turbofit Model Research Report — 2026-06-25

## Summary
- Models tracked: 27
- Data source: OpenRouter API (`/api/v1/models`)
- Free models: 7
- Models with cache pricing: 11
- Models with vision: 11

## Model Pricing (live from OpenRouter API)

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
| nvidia/nemotron-3-super-120b-a12b | no | $0.0900 | - | $0.4500 | 1000K |  |
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

## Monthly Cost Projections (single model, no aux)

Assumes all tokens go to this model (no aux offset). Real cost will be lower if using a cheaper aux.

| Slug | Light user (hobbyist) | Moderate user (daily agent work) | Heavy user (multi-agent, long sessions) | Extreme (24/7 agent fleet) |
|------|------|------|------|------|
| deepseek/deepseek-v3.2 | $0.40 | $2.88 | $14.16 | $56.63 |
| deepseek/deepseek-v3.2-exp | $0.48 | $3.41 | $16.76 | $67.05 |
| deepseek/deepseek-v4-flash | $0.14 | $0.99 | $4.50 | $17.37 |
| deepseek/deepseek-v4-pro | $0.65 | $4.45 | $19.66 | $74.74 |
| minimax/minimax-m3 | $0.63 | $4.72 | $21.60 | $84.24 |
| moonshotai/kimi-k2.6 | $1.60 | $12.27 | $56.45 | $221.17 |
| moonshotai/kimi-k2.7-code | $1.71 | $12.94 | $59.40 | $232.29 |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | $0.65 | $4.56 | $22.50 | $90.00 |
| nvidia/nemotron-3-nano-30b-a3b | $0.12 | $0.93 | $4.50 | $18.00 |
| nvidia/nemotron-3-nano-30b-a3b:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-3-super-120b-a12b | $0.24 | $1.89 | $9.11 | $36.45 |
| nvidia/nemotron-3-super-120b-a12b:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-3-ultra-550b-a55b | $1.11 | $8.34 | $38.25 | $149.40 |
| nvidia/nemotron-3-ultra-550b-a55b:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-3.5-content-safety:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-nano-12b-v2-vl:free | FREE | FREE | FREE | FREE |
| nvidia/nemotron-nano-9b-v2:free | FREE | FREE | FREE | FREE |
| qwen/qwen3.5-flash-02-23 | $0.16 | $1.21 | $5.85 | $23.40 |
| qwen/qwen3.6-flash | $0.56 | $4.39 | $21.09 | $84.38 |
| qwen/qwen3.6-plus | $0.97 | $7.61 | $36.56 | $146.25 |
| qwen/qwen3.7-max | $2.29 | $16.65 | $75.94 | $294.75 |
| qwen/qwen3.7-plus | $0.67 | $5.03 | $23.04 | $89.86 |
| xiaomi/mimo-v2.5 | $0.22 | $1.62 | $7.88 | $31.50 |
| xiaomi/mimo-v2.5-pro | $0.65 | $4.45 | $19.66 | $74.74 |
| z-ai/glm-5.1 | $1.83 | $13.34 | $60.80 | $236.00 |
| z-ai/glm-5.2 | $1.78 | $12.98 | $59.17 | $229.77 |

## Pairing Cost Projections (main + aux)

Based on Hermes Agent aux offset (40-85% of tokens route to aux).

| Pairing | Light user (hobbyist) | Moderate user (daily agent work) | Heavy user (multi-agent, long sessions) | Extreme (24/7 agent fleet) |
|---------|------|------|------|------|
| GLM 5.2 + Qwen 3.5 Flash | $0.97 | $5.92 | $21.85 | $74.99 |
| GLM 5.2 + Mimo V2.5 | $1.00 | $6.16 | $23.27 | $81.07 |
| GLM 5.2 + MiniMax M3 | $1.21 | $8.02 | $32.87 | $120.62 |
| GLM 5.2 + Kimi K2.7 Code | $1.74 | $12.95 | $59.33 | $231.66 |
| Qwen 3.7 MAX + Qwen 3.5 Flash | $1.23 | $7.39 | $26.88 | $91.24 |
| Qwen 3.7 MAX + Mimo V2.5 | $1.26 | $7.63 | $28.29 | $97.31 |
| Qwen 3.7 MAX + MiniMax M3 | $1.46 | $9.49 | $37.90 | $136.87 |
| Qwen 3.7 MAX + Kimi K2.7 Code | $2.00 | $14.42 | $64.36 | $247.91 |
| DeepSeek V4 Pro + Qwen 3.5 Flash | $0.40 | $2.51 | $9.99 | $36.24 |
| DeepSeek V4 Pro + Mimo V2.5 | $0.43 | $2.75 | $11.41 | $42.31 |
| DeepSeek V4 Pro + MiniMax M3 | $0.64 | $4.61 | $21.02 | $81.87 |
| DeepSeek V4 Pro + Kimi K2.7 Code | $1.18 | $9.54 | $47.48 | $192.90 |
| DeepSeek V4 Flash + Qwen 3.5 Flash | $0.15 | $1.12 | $5.45 | $21.89 |
| DeepSeek V4 Flash + Mimo V2.5 | $0.18 | $1.37 | $6.86 | $27.97 |
| DeepSeek V4 Flash + MiniMax M3 | $0.39 | $3.23 | $16.47 | $67.52 |
| DeepSeek V4 Flash + Kimi K2.7 Code | $0.92 | $8.16 | $42.93 | $178.56 |
| Mimo V2.5 Pro + Qwen 3.5 Flash | $0.40 | $2.51 | $9.99 | $36.24 |
| Mimo V2.5 Pro + Mimo V2.5 | $0.43 | $2.75 | $11.41 | $42.31 |
| Mimo V2.5 Pro + MiniMax M3 | $0.64 | $4.61 | $21.02 | $81.87 |
| Mimo V2.5 Pro + Kimi K2.7 Code | $1.18 | $9.54 | $47.48 | $192.90 |

## Free Models (zero cost)

- **nvidia/nemotron-3-nano-30b-a3b:free** — 256K ctx, text-only
- **nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free** — 256K ctx, vision
- **nvidia/nemotron-3-super-120b-a12b:free** — 1000K ctx, text-only
- **nvidia/nemotron-3-ultra-550b-a55b:free** — 1000K ctx, text-only
- **nvidia/nemotron-3.5-content-safety:free** — 128K ctx, vision
- **nvidia/nemotron-nano-12b-v2-vl:free** — 128K ctx, vision
- **nvidia/nemotron-nano-9b-v2:free** — 128K ctx, text-only

## Cache Pricing Analysis

Models with cache read pricing offer significant savings for repeated context (common in Hermes Agent long sessions):

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

## Action Items

1. Compare prices above with `model-database.yaml` — update any that changed
2. Add any NEW models not yet in the database
3. Check if any free models changed status
4. Verify cache pricing is reflected in the pairing matrix
5. Run `bash scripts/sync-github.sh` after updates
