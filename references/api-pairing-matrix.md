# API Pairing Matrix — Optimal Main+Aux by Price & Context (2026-06-25)

Pairings use the user's specified model universe. Prices in $/M tokens (input/output).
Aux offset: 40-85% of total tokens route to aux (compression, vision, web summarization, skill search, title gen, MCP routing, approval scoring — 11 slots).

## Model Universe

### Text-Only Mains (no vision — MUST pair with vision aux)

| Model | Max Ctx | Input $/M | Output $/M | Provider(s) | Through Nous? |
|-------|---------|-----------|-----------|--------------|---------------|
| GLM 5.2 | 1M | $0.95 (OR) / $1.40 (Z.AI) | $3.00 (OR) / $4.40 (Z.AI) | Z.AI, OpenRouter, Nous | ✅ `z-ai/glm-5.2` |
| Qwen 3.7 MAX | 1M | $1.25 | $3.75 | DashScope, OpenRouter, Nous | ✅ `qwen/qwen3.7-max` |
| DeepSeek V4 Pro | 1M | $0.435 (DS API) / $1.74 (OR cache miss) | $0.87 (DS) / $3.48 (OR) | DeepSeek, OpenRouter, Nous, NIM (free) | ✅ `deepseek/deepseek-v4-pro` |
| DeepSeek V4 Flash | 1M | $0.09 (OR) / $0.14 (DS) | $0.18 (OR) / $0.28 (DS) | DeepSeek, OpenRouter, Nous, NIM (free) | ✅ `deepseek/deepseek-v4-flash` |
| Mimo V2.5 Pro | 1M | ~$1.00 | ~$3.00 | Xiaomi, OpenRouter, Nous | ✅ `xiaomi/mimo-v2.5-pro` |

### Vision-Capable Models (can serve as main OR aux)

| Model | Max Ctx | Input $/M | Output $/M | Provider(s) | Through Nous? |
|-------|---------|-----------|-----------|--------------|---------------|
| Mimo V2.5 | 1M | $0.105 | $0.28 | Xiaomi, OpenRouter, Nous | ✅ `xiaomi/mimo-v2.5` |
| Qwen 3.7 Plus | 1M | $0.32 (OR) / $0.50 (Fireworks) | $1.28 (OR) / $3.00 (Fireworks) | DashScope, OpenRouter, Fireworks, Nous | ✅ `qwen/qwen3.7-plus` |
| MiniMax M3 | 1M | ~$0.30 | ~$1.20 | NIM (free), OpenRouter, Nous | ✅ `minimaxai/minimax-m3` |
| Kimi K2.7 Code | 256K | $0.74 (OR) / $0.95 (Requesty) | $3.50 (OR) / $4.00 (Requesty) | OpenRouter, Moonshot, Nous | ✅ `moonshotai/kimi-k2.7-code` |
| Kimi K2.6 | 1M | $0.60 | $2.50 | OpenRouter (was free), Moonshot | ✅ `moonshotai/kimi-k2.6` |
| Qwen 3.6 Plus | 1M | FREE (OR preview) | FREE (OR preview) | OpenRouter | ❌ Not through Nous |
| Qwen 3.5 Flash | 1M | $0.065 | $0.26 | OpenRouter, Nous | ✅ `qwen/qwen3.5-flash-02-23` |
| Qwen 3.6 Flash | 262K | $0.1875 | $1.125 | OpenRouter, Nous | ✅ `qwen/qwen3.6-flash` |

### Free Endpoints (outside Nous gateway)

| Model | Via | Ctx | Vision | Notes |
|-------|-----|-----|--------|-------|
| DeepSeek V4 Pro | NIM | 1M | ❌ | `NVIDIA_API_KEY`, ~1000 RPM |
| DeepSeek V4 Flash | NIM | 1M | ❌ | `NVIDIA_API_KEY`, ~1000 RPM |
| MiniMax M3 | NIM | 1M | ✅ | `NVIDIA_API_KEY`, ~1000 RPM |
| Qwen 3.6 Plus | OpenRouter (`:free`) | 1M | ✅ | Rate-limited, 10% bonus credits |
| Kimi K2.6 | OpenRouter (Crucible) | 1M | ✅ | Rate-limited, may be unreliable |

## Provider & Gateway Indicators

| Indicator | Meaning |
|-----------|---------|
| 🟢 **NOUS** | Both main+aux through Nous. Tool Gateway is active (Firecrawl, FAL, OpenAI TTS, Browser Use) — TG is a subscription feature, active whenever the user has a Nous Portal subscription regardless of which models are selected. |
| 🟡 **NOUS+OR** | Main through Nous, aux through OpenRouter (10% credit bonus on OR credits) |
| 🟠 **NOUS+NIM** | Main through Nous, aux through NIM (free, separate key) |
| 🔵 **OR** | Both through OpenRouter (10% bonus, no TG) |
| ⚪ **NIM** | Both through NIM (free, no TG, separate key) |
| 🆓 **FREE** | Zero cost |

**Note:** The Nous Tool Gateway is a subscription feature — it is active whenever the user has a Nous Portal subscription, regardless of which models are used for main or aux. The gateway indicators above show routing paths, not TG activation. TG covers Firecrawl (web search), FAL (image generation), OpenAI TTS, and Browser Use.

## Pairing Matrix — by Price Tier × Context Level

### TIER 1: 1M Context (maximum)

| # | Main | Aux | Gateway | Main Cost | Aux Cost | Est. Blended (60% aux) | Notes |
|---|------|-----|---------|-----------|----------|----------------------|-------|
| 1A | GLM 5.2 | Qwen 3.6 Plus | 🟡 NOUS+OR | $0.95/$3.00 | FREE | ~$0.38/$1.20 blended | Top reasoning + free vision aux. TG active on main. |
| 1B | GLM 5.2 | Mimo V2.5 | 🟢 NOUS+TG | $0.95/$3.00 | $0.105/$0.28 | ~$0.50/$1.87 blended | Same-family vision aux, both through Nous. Full TG. |
| 1C | GLM 5.2 | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.95/$3.00 | $0.065/$0.26 | ~$0.44/$1.42 blended | Cheapest vision aux through Nous. Full TG. |
| 1D | Qwen 3.7 MAX | Qwen 3.6 Plus | 🟡 NOUS+OR | $1.25/$3.75 | FREE | ~$0.50/$1.50 blended | #2 reasoning + free vision aux. |
| 1E | Qwen 3.7 MAX | Mimo V2.5 | 🟢 NOUS+TG | $1.25/$3.75 | $0.105/$0.28 | ~$0.56/$2.27 blended | Premium main + cheap vision aux. Full TG. |
| 1F | DeepSeek V4 Pro | MiniMax M3 | 🟠 NOUS+NIM | $0.435/$0.87 (DS) | FREE (NIM) | ~$0.17/$0.35 blended | Zero-cost aux via NIM. Main via Nous for TG. |
| 1G | DeepSeek V4 Pro | Qwen 3.6 Plus | 🟡 NOUS+OR | $0.435/$0.87 | FREE | ~$0.17/$0.35 blended | Free vision aux via OR. |
| 1H | DeepSeek V4 Pro | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.435/$0.87 | $0.065/$0.26 | ~$0.20/$0.47 blended | Cheapest full-TG pairing at 1M. |
| 1I | DeepSeek V4 Pro | DeepSeek V4 Flash | ⚪ NIM | FREE | FREE | $0.00/$0.00 | Both free on NIM. No vision — text-only workloads. |
| 1J | DeepSeek V4 Flash | MiniMax M3 | ⚪ NIM | FREE | FREE | $0.00/$0.00 | Free text + free vision. Best zero-cost pair. |
| 1K | DeepSeek V4 Flash | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.09/$0.18 | $0.065/$0.26 | ~$0.07/$0.21 blended | Ultra-cheap, full TG. |
| 1L | Mimo V2.5 Pro | Mimo V2.5 | 🟢 NOUS+TG | ~$1.00/$3.00 | $0.105/$0.28 | ~$0.46/$1.51 blended | Same-family pair. Full TG. |
| 1M | MiniMax M3 | Qwen 3.6 Plus | 🔵 OR | ~$0.30/$1.20 | FREE | ~$0.12/$0.48 blended | Vision main + free vision aux. 10% OR bonus. |
| 1N | MiniMax M3 | Qwen 3.5 Flash | 🟢 NOUS+TG | ~$0.30/$1.20 | $0.065/$0.26 | ~$0.16/$0.58 blended | Vision main + cheap vision aux. Full TG. |

### TIER 2: 512K Context

Same pairings as Tier 1 — all listed models support 512K (subset of their 1M max).
Key difference: **tiered pricing** kicks in above 256K for some models (Qwen 3.6 Plus, Qwen 3.7 Plus). At 512K, expect ~1.5-2x output pricing for those models.

| Best Value Pair | Gateway | Blended Cost | Notes |
|----------------|---------|-------------|-------|
| DeepSeek V4 Flash + Qwen 3.5 Flash | 🟢 NOUS+TG | ~$0.07/$0.21 | Cheapest 512K with TG |
| DeepSeek V4 Pro + MiniMax M3 | 🟠 NOUS+NIM | ~$0.17/$0.35 | Free vision aux via NIM |
| GLM 5.2 + Qwen 3.6 Plus | 🟡 NOUS+OR | ~$0.57/$1.80 | Premium reasoning, free aux |

### TIER 3: 262K Context

At 262K, cheaper models become available and Kimi K2.7 Code enters the picture.

| # | Main | Aux | Gateway | Main Cost | Aux Cost | Est. Blended | Notes |
|---|------|-----|---------|-----------|----------|-------------|-------|
| 3A | GLM 5.2 | Kimi K2.7 Code | 🟢 NOUS+TG | $0.95/$3.00 | $0.74/$3.50 | ~$0.83/$3.30 | Top main + strong coding aux with vision |
| 3B | DeepSeek V4 Flash | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.09/$0.18 | $0.065/$0.26 | ~$0.07/$0.21 | Cheapest pair with TG at 262K |
| 3C | DeepSeek V4 Flash | MiniMax M3 | ⚪ NIM | FREE | FREE | $0.00 | Free text + free vision |
| 3D | Mimo V2.5 Pro | Mimo V2.5 | 🟢 NOUS+TG | ~$1.00/$3.00 | $0.105/$0.28 | ~$0.46/$1.51 | Same-family, full TG |
| 3E | Qwen 3.6 Flash | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.1875/$1.125 | $0.065/$0.26 | ~$0.11/$0.55 | Cheap 262K pair |
| 3F | Kimi K2.7 Code | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.74/$3.50 | $0.065/$0.26 | ~$0.33/$1.42 | Coding main + vision aux |

### TIER 4: 132K Context (Hermes floor is 64K — 132K is comfortable)

Budget territory. Cheapest viable pairings for users who only need short context.

| # | Main | Aux | Gateway | Main Cost | Aux Cost | Est. Blended | Notes |
|---|------|-----|---------|-----------|----------|-------------|-------|
| 4A | DeepSeek V4 Flash | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.09/$0.18 | $0.065/$0.26 | ~$0.07/$0.21 | Ultra-cheap, full TG |
| 4B | DeepSeek V4 Flash | MiniMax M3 | ⚪ NIM | FREE | FREE | $0.00 | Zero cost |
| 4C | Qwen 3.5 Flash | Qwen 3.5 Flash | 🟢 NOUS+TG | $0.065/$0.26 | $0.065/$0.26 | $0.065/$0.26 | Same model for both — simplest setup |
| 4D | Mimo V2.5 | Mimo V2.5 | 🟢 NOUS+TG | $0.105/$0.28 | $0.105/$0.28 | $0.105/$0.28 | Self-pair, cheapest vision-vision |

## Price Tier Summary

| Tier | Target Blended Cost | Best Pairing | Gateway | Context |
|------|-------------------|-------------|---------|---------|
| 🆓 Free | $0.00 | DeepSeek V4 Flash + MiniMax M3 (NIM) | ⚪ NIM | 1M |
| 💰 Budget | <$0.25/M blended | DeepSeek V4 Flash + Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | 1M |
| 🎯 Mid | $0.25-$1.00/M blended | DeepSeek V4 Pro + Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | 1M |
| ⭐ Premium | $1.00-$2.00/M blended | GLM 5.2 + Mimo V2.5 (Nous) | 🟢 NOUS+TG | 1M |
| 👑 Top | >$2.00/M blended | GLM 5.2 + Kimi K2.7 Code (Nous) | 🟢 NOUS+TG | 262K |

## Pairing Rules

1. **Text-only mains MUST pair with vision aux.** GLM 5.2, Qwen 3.7 MAX, DeepSeek V4 Pro/Flash, Mimo V2.5 Pro → need vision aux (MiniMax M3, Mimo V2.5, Qwen 3.7 Plus, Qwen 3.6 Plus, Qwen 3.5 Flash, Kimi K2.6/K2.7).
2. **Vision mains can pair with any aux** — but a vision aux is still preferred for the vision slot specifically.
3. **Same-family pairings are efficient** — Mimo Pro/V2.5, Qwen MAX/Plus, DeepSeek Pro/Flash share tokenizers and architecture, reducing routing overhead.
4. **Free Qwen 3.6 Plus is the best free aux** — 1M ctx, vision, free on OpenRouter. Use via OR directly (10% bonus). Not available through Nous gateway.
5. **NIM free endpoints are the best zero-cost mains** — DeepSeek V4 Pro/Flash and MiniMax M3 are free via `NVIDIA_API_KEY`. No Tool Gateway.
6. **Nous Tool Gateway is active when main is through Nous** — even if aux routes elsewhere, the TG covers web search, image gen, TTS, browser.
7. **10% OpenRouter credit bonus** — when adding credits to OR, you get 10% extra. Factor into cost calculations for OR-routed aux.

## Recommended Default Pairings by Hardware Tier

### Beefy (≥24GB VRAM, dual GPU) — local scaling ladder + API fallback

| Context Level | Main | Aux | Gateway | Notes |
|--------------|------|-----|---------|-------|
| 1M | Darwin 28B Reason (local) | Carnice 35A3B (local) | N/A (local) | Flagship dual-local |
| 512K | Darwin 28B (local) | Carnice cpu-moe (local) | N/A | Aux experts to CPU |
| 262K | Prism Eagle 27B (local) | Carnice (local) | N/A | Swap main to lighter model |
| 132K | Darwin Apex 35A3B (local) | API (Qwen 3.6 Plus free) | 🟡 via OR | Drop local aux, API vision aux |
| API fallback | GLM 5.2 (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | Full fallback to cloud |

### Modest (8-24GB VRAM, single GPU) — API-first with optional local aux

| Context Level | Main | Aux | Gateway | Notes |
|--------------|------|-----|---------|-------|
| 1M | DeepSeek V4 Pro (Nous) | Qwen 3.6 Plus (OR free) | 🟡 NOUS+OR | Best value 1M pair |
| 512K | DeepSeek V4 Pro (Nous) | MiniMax M3 (NIM free) | 🟠 NOUS+NIM | Free vision aux via NIM |
| 262K | DeepSeek V4 Flash (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | Cheap pair, full TG |
| 132K | DeepSeek V4 Flash (NIM free) | MiniMax M3 (NIM free) | ⚪ NIM | Zero cost |

### Thin (<8GB VRAM / no GPU) — API-only, cost-minimized

| Context Level | Main | Aux | Gateway | Notes |
|--------------|------|-----|---------|-------|
| 1M | DeepSeek V4 Flash (NIM free) | MiniMax M3 (NIM free) | ⚪ NIM | Zero cost, both free |
| 512K | DeepSeek V4 Flash (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | Full TG, ultra-cheap |
| 262K | DeepSeek V4 Flash (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | Same pair, lower ctx |
| 132K | Qwen 3.5 Flash (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | Simplest viable setup |

## Blended Cost Calculation

Blended cost assumes 60% of tokens route to aux (midpoint of 40-85% range). Actual split depends on workload:

```
blended_input = (main_in × 0.40) + (aux_in × 0.60)
blended_output = (main_out × 0.40) + (aux_out × 0.60)
```

For heavy-agent workloads (long conversations, lots of compression, vision tasks), aux offset approaches 85%. For simple Q&A, aux offset is closer to 40%.

## See Also

- `api-model-rankings.md` — individual model rankings and pricing
- `scaling-ladder.md` — full VRAM scaling ladders for all hardware tiers
- `curated-lineup.md` — the curated local + API picks
- `binary-selection.md` — atomic fork vs stock decision tree
