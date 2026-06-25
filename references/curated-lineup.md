# Curated Model Lineup

Opinionated picks for main + aux roles, local + API. **Good intelligence is always available no matter your hardware.**

## Hardware Tiers

| Tier | VRAM | Setup | Default Main | Default Aux | Gateway |
|------|------|-------|-------------|-------------|---------|
| **Beefy** | ≥24GB | Local main + local aux | 27-28B dense (Q4) | 35B MoE 3B-active | N/A (local) |
| **Modest** | 8-24GB | API main + free/cheap aux | DeepSeek V4 Pro (API) | Qwen 3.6 Plus (OR free) or MiniMax M3 (NIM free) | 🟡/⚪ |
| **Thin** | <8GB or no GPU | API main + API aux | DeepSeek V4 Flash (NIM free) | MiniMax M3 (NIM free) | ⚪ NIM |

`serve auto` detects your tier. `--api` forces API mode. `--free` restricts to free endpoints.

## Featured Local Model Archetypes (Beefy tier)

Users register their own local GGUF models. The archetypes below describe what *kind* of model fits each role. `serve recommend` scans the catalog and picks the best match.

### Main archetypes (by VRAM budget)

| Archetype | Size (Q4) | VRAM w/ KV | Tier | Speed (typical) | Key Feature |
|-----------|----------|------------|------|----------------|-------------|
| 27-28B dense (reasoning) | 14-17 GB | ~22 GB | S | 35-40 tok/s | Smartest dense, 1M ctx capable |
| 27B hybrid/Mamba | 14 GB | ~16 GB | S | 100-120 tok/s | Mamba2+GDN, smallest 27B |
| 27B dense + MTP | 17 GB | ~22 GB | SF | ~100 tok/s | MTP + multimodal |
| 27B dense (coder) | 17 GB | ~22 GB | SF | ~100 tok/s | Coding-optimized MTP |
| 35B MoE (3B active) | 11-17 GB | ~11-17 GB | SD | 30-110 tok/s | MoE, vision, 1M ctx |

### Aux archetype

| Archetype | Size (Q4) | VRAM | Tier | Speed | Key Feature |
|-----------|----------|------|------|-------|-------------|
| 35B MoE (3B active) | 11 GB | ~11 GB (17 GB w/o cpu-moe) | SF | 30 tok/s (10 w/ cpu-moe) | Always-on, vision, 1M ctx |
| Small multimodal (≤3B) | 2-3 GB | ~3 GB | C | 30-80 tok/s | Ultra-light aux for Modest tier |

### What users register

Users add their own models via `serve register`:

```bash
# Register your main model
serve register my-main /path/to/28B-model.Q4_K_M.gguf --port 11500

# Register your aux model
serve register my-aux /path/to/35B-MoE-model.gguf --port 8082

# Then edit ~/.config/turbofit/models.yaml to set:
#   tier, vision, role, presets, mmproj, binary, etc.
```

The catalog schema is fully documented in SKILL.md. `serve recommend` ranks catalog entries by fit (ctx≥64K, tok/s≥25, vision bonus, tier priority).

## API Main (ranked by performance — free first)

### Text-only mains (must pair with vision aux)

| Tier | Model | Vision | Cost | Context | Through Nous? |
|------|-------|--------|------|---------|---------------|
| S | GLM 5.2 | ❌ | $0.95/$3.00 (OR) / $1.40/$4.40 (Z.AI) | 1M | ✅ `z-ai/glm-5.2` |
| S | Qwen 3.7 MAX | ❌ | $1.25/$3.75 | 1M | ✅ `qwen/qwen3.7-max` |
| S | DeepSeek V4 Pro | ❌ | $0.435/$0.87 (DS) / FREE (NIM) | 1M | ✅ `deepseek/deepseek-v4-pro` |
| SF | DeepSeek V4 Flash | ❌ | $0.09/$0.18 (OR) / FREE (NIM) | 1M | ✅ `deepseek/deepseek-v4-flash` |
| SF | Mimo V2.5 Pro | ❌ | ~$1.00/$3.00 | 1M | ✅ `xiaomi/mimo-v2.5-pro` |

### Vision-capable mains

| Tier | Model | Vision | Cost | Context | Through Nous? |
|------|-------|--------|------|---------|---------------|
| SF | MiniMax M3 | ✅ | ~$0.30/$1.20 / FREE (NIM) | 1M | ✅ `minimaxai/minimax-m3` |
| SF | Qwen 3.7 Plus | ✅ | $0.32/$1.28 (OR) | 1M | ✅ `qwen/qwen3.7-plus` |
| F | Mimo V2.5 | ✅ | $0.105/$0.28 | 1M | ✅ `xiaomi/mimo-v2.5` |
| SF | Kimi K2.7 Code | ✅ | $0.74/$3.50 | 256K | ✅ `moonshotai/kimi-k2.7-code` |
| F | Qwen 3.6 Plus | ✅ | FREE (OR preview) | 1M | ❌ OR only |
| SD | Qwen 3.5 Flash | ✅ | $0.065/$0.26 | 1M | ✅ `qwen/qwen3.5-flash-02-23` |

## API Aux (ranked by vision > speed > cost — free first)

| Tier | Model | Vision | Cost | Context | Through Nous? |
|------|-------|--------|------|---------|---------------|
| F | Qwen 3.6 Plus | ✅ | FREE (OR) | 1M | ❌ OR only |
| SF | MiniMax M3 | ✅ | FREE (NIM) | 1M | ✅ |
| F | Mimo V2.5 | ✅ | $0.105/$0.28 | 1M | ✅ |
| SD | Qwen 3.5 Flash | ✅ | $0.065/$0.26 | 1M | ✅ |
| SF | Kimi K2.6 | ✅ | $0.60/$2.50 | 1M | ✅ |
| SF | Kimi K2.7 Code | ✅ | $0.74/$3.50 | 256K | ✅ |

## Cascading Scaling Ladders

### Beefy (≥24GB) — 7 steps

| Step | State | Main Archetype | Aux | Context | Action |
|------|-------|---------------|-----|---------|--------|
| 1 | Ideal | 27-28B dense (Q4) | 35B MoE (local) | 1M | Nothing |
| 2 | Mild pressure | 27-28B dense | 35B MoE (cpu-moe) | 1M | Offload aux to CPU |
| 3 | Moderate | 27-28B dense | 35B MoE | 512K | Drop context |
| 4 | High | 27-28B dense | API vision (free) | 262K | Drop local aux |
| 5 | Critical | 27B hybrid/Mamba | API vision (cheap) | 262K | Swap to lighter main |
| 6 | Extreme | 35B MoE (3B active) | API vision (cheap) | 132K | Swap to MoE main |
| 7 | API-only | API main (Nous) | API vision (Nous) | 1M | Full cloud fallback |

### Modest (8-24GB) — 5 steps

| Step | State | Main | Aux | Context | Gateway |
|------|-------|------|-----|---------|---------|
| 1 | Comfortable | DeepSeek V4 Pro (Nous) | Qwen 3.6 Plus (OR free) | 1M | 🟡 NOUS+OR |
| 2 | Budget | DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | 1M | ⚪ NIM |
| 3 | Tight | DeepSeek V4 Flash (Nous) | Qwen 3.5 Flash (Nous) | 262K | 🟢 NOUS+TG |
| 4 | Minimal | DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | 132K | ⚪ NIM |
| 5 | Local aux | DeepSeek V4 Pro (Nous) | Small local model | 1M | 🟡 NOUS+local |

### Thin (<8GB) — 4 steps

| Step | State | Main | Aux | Context | Gateway |
|------|-------|------|-----|---------|---------|
| 1 | Zero cost | DeepSeek V4 Flash (NIM) | MiniMax M3 (NIM) | 1M | ⚪ NIM |
| 2 | Budget TG | DeepSeek V4 Flash (Nous) | Qwen 3.5 Flash (Nous) | 1M | 🟢 NOUS+TG |
| 3 | Mid | DeepSeek V4 Pro (Nous) | Qwen 3.5 Flash (Nous) | 1M | 🟢 NOUS+TG |
| 4 | Premium | GLM 5.2 (Nous) | Qwen 3.5 Flash (Nous) | 1M | 🟢 NOUS+TG |

See `scaling-ladder.md` for full step-by-step details.

## Pairing Rules

1. **Text-only mains MUST pair with vision aux.** GLM 5.2, Qwen 3.7 MAX, DeepSeek V4 Pro/Flash, Mimo V2.5 Pro → need vision aux.
2. **Vision mains can pair with any aux.**
3. **Same-family pairings are efficient** — Mimo Pro/V2.5, Qwen MAX/Plus, DeepSeek Pro/Flash share tokenizers.
4. **Free Qwen 3.6 Plus is the best free aux** — 1M ctx, vision, free on OpenRouter (not through Nous).
5. **NIM free endpoints are the best zero-cost mains** — DeepSeek V4 Pro/Flash, MiniMax M3.
6. **Nous Tool Gateway active when main is through Nous** — covers Firecrawl, FAL, OpenAI TTS, Browser Use.
7. **10% OpenRouter credit bonus** — factor into cost calculations.

See `api-pairing-matrix.md` for the complete pairing matrix with all combinations.

## Key Principles

- Main is always protected until Step 5 (Beefy)
- MoE expert offload is the first pressure valve
- Context drops only when absolutely necessary
- API aux kicks in when local aux can't fit
- Each step preserves maximum intelligence while respecting VRAM
- **Zero-cost full Hermes-Agent capability is available to anyone regardless of hardware**
