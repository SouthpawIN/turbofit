# Scaling ladder — universal VRAM + API adaptation

The scaling ladder is what makes turbofit opinionated. Its philosophy: **good intelligence is always available no matter what hardware you have** — from dual 24GB GPUs down to no GPU at all.

Three modes:
1. **Pull mode:** user invokes `serve downscale` → probes VRAM → adapts.
2. **Push mode (planned):** cron watches VRAM → triggers `serve downscale` automatically.
3. **API mode:** no local GPU — ladder is pure cost optimization across API pairings.

## Hardware Tier Detection

`serve auto` probes `nvidia-smi` (or falls back to API-only). Three tiers:

| Tier | Total VRAM | Typical GPUs | Strategy |
|------|-----------|-------------|----------|
| **Beefy** | ≥24GB | 2× RTX 3090/4090, or single 24GB+ | Local main + local aux |
| **Modest** | 8-24GB | RTX 3060/4060 (12GB), RTX 3070/4070 (8-16GB) | API main + local or API aux |
| **Thin** | <8GB or no NVIDIA GPU | Integrated graphics, GT 1030, no GPU | API main + API aux |

---

## Context-Level Ladder (the 4 tiers)

Each context level corresponds to a VRAM budget (local) or a price tier (API):

| Level | Context | Local VRAM Need (Beefy) | API Price Tier | Philosophy |
|-------|---------|------------------------|---------------|------------|
| 1 | 1M | ~40GB+ (dual GPU ideal) | Free-Budget | Maximum intelligence + maximum context |
| 2 | 512K | ~28GB (dual, pressured) | Budget-Mid | Slight reduction, still strong |
| 3 | 262K | ~16GB (single GPU viable) | Mid-Premium | Swap to lighter model, cheaper API |
| 4 | 132K | ~8GB (survival) | Budget | Minimal context, minimal cost |

---

## Beefy Tier (≥24GB VRAM)

### Local VRAM Scaling Ladder (7 steps)

The ladder uses model **archetypes** — users plug in their own local models that match the archetype. The `serve recommend` command scans the catalog and picks the best match.

```
STEP  STATE            ACTION                              MAIN ARCHETYPE         AUX ARCHETYPE        CTX
───── ──────────────── ─────────────────────────────────── ────────────────────── ──────────────────── ────────
  1   Ideal            Nothing                             27-28B dense (Q4)      35B MoE (3B active)  1M
  2   Mild pressure    Offload aux experts to CPU          27-28B dense (Q4)      35B MoE (cpu-moe)    1M
  3   Moderate         Drop context of both               27-28B dense (Q4)      35B MoE (3B active)  512K
  4   High             Drop aux, set aux to API            27-28B dense (Q4)      API vision (free)    262K
  5   Critical         Swap main to small dense            27B hybrid/Mamba (Q4)  API vision (cheap)   262K
  6   Extreme          Swap main to MoE (3B active)       35B MoE (3B active)    API vision (cheap)   132K
  7   API-only         No local serving viable             API main (Nous)        API vision (Nous)    1M
```

### What each archetype means

| Archetype | Typical Size (Q4) | VRAM | Examples (user's catalog) | Generic examples |
|-----------|------------------|------|--------------------------|-----------------|
| 27-28B dense | 14-17 GB | ~22 GB with KV cache | Darwin 28B Reason, Qwopus 27B | Any Qwen3.6-27B fine-tune, Llama 3.3 27B |
| 27B hybrid/Mamba | 14 GB | ~16 GB | Prism Eagle 27B | Any Mamba2/GDN hybrid at 27B |
| 35B MoE (3B active) | 11-17 GB | ~11-17 GB | Carnice 35A3B, Darwin Apex | Any Qwen3.5/3.6-35B-A3B MoE |

### Step-by-step behavior

**Step 1: Ideal — Dual local at 1M**
Both models loaded, both at 1M context, full spec decoding. ≥8 GB free VRAM after loading both.
- main: 27-28B dense (e.g. Darwin 28B Reason, Q4_K_M, ~22 GB)
- aux: 35B MoE 3B-active (e.g. Carnice 35A3B, ~17 GB)
- Requires: dual GPU or 48GB+ VRAM

**Step 2: Mild pressure — Offload aux MoE experts to CPU**
Something else loaded onto a GPU. Move aux MoE expert weights to CPU RAM with `--cpu-moe`. Router + shared layers stay on GPU. Aux keeps serving, ~10 tok/s instead of ~30.
- main: unchanged
- aux: same MoE with `--cpu-moe-4` flag (drops to ~11 GB on GPU)

**Step 3: Moderate pressure — Drop context to 512K**
KV cache is the biggest variable. Shrink context on both:
- main ctx: 1M → 512K
- aux ctx: 1M → 512K
Both still serving with spec decoding.

**Step 4: High pressure — Drop local aux, API vision aux**
Kill the local aux. Aux routes to free API vision model:
- main: local dense @ 262K
- aux: Qwen 3.6 Plus (OpenRouter free, 1M ctx, vision) — 🟡 via OR
- Or: MiniMax M3 (NIM free, 1M ctx, vision) — ⚪ via NIM

**Step 5: Critical — Swap main to lighter dense/hybrid**
Main's VRAM footprint too large. Swap to a smaller model:
- main: 27B hybrid/Mamba (~14 GB, e.g. Prism Eagle) @ 262K
- aux: API vision (Qwen 3.5 Flash via Nous, $0.065/$0.26) — 🟢 NOUS+TG

**Step 6: Extreme — Swap to MoE main**
Swap to MoE (3B active per token, lower VRAM):
- main: 35B MoE 3B-active (e.g. Darwin Apex) @ 132K
- aux: API vision (Qwen 3.5 Flash via Nous) — 🟢 NOUS+TG

**Step 7: API-only fallback — No local serving viable**
GPU fully occupied or down. Full cloud:
- main: GLM 5.2 (Nous) or DeepSeek V4 Pro (NIM free)
- aux: Qwen 3.5 Flash (Nous) or MiniMax M3 (NIM free)
- See API pairing matrix for all options

### Beefy API Fallback Pairings (when local is unavailable)

| Context | Main | Aux | Gateway | Blended Cost |
|---------|------|-----|---------|-------------|
| 1M | GLM 5.2 (Nous) | Qwen 3.5 Flash (Nous) | 🟢 NOUS+TG | ~$0.44/$1.42 |
| 1M | GLM 5.2 (Nous) | Qwen 3.6 Plus (OR free) | 🟡 NOUS+OR | ~$0.38/$1.20 |
| 1M | DeepSeek V4 Pro (NIM) | MiniMax M3 (NIM) | ⚪ NIM | FREE |

---

## Modest Tier (8-24GB VRAM, single GPU)

Hardware: single RTX 3060 (12GB), RTX 4070 (12GB), RTX 3070 (8GB), RTX 4070 Ti (16GB), single RTX 3090 (24GB).

### Modest Scaling Ladder (5 steps)

```
STEP  STATE            ACTION                              MAIN                   AUX                  CTX     GATEWAY
───── ──────────────── ─────────────────────────────────── ────────────────────── ──────────────────── ──────── ────────
  1   Comfortable      API main + free vision aux           DeepSeek V4 Pro (Nous) Qwen 3.6 Plus (OR)  1M       🟡 NOUS+OR
  2   Budget           Both free via NIM                    DeepSeek V4 Flash (NIM) MiniMax M3 (NIM)    1M       ⚪ NIM
  3   Tight            Drop context, cheap pair            DeepSeek V4 Flash (Nous) Qwen 3.5 Flash     262K     🟢 NOUS+TG
  4   Minimal          132K, zero cost                     DeepSeek V4 Flash (NIM) MiniMax M3 (NIM)    132K     ⚪ NIM
  5   Local option     Small local aux (≤3GB) + API main   DeepSeek V4 Pro (Nous)  Local small model   1M       🟡 NOUS+local
```

### Step-by-step behavior

**Step 1: Comfortable — API main + free vision aux**
Best value 1M pair. Main through Nous (TG active), aux via OpenRouter free:
- main: DeepSeek V4 Pro ($0.435/$0.87 via DeepSeek API, or free via NIM)
- aux: Qwen 3.6 Plus (FREE on OpenRouter, 1M ctx, vision)
- Blended: ~$0.17/$0.35
- Gateway: 🟡 NOUS+OR (TG active on main, 10% OR bonus on aux credits)

**Step 2: Budget — Both free via NIM**
Zero cost, both through NVIDIA NIM:
- main: DeepSeek V4 Flash (free, 1M ctx, no vision)
- aux: MiniMax M3 (free, 1M ctx, vision ✅)
- Blended: $0.00
- Gateway: ⚪ NIM (no TG, separate `NVIDIA_API_KEY`)

**Step 3: Tight — Drop context to 262K**
Cheapest pair with full Tool Gateway at 262K:
- main: DeepSeek V4 Flash (Nous, $0.09/$0.18)
- aux: Qwen 3.5 Flash (Nous, $0.065/$0.26)
- Blended: ~$0.07/$0.21
- Gateway: 🟢 NOUS+TG

**Step 4: Minimal — 132K context, zero cost**
- main: DeepSeek V4 Flash (NIM free)
- aux: MiniMax M3 (NIM free)
- Blended: $0.00
- Gateway: ⚪ NIM

**Step 5: Local option — Small local aux alongside API main**
If the GPU has room for a tiny model (2-3 GB), run a small multimodal model locally as aux:
- main: DeepSeek V4 Pro (Nous, $0.435/$0.87)
- aux: Any small local vision model (e.g. Qwen2.5-Omni-3B at 2.5 GB, or any sub-3B multimodal GGUF)
- Blended: ~$0.17/$0.35 (only main costs)
- Gateway: 🟡 NOUS+local (TG active, aux is zero-cost local)

---

## Thin Tier (<8GB VRAM / no GPU)

Hardware: Integrated graphics, no NVIDIA GPU, or very low VRAM card.

### Thin Scaling Ladder (4 steps — pure API)

```
STEP  STATE            ACTION                              MAIN                   AUX                  CTX     GATEWAY
───── ──────────────── ─────────────────────────────────── ────────────────────── ──────────────────── ──────── ────────
  1   Zero cost        Both free via NIM                   DeepSeek V4 Flash (NIM) MiniMax M3 (NIM)    1M       ⚪ NIM
  2   Budget TG       Cheap pair through Nous             DeepSeek V4 Flash (Nous) Qwen 3.5 Flash     1M       🟢 NOUS+TG
  3   Mid tier         Better main, same aux              DeepSeek V4 Pro (Nous)  Qwen 3.5 Flash      1M       🟢 NOUS+TG
  4   Premium         Top reasoning + cheap vision aux    GLM 5.2 (Nous)         Qwen 3.5 Flash      1M       🟢 NOUS+TG
```

### Step-by-step behavior

**Step 1: Zero cost — Both free via NIM**
No GPU, no money. Both models free via NVIDIA NIM:
- main: DeepSeek V4 Flash (free, 1M ctx, no vision)
- aux: MiniMax M3 (free, 1M ctx, vision ✅)
- Blended: $0.00
- Gateway: ⚪ NIM
- Requires: `NVIDIA_API_KEY` from build.nvidia.com (free signup, ~1000 RPM)

**Step 2: Budget TG — Ultra-cheap pair with full Tool Gateway**
- main: DeepSeek V4 Flash (Nous, $0.09/$0.18)
- aux: Qwen 3.5 Flash (Nous, $0.065/$0.26)
- Blended: ~$0.07/$0.21
- Gateway: 🟢 NOUS+TG (Firecrawl, FAL, OpenAI TTS, Browser Use all active)

**Step 3: Mid tier — Better main**
- main: DeepSeek V4 Pro (Nous, $0.435/$0.87)
- aux: Qwen 3.5 Flash (Nous, $0.065/$0.26)
- Blended: ~$0.20/$0.47
- Gateway: 🟢 NOUS+TG

**Step 4: Premium — Top reasoning**
- main: GLM 5.2 (Nous, $0.95/$3.00)
- aux: Qwen 3.5 Flash (Nous, $0.065/$0.26)
- Blended: ~$0.44/$1.42
- Gateway: 🟢 NOUS+TG

---

## Cross-Tier Escalation

When a user's situation changes, the ladder transitions smoothly:

| Transition | Trigger | Action |
|-----------|---------|--------|
| Beefy → Modest | GPU failure / VRAM loss | Skip to Beefy Step 7 (API fallback) |
| Modest → Thin | GPU removed / VRAM < 8GB | Drop to Thin Step 1 (NIM free) |
| Thin → Modest | New GPU installed (8-24GB) | Jump to Modest Step 1 |
| Modest → Beefy | Second GPU / VRAM ≥ 24GB | Jump to Beefy Step 1 (local dual) |
| Any → API fallback | Network outage on local | Use `serve auto main --api` |

## How to use this ladder

### For any user (regardless of hardware):

```bash
# 1. Let turbofit detect your hardware and pick the best setup
serve auto main

# 2. If you want to see all options for your tier
serve recommend

# 3. If VRAM is tight, walk down the ladder
serve downscale

# 4. Force API-only mode (no local GPU needed)
serve auto main --api

# 5. Force free-only endpoints (zero cost)
serve auto main --free

# 6. Browse all available models
serve catalog
```

### Registering your own local models

Users add their own local models to the catalog:

```bash
# Register a model GGUF you've downloaded
serve register my-model /path/to/model.gguf --port 11500

# Set its metadata (tier, vision, role, etc.)
# Edit ~/.config/turbofit/models.yaml to add:
#   tier: s          # s | sf | sd | f | c
#   vision: true     # if it has an mmproj
#   role: main       # main | aux | either
#   size_gb: 16.0    # disk footprint
```

The `serve auto` picker then scans your catalog and picks the best model for each archetype slot based on tier, ctx, and available VRAM.

## Why this ladder specifically

### 1M ctx is the goal, not 64K
Hermes-Agent hard floor is 64K, but the goal is 1M. The ladder only drops context when absolutely necessary.

### Main is always protected (Beefy)
In the Beefy ladder, main stays at the top archetype until Step 5. Steps 1-4 only touch aux.

### API tier philosophy: free first, then cheapest
The Modest and Thin ladders lead with free NIM endpoints, then cheapest Nous pairings. Premium is optional.

### Tool Gateway priority
Pairings that route main through Nous get the Tool Gateway (Firecrawl, FAL, OpenAI TTS, Browser Use). The 🟢 NOUS+TG indicator marks pairings where TG is fully active.

### Don't kill the user's work
The ladder never kills a model mid-response. It only adapts between requests.

## Planned additions (not yet shipped)

| Feature | Why |
|---|---|
| `serve downscale --api` | Force API-mode downscale (skip local VRAM probe) |
| `serve pair <main> <aux>` | One-shot pairing command — sets both main+aux in one call |
| `serve pair list` | Show all pairings from the matrix, filtered by tier/ctx/cost |
| `serve pair recommend` | Auto-pick best pairing for current hardware + budget |
| Push mode (cron) | `*/5 * * * *` cron that runs `serve downscale` when free_GB drops |
| Per-GPU ladder | If GPU0 is free but GPU1 is full, keep main on GPU0, drop aux on GPU1 |
| Auto catalog discovery | Scan common GGUF directories and auto-register models |

## See also

- `../SKILL.md` — full turbofit overview
- `api-pairing-matrix.md` — complete pairing matrix with all combinations
- `curated-lineup.md` — the curated picks (local archetypes + API models)
- `api-model-rankings.md` — individual model pricing and rankings
