# Scaling ladder — polite VRAM adaptation

The scaling ladder is what makes turbofit opinionated. Its philosophy: **good intelligence is always available no matter what the user loads onto their GPUs**. It stays out of the way of whatever the user wants to do alongside their AI.

Two modes:
1. **Pull mode:** user invokes `serve downscale` → probes VRAM → adapts.
2. **Push mode (planned):** cron watches VRAM → triggers `serve downscale` automatically.

## The 7-step polite VRAM scaling ladder

```
STEP  STATE            ACTION                              MAIN               AUX                CTX
───── ──────────────── ─────────────────────────────────── ────────────────── ────────────────── ────────
  1   Ideal            Nothing                             Darwin 28B         Carnice 35A3B      1M
  2   Mild pressure    Offload aux experts to CPU          Darwin 28B         Carnice (CPU-moe)  1M
  3   Moderate         Drop context of both                Darwin 28B         Carnice 35A3B      128K
  4   High             Drop aux, set aux to auto           Darwin 28B         auto (main)        128K
  5   Critical         Swap main to small dense            Prism Eagle 27B    auto (main)        128K
  6   Extreme          Swap main to MoE (3B active)        Darwin Apex 35A3B  auto (main)        64K
  7   Survival         Offload experts of main             Darwin Apex (CPU-moe) auto (main)     64K
```

## Step-by-step behavior

### Step 1: Ideal — Darwin 28B @ 1M + Carnice 35A3B @ 1M

Both models loaded, both at 1M context, full spec decoding. The user has ≥8 GB free VRAM after loading both. This is the flagship state:
- **main:** Darwin 28B Reason (Q4_K_M, turbo4 KV, 262K ctx, ~22 GB)
- **aux:** Carnice Apex 35A3B (MoE, turbo4 KV, 1M ctx, ~17 GB)

### Step 2: Mild pressure — Offload Carnice experts to CPU

User loaded something onto the GPU (training, another app). Free VRAM < 8 GB but main is intact. **Don't touch main.** Move Carnice's MoE expert weights to CPU RAM with `--cpu-moe`. Only router + shared layers stay on GPU. Carnice keeps serving, just slower (~10 tok/s instead of ~30 tok/s from PCIe-bound experts).

```bash
serve stop carnice-apex-35a3b-compact
serve carnice-apex-35a3b-compact   # with --cpu-moe flag
```

### Step 3: Moderate pressure — Drop context of both

User loaded even more stuff. Free VRAM < 4 GB. KV cache is the biggest variable, so shrink context on both models:
- main ctx: 1M → 128K (saves ~15 GB KV on Darwin 28B)
- aux ctx: 1M → 128K (saves ~12 GB KV on Carnice)

Both models keep serving at reduced context. Spec decoding still active.

```bash
serve stop darwin-28b-reason
cat <<'EOF' > /tmp/restart-main-128k.yaml
ctx: 131072
EOF
serve darwin-28b-reason   # uses new ctx from curated.yaml override

serve stop carnice-apex-35a3b-compact
serve carnice-apex-35a3b-compact   # uses ctx=131072
```

### Step 4: High pressure — Drop Carnice, aux set to auto

Free VRAM < 2 GB even at 128K. **Kill the aux slot entirely.** Carnice stops completely. Aux role is rewired to use the main model:

```bash
serve stop carnice-apex-35a3b-compact
serve auto aux   # picks aux_api (free DeepSeek V4 Flash or MiniMax M3)
```

Aux becomes a cloud API call. Main still serves at 128K.

### Step 5: Critical — Swap to Prism Eagle 27B

Free VRAM < 1 GB or user explicitly requested smaller. Darwin Reason is 22 GB — too much. Swap to the smallest dense model in the fleet:

- **main:** Prism Eagle 27B (only 14 GB, draft-mtp for speed)

```bash
serve stop darwin-28b-reason
serve auto main   # picks Prism Eagle from s.tier.speed_pick
```

Ctx stays at 128K. Spec decoding (draft-mtp) stays active.

### Step 6: Extreme — Swap to Darwin Apex 36B Opus APEX MoE

User is still filling the GPU. Prism Eagle isn't enough. Swap to the only MoE recommended for main:

- **main:** Darwin Apex 36B Opus APEX (MoE, 3B active per token, ~17 GB)

```bash
serve stop prism-eagle-27b
serve darwin-apex-36b-i-compact   # from sf.tier
```

Ctx drops to 64K (minimum for Hermes to initialize). NextN spec decoding active. MoE experts are the natural pressure valve.

### Step 7: Survival — Offload experts of Darwin Apex

Last resort before Hermes shuts down entirely. MoE expert weights move to CPU:

- **main:** Darwin Apex 35A3B with `--cpu-moe` flag
- **ctx:** 64K
- **speed:** ~5 tok/s (PCIe-bound) but still responding

```bash
serve stop darwin-apex-36b-i-compact
serve darwin-apex-36b-cpu-moe   # needs explicit --cpu-moe catalog entry
```

Good intelligence survives even at this level.

## Why this ladder specifically

### 1M ctx floor is the goal, not 64K

Hermes-Agent hard floor is 64K, but the goal is 1M. The ladder only drops context when absolutely necessary. Each step tries to preserve context before dropping it.

### Main is always protected until Step 5

Steps 1-4 only touch aux. Main stays at Darwin 28B until VRAM truly forces a swap. This is "good intelligence is always available."

### MoE for headroom, not size

Darwin Apex (35A3B MoE) is only 3B-active per token. It's smaller than a dense 27B under load but has more parameters available. **It's the only MoE recommended for main** because:
- MoE expert offload is a free tier-downgrade
- Spec decoding (NextN) gives ~107 tok/s on par with dense 27B
- The ladder prefers MoE for the auto-pick when VRAM is borderline

### Don't kill the user's work

The ladder never kills a model mid-response. It only adapts between requests. If the user is generating, the ladder waits.

### Aux is always Carnice — except when the user loads onto their GPU

Carnice 35A3B is the aux in all cases. The ladder only drops Carnice when the user explicitly loads something onto the GPU (training, inference, rendering). This is the "polite" part — your AI stays out of the way of whatever you're doing.

## Planned additions (not yet shipped)

| Feature | Why |
|---|---|
| `--cpu-moe` flag support | MoE expert offload needs explicit catalog entries |
| Context override CLI | `serve set-ctx <alias> <ctx>` for quick context changes without re-launching |
| Quant downgrade (Q4 → Q3) | Catalog already supports different quants per alias; ladder needs to pick |
| Per-GPU ladder | If GPU0 is free but GPU1 is full, keep main on GPU0, drop aux on GPU1 |
| Push mode (cron) | `*/5 * * * *` cron that runs `serve downscale` when free_GB drops |
| Benchmark ranking | Head-to-head scoring for Qwopus vs Qwable vs Carwin vs Darwin (all at Q4) |

## See also

- `../SKILL.md` — full turbofit overview
- `curated-lineup.md` — the curated picks this ladder operates on
