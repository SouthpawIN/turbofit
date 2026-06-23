---
name: turbofit
description: "Opinionated unified local LLM backend (turbofit v5). Picks the best main + aux model for your hardware, launches them detached, wires Hermes-Agent config, and adapts to live VRAM pressure via a scaling ladder. Replaces llama-launch, omni-va, and ad-hoc llama-server scripts. Catalog schema supports per-model binary pinning (atomic fork vs stock), named flag presets (nextn, draft-mtp, turbo4-kv, vision-mmproj), tier ladder (s/sf/sd/f/c), and the 64K Hermes context floor is enforced everywhere. End-user UX is `serve auto main` and the user is done."
version: 5.0.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [llama.cpp, ollama, vllm, sglang, llmfit, gguf, hermes-agent, nvidia-nim, catalog, scaling-ladder, opinionated, unified-backend]
metadata:
  hermes:
    tags: [llama.cpp, ollama, vllm, sglang, llmfit, gguf, hermes-agent, nvidia-nim, catalog, scaling-ladder, opinionated, unified-backend]
    related_skills: [southpaw-models, local-llm-fleet-management, llama-cpp, gguf-quantization, omni-va-local-server, evolutionary-radio]
  changelog: |
    5.0.0 (2026-06-22): Opinionated unified backend.
      - 18 named flag presets: nextn/nextn-tight, draft-mtp/draft-mtp-tight, turbo4/3/2-kv, q8/q4-kv, cpu-moe-2/4/8, no-mmap, split-none, mlock, parallel-4/2/1.
      - Per-model binary pin: atomic fork for TurboQuant+NextN models, stock for legacy.
      - New commands: serve vram, serve auto [main|aux], serve downscale, serve stop-all, serve fetch, serve bench, serve recommend.
      - serve recommend ranks catalog by ctx≥64K, tok/s≥25, vision bonus, tier priority — all hardware-neutral (tok/s measured locally, varies by GPU).
      - All local models vision-enabled via shared mmproj-F32.gguf symlink.
      - serve herm fixed: launches herm TUI only (herm handles hermes internally — no double-launch).
      - NIM API ranking: DeepSeek V4 Pro → GLM 5.1 → DS V4 Flash → MiniMax M3 → Nemotron Ultra.
      - Default aux: Step 3.7 Flash (free, vision). Fallback: MiniMax M3 (free, vision, 1M ctx) → Qwen 3.5 Flash 02-23 (paid, vision, 1M ctx).
      - Multi-GPU tensor split via catalog extra_args (no hardcoded GPU values — hardware-neutral).
      - Migrated 23 curated picks from llama-launch catalog; 13 dead launchers consolidated.
      - Replaces: llama-launch, omni-va, model-server.sh, start-{qwen,glm,...}-server.sh.
    4.0.0: Initial multi-launcher (llama.cpp / Ollama / vLLM / SGlang), NVIDIA NIM API list.
---

# Turbofit v5 — opinionated unified backend

End-user UX is `serve auto main` — picks the best main model for your system state, launches it detached, and wires Hermes config. The user is done.

For the scaling/adaptation half of the opinionated story, see [`references/scaling-ladder.md`](references/scaling-ladder.md).

## When to use

Trigger phrases: "set up my local LLM", "launch a model", "what model should I run", "my GPU is busy, scale down", "I need a coder model", "I need vision", "turbofit auto", "swap models", "swap main", "swap aux", "which model fits my box", "stop everything".

## Quick start

```bash
# Source the shim (one-time, already in ~/.bashrc)
source ~/.hermes/skills/turbofit/scripts/turbofit.sharco

# One-shot: pick best main for your box, launch, wire Hermes
serve auto main                    # opinionated: tier ladder, ≥25 tok/s, 64K ctx
serve auto main --vision           # require vision
AUTO_CTX=131072 serve auto main    # raise ctx target

# Stop everything
serve stop-all

# Browse the catalog (featured first, tier-ordered)
serve catalog

# Launch a specific model
serve darwin-28b-reason
serve string darwin-apex-36b-i-compact   # print launch string, don't launch

# Register a new model
name mymodel /path/to/file.gguf --port 9090

# Wire to Hermes (main / aux)
serve main mymodel --ui tui
serve aux mymodel --ui tui

# VRAM probe (live)
serve vram

# Adapt to current VRAM pressure
serve downscale
```

## Opinionated defaults (turbofit v5)

| Setting | Value | Source |
|---|---|---|
| ctx floor | **65536** tokens | Hermes-Agent hard requirement |
| tok/s floor (main) | **25 tok/s** | Spec decoding assumption |
| prefer vision on main | **yes** | Use `vision:` field in catalog |
| tier ladder | `s → sf → sd → f → c` | `serve auto` |
| scale down triggers | free VRAM < 14GB → < 8GB → < 4GB | `serve downscale` |
| per-model binary pin | atomic (TurboQuant+NextN) or stock | `binary:` field |

**End-user UX:** `serve auto main` → done. If VRAM pressure hits, `serve downscale` adapts.

## Catalog schema (v5 extended)

```yaml
models:
  <alias>:
    # Required
    launcher: llama-cpp           # llama-cpp | ollama | vllm | sglang
    path: /abs/path/to.gguf        # OR HF repo for vllm/sglang
    port: 11500                    # auto-assigned if absent

    # Recommended
    ctx: 262144                    # 64K floor enforced
    gpu: 0                         # 0 | 1 (single-GPU target)
    gpu: 0                         # 0 | 1 (single-GPU target)
    mmproj: /path/to/mmproj.gguf   # vision projector
    presets: [nextn, turbo4-kv, no-mmap]   # see below
    extra_args: [--draft-block-size 3]           # raw flag list
    aliases: [short, alt]          # alternative names resolved by serve的政策
    description: "..."             # shown in `serve catalog`
    tags: [qwen, mtp, featured]

    # Opinionated metadata (used by `serve auto` and Garage UI)
    tier: s                        # s | sf | sd | f | c
    featured: true                 # Garage top row
    tok_s_target: 107              # measured throughput
    vision: true                   # has vision tower + mmproj
    size_gb: 16.0                  # disk footprint
    hf_repo: org/repo              # for `serve fetch <alias>` (future)
    role: main                     # main | aux | either
```

### Named flag presets

Apply by listing in `presets:`. Multiple presets merge; later presets override earlier flags.

| Preset | Expands to |
|---|---|
| `nextn` | `--spec-type nextn --draft-block-size 3` |
| `nextn-tight` | `--spec-type nextn --draft-block-size 2` |
| `draft-mtp` | `--spec-type draft-mtp` |
| `draft-mtp-tight` | `--spec-type draft-mtp --draft-block-size 2` |
| `turbo4-kv` | `-ctk turbo4 -ctv turbo4` |
| `turbo3-kv` | `-ctk turbo3 -ctv turbo3` |
| `turbo2-kv` | `-ctk turbo2 -ctv turbo2` |
| `q8-kv` | `-ctk q8_0 -ctv q8_0` |
| `q4-kv` | `-ctk q4_0 -ctv q4_0` |
| `no-mmap` | `--no-mmap` |
| `split-none` | `--split-mode none` |
| `mlock` | `--mlock` |
| `cpu-moe-2` / `cpu-moe-4` / `cpu-moe-8` | `--n-cpu-moe N` (MoE expert offload) |
| `parallel-4` / `parallel-2` / `parallel-1` | `--parallel N` |

Multi-GPU tensor split: use `extra_args: ['--tensor-split', 'X,Y']` in catalog — no hardcoded preset.

### Tier ladder (used by `serve auto`)

| Tier | Meaning | Examples |
|---|---|---|
| `s` | smartest | Darwin Reason, Darwin Apex-Compact, Prism Eagle |
| `sf` | smart + fast | Carwin-MTP, Qwopus v2-MTP, Qwopus Coder-MTP |
| `sd` | smart + dense | Carnice Apex Compact |
| `f` | fast | Qwable MTP, Qwopus abliterated-MTP |
| `c` | cheap | Qwen legacy, devstral, step-flash, omni-3b |

## Commands

```bash
# Install / update
serve install                            # llama.cpp from source (atomic fork if path set)
serve install <launcher>                 # one launcher: llama-cpp, ollama, vllm, sglang
serve update                             # update llama.cpp
serve update <launcher|all>              # specific launcher or all
serve check                              # version status

# Hardware / VRAM
serve fit <model> [ctx]                  # llmfit fit check (default ctx=65536)
serve vram                               # live GPU VRAM probe (JSON)
serve recommend                          # scan catalog, rank by fit (ctx≥64K, tok/s≥25, Q4, vision)

# Catalog
serve register <alias> <path>            # register model
           [--launcher llama-cpp|ollama|vllm|sglang] [--port N]
serve catalog                            # show registered (featured first, tier-ordered)

# Launch
serve <alias>                            # launch detached, shows backend/port/logs
serve string <alias>                     # print launch string, don't launch
serve stop <alias>                       # stop a running server
serve stop-all                           # stop everything
serve list                               # list running + detect rogue llama-servers

# Fetch / benchmark
serve fetch <alias>                      # download missing model from HF (uses hf_repo)
serve bench <alias>                      # lm-eval-harness benchmark (launches if needed)

# Opinionated auto (turbofit v5)
serve auto main [--vision] [--ui ...]    # pick best main, launch, wire Hermes
serve auto aux [--ui ...]                # pick best aux, launch, wire Hermes
serve downscale                          # adapt to current VRAM pressure
AUTO_CTX=131072 serve auto main          # raise ctx target

# Hermes routing
serve main <alias> [--ui tui|dashboard|gateway|desktop|herm]
serve aux  <alias> [--ui ...]
serve herm <alias>                       # launch + main + herm TUI (herm handles hermes internally)
serve herm aux <alias>

# NVIDIA NIM API (curated)
serve api list
serve api use <rank|api_id> [main|aux]
```

## How the auto-picker decides

1. **Filter** catalog by `role` (main | aux) and (if `--vision`) `vision: true`.
2. **Filter** by `ctx >= target` (default 65536).
3. **Sort** by `(tier_rank, featured, -tok_s_target)` — best tier wins, then featured, then speed.
4. **Skip** entries whose `path:` doesn't exist (catalog has them but disk doesn't).
5. **Launch** via the per-model `binary:` if set, else stock `llama-server`.

The auto-picker does NOT yet weight by current VRAM headroom. Use `serve downscale` to adapt after the fact, or run `serve vram` first to know.

## How the scaling ladder works

Triggered by `serve downscale` (or invoked automatically by future cron):

```
free_VRAM > 14GB   → no change
free_VRAM 8-14GB   → stop aux (keep main at full ctx)
free_VRAM 4-8GB    → stop aux, shrink main ctx to 64K
free_VRAM < 4GB    → stop all, swap main to c-tier (cheap) + CPU offload
```

The ladder is intentionally simple today. Planned additions:
- MoE expert offload (`--cpu-moe`) for tier-swap step
- Quant downgrade (Q4 → Q3) for tier-swap step
- Auto-swap main from s-tier dense → sf-tier MoE (more headroom)
- Wake-on-ping proxy mode for aux

## Pitfalls

### 7. HF download: use exact filenames, never wildcards

`hf download <repo> --include="*.gguf"` downloads **every quant in the repo** (BF16, F16, Q8_0, Q2_K, Q3_K_S, ...). For Qwen/Coder/Darwin repos this can be 100-200 GB of junk. Only Q4_K_M + mmproj are needed.

**Correct pattern:**

```bash
hf download <repo> --include="*Q4_K_M*" --include="mmproj-F32.gguf" --local-dir <dest>
```

Even safer: download only the one specific file you need:

```bash
hf download <repo> --include="<exact-filename>.gguf" --local-dir <dest>
```

### 8. `serve` in PATH is Ray's CLI (v5)
Ray ships a `serve` Python script at `~/.local/bin/serve`. The turbofit bash shim (in `turbofit.sharco`) overrides it as a function. If you bypass the shim, you'll hit Ray. Use `tf` as a future alias if you want a separate namespace.

### 2. Per-model `binary:` is REQUIRED for TurboQuant + NextN models
Stock `llama-server` doesn't support `-ctk turbo4` or `--spec-type nextn`. If you remove the `binary:` field from `darwin-28b-reason` or `qwopus-27b-coder-mtp`, the launch will fail with "Unsupported cache type" or "unknown speculative type".

The atomic fork lives at `~/projects/LLM-Infra/llama.cpp-atomic/build/bin/llama-server`.

### 3. Context floor is enforced everywhere
Every launch string uses `ctx >= 65536`. Hermes-Agent crashes on first multi-turn message if ctx < 64K.

### 4. `flash_attn` is tri-state
`-fa on` / `-fa off` / `-fa auto`. Bare `--flash-attn` errors out.

### 5. `--gpu` is not a valid llama.cpp flag
Use `-ngl N` to control GPU layers, or `--device CUDA0` + `--main-gpu 0` for device selection. The catalog's `gpu:` field maps to `--main-gpu N` automatically.

### 6. NVIDIA NIM has TWO tiers — free and paid serverless
The 5 models listed all have **free endpoints** at `https://integrate.api.nvidia.com/v1` — covered by your `NVIDIA_API_KEY` from a free `build.nvidia.com` signup (~1000 RPM, no credit card).

NVIDIA also sells a paid serverless tier with the same model IDs (e.g. $1.30/$2.60 for DeepSeek V4 Pro) under the same base URL. The same API key works for both, but the free tier has tighter rate limits.

### 7. The 12 Omni*/Senter* training artifacts are excluded
These are user-made Darwin-merged models used for training research, not fleet picks. They live in `~/.config/llama-launch/models.yaml` (legacy) but not in the turbofit catalog. Add them back with `name <alias> <path>` if needed.

## Extending turbofit

### Adding a new model

```bash
name mynew /path/to/file.gguf --port 11530
# then edit ~/.config/turbofit/models.yaml to add tier, presets, gpu, etc.
serve mynew
```

### Adding a new flag preset

Edit `~/.hermes/skills/turbofit/scripts/serve`, find `declare -A PRESET_FLAGS=(`, add a line like:

```bash
[my-preset]='--my-flag value'
```

### Adding a new tier

Edit `scale_pick()` and `list_catalog()` in the same script.

### Adding a new launcher

Extend the four `declare -A LAUNCHER_*` arrays at the top, add cases in `install_launcher()`, `update_all_launchers()`, `generate_string()`, and `launch_server()`.

## See also

- `references/scaling-ladder.md` — full 7-step polite VRAM scaling ladder
- `references/curated-lineup.md` — the hand-picked cascade with per-model rationale
- `southpaw-models` — curated lineup rationale (Darwin / Prism / Carnice / Carwin / Qwopus)
- `local-llm-fleet-management` — legacy catalog mechanics (now subsumed by turbofit)
- `llama-cpp` — llama.cpp build + GGUF discovery
- `omni-va-local-server` — wake-on-ping proxy (kept for Carnice slot)
- `evolutionary-radio` — where Darwin-merged checkpoints run

## Design rules (hardware-neutral, learned this session)

1. **No multi-GPU hardcoding.** Tensor split is user-configurable via catalog `extra_args: ['--tensor-split', 'X,Y']`. No preset assumes dual-GPU or specific VRAM amounts.
2. **tok/s varies by GPU.** Catalog `tok_s_target` values are measured locally — mark them as approximate. Use relative ranking (tier priority > vision bonus > speed bonus) rather than absolute claims.
3. **All models get vision.** Every local Qwen-family model can use the same `mmproj-F32.gguf` via symlink. Set `vision: true` and `mmproj:` in every catalog entry. One mmproj file serves the entire fleet.
4. **Model picks ARE the cascade.** The `curated.yaml` slot-ordered picks are Chris's hand-chosen scale-down sequence — not generic examples. When building recommendations, preserve the actual model aliases and their tier assignments.