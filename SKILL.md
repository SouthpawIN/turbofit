---
name: turbofit
description: "Opinionated unified LLM backend (turbofit v5.1). Picks the best main + aux model for your hardware — local or API — launches them detached, wires Hermes-Agent config, and adapts to live VRAM pressure via a scaling ladder. Three hardware tiers: Beefy (local+local), Modest (API+local), Thin (API+API). `serve auto main` auto-detects GPU and suggests the right setup. API fallback is always available (free: DeepSeek V4 Pro + Kimi K2.6). Replaces llama-launch, omni-va, and ad-hoc llama-server scripts. Catalog schema supports per-model binary pinning (atomic fork vs stock), named flag presets (nextn, draft-mtp, turbo4-kv, vision-mmproj), tier ladder (s/sf/sd/f/c), and the 64K Hermes context floor is enforced everywhere. End-user UX is `serve auto main` and the user is done."
version: 1.1.0
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
serve auto main                    # opinionated: auto-detects hardware, picks local or API
serve auto main --vision           # require vision
serve auto main --api              # force API mode (no local GPU needed)
serve auto main --free             # only free API endpoints
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

# Opinionated auto (turbofit v5.1 — hardware-aware, API-aware)
serve auto main [--vision] [--api] [--free] [--ui ...]    # pick best main (auto-detects hardware, picks local or API)
serve auto aux [--vision] [--api] [--free] [--ui ...]     # pick best aux
serve downscale                          # adapt to current VRAM pressure
AUTO_CTX=131072 serve auto main          # override ctx target

# Hardware-aware auto-detection:
#   ≥24GB VRAM → local main + local aux
#   8-24GB VRAM → API main + local aux (or local main + API aux)
#   <8GB / no GPU → API main + API aux (free: DeepSeek V4 Pro + Kimi K2.6)
# Use --api to force API mode, --free to restrict to free endpoints

# Hermes routing
serve main <alias> [--ui tui|dashboard|gateway|desktop|herm]
serve aux  <alias> [--ui ...]
serve herm <alias>                       # launch + main + herm TUI (herm handles hermes internally)
serve herm aux <alias>
serve herm                               # auto-pick main + launch herm TUI

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

## How the scaling ladder works (v5.1 — 7 steps + 3 hardware tiers)

The ladder now covers three hardware profiles, not just beefy GPU owners:

### Hardware Tiers (auto-detected by `serve auto`)

| Tier | VRAM | Recommendation | Default Main | Default Aux |
|------|------|----------------|-------------|-------------|
| **Beefy** | ≥24GB | Local main + local aux | Darwin Reason 28B | Carnice 35A3B |
| **Modest** | 8-24GB | API main + local aux (or local main + API aux) | DeepSeek V4 Pro (API) | Carnice (cpu-moe) |
| **Thin** | <8GB or no GPU | API main + API aux | DeepSeek V4 Pro (free, NIM) | Kimi K2.6 (free) |

`serve auto` detects which tier you're in by probing `nvidia-smi`. If no NVIDIA GPU is found, it defaults to **Thin** (API-only). Use `--api` to force API mode regardless.

### Beefy-tier scaling ladder (7 steps, triggered by `serve downscale`)

Full ladder defined in `curated.yaml` under `scaling_ladder.steps`:

```
Step  1: Ideal          — Darwin Reason 28B @ 1M + Carnice 35A3B @ 1M aux (dual GPU)
Step  2: Mild pressure  — Offload MoE experts from Carnice to CPU (--cpu-moe-4), ~27 tok/s
Step  3: Moderate       — Drop both models' context to 128K
Step  4: High pressure  — Drop Carnice entirely, route aux to API (Kimi K2.6 free)
Step  5: Swap dense     — Swap main to Prism Eagle 27B (14GB, 121 tok/s, Mamba2+GDN hybrid)
Step  6: Apex MoE       — Swap to Darwin Apex 35A3B MoE + CPU-offload experts (--cpu-moe-4)
Step  7: API-only       — No local serving viable. API main + API aux. Zero cost with free endpoints.
```

Each step preserves maximum intelligence while respecting VRAM. The cascade is the **hand-picked sequence for this system** — not a generic "shrink everything" rule. Never auto-skip steps based on free VRAM alone; always present the user with the ladder and let them choose, or use `serve downscale` which walks it conservatively.

### API model rankings (by volume performance)

**Main API (ranked by reasoning/coding quality, free first):**

| Tier | Model | Vision | Cost | Why |
|------|-------|--------|------|-----|
| S | DeepSeek V4 Pro | No | FREE (NIM) | Best open reasoning + coding, 1M ctx |
| S | Kimi K2.6 | Yes | FREE (OpenRouter) | Strong coding, vision, 1M ctx |
| S | GLM 5.1 | No | Paid | Agentic + long-horizon |
| S | Qwen 3.7 Max | No | Paid | Qwen frontier, 1M ctx |
| SF | DeepSeek V4 Flash | No | FREE (NIM) | Fast reasoning, best value |
| SF | MiniMax M3 | Yes | FREE (NIM) | Vision + 1M ctx |
| SF | Nemotron Ultra | No | FREE (NIM) | 550B MoE, reasoning budget |
| F | DeepSeek V4 Flash | No | $0.10/$0.20 | Cheapest paid main |
| F | Qwen 3.5 Flash | Yes | $0.065/$0.26 | Cheapest vision main |

**Aux API (ranked by vision > speed > cost, free first):**

| Tier | Model | Vision | Cost | Why |
|------|-------|--------|------|-----|
| SD | Kimi K2.6 | Yes | FREE | Best free aux — vision + reasoning |
| SD | MiniMax M3 | Yes | FREE (NIM) | Vision + 1M ctx |
| SD | Step 3.7 Flash | Yes | FREE | Fast, vision |
| SD | DeepSeek V4 Flash | No | FREE (NIM) | Fast reasoning (no vision — pair with vision main) |
| F | Qwen 3.5 Flash | Yes | $0.065/$0.26 | Cheapest paid vision aux |

## Pitfalls

- **`serve list` hangs on wake-on-ping proxy ports.** The rogue-port scanner iterates ALL listening ports and curls each one. When it hits a turbofit daemon's proxy port, the proxy tries to wake the backend (which takes 30+ seconds to load), causing a hang. Fixed by adding `--max-time 2` to all curl calls in the port scanner.
- **`serve auto main` double-launches on occupied ports.** If a systemd daemon is already running for the picked model, `serve_main` still calls `launch_server`, which tries to bind the same port and hangs in the health-check loop. Fixed by checking `systemctl --user is-active turbofit-${alias}.service` and PID files before launching — if already running, skip launch and just wire Hermes config.
- **`serve herm` crashes with `set -euo pipefail` + shift.** The case statement had `shift; serve_herm "$@"` but after the command parser already shifted past the command name, the extra shift fails on empty args and `set -e` causes silent exit. Fixed by removing the redundant shift.
- **`start_ui` doesn't recognize `herm_main`/`herm_aux` UI values.** `serve_herm` sets `UI="herm_main"` or `UI="herm_aux"` but `start_ui` only matched the case `herm)`. The unrecognized case falls through to "Unknown UI" error. Fixed by matching `herm|herm_main|herm_aux)`.
- **`--main-gpu` must be 0 when `CUDA_VISIBLE_DEVICES` is set.** The llama-proxy sets `CUDA_VISIBLE_DEVICES=<gpu_id>` which makes only one GPU visible to llama-server. But the catalog's `--main-gpu N` flag uses the physical GPU index. When `CUDA_VISIBLE_DEVICES=1` is set, `--main-gpu 1` is invalid (only device 0 exists). Fixed in llama-proxy by rewriting `--main-gpu` to 0 in the extra args before spawning.
- **`LLAMA_LIB_DIR` must be set in the systemd service, not just `LD_LIBRARY_PATH`.** The llama-proxy reads `LLAMA_LIB_DIR` (not `LD_LIBRARY_PATH`) to set the library path for the spawned backend process. If only `LD_LIBRARY_PATH` is set, the proxy uses its default (stock llama.cpp path) and the atomic fork's symbols aren't found.
- **`--cpu-moe-draft` and `--mmproj-draft` are not supported by the current atomic fork.** The `generate_string` function was adding these automatically for MoE models with spec decoding. Removed — the draft model shares the same GGUF, so `--cpu-moe` applies to both automatically.
- **Carnice mmproj-BF16.gguf crashes the atomic fork's clip loader.** The 35B-A3B mmproj file causes `ggml_backend_buffer_set_usage` abort in `clip_model_loader::load_tensors`. Vision disabled for Carnice until the atomic fork is fixed or stock llama.cpp is used.

- **Systemd services override turbofit.** If legacy `llama-*.service` or `omni-va.service` units are running, they will hold ports and restart killed processes, silently replacing turbofit-managed launches. Before `serve <alias>`, run `systemctl --user list-units | grep llama` and stop+disable any conflicting units on the target port.
- **YAML duplicate keys silently overwrite.** When editing `models.yaml`, never paste a new entry that shares keys with the one above it (e.g. two `role:` lines, two `mmproj:` lines). YAML last-key-wins means the earlier value is silently discarded. Always verify with `grep` after editing.
- **mmproj must match the text model's `n_embd`.** A 27B-dense mmproj (`n_embd=5120`) will NOT work with a 35A3B MoE (`n_embd=2048`). The error is `mismatch between text model (n_embd = X) and mmproj (n_embd = Y)`. For Qwen3.6-35B-A3B MoE models, use the mmproj from `unsloth/Qwen3.6-35B-A3B-GGUF` (mmproj-BF16.gguf, ~889MB). For 27B dense models, use the per-model F32 mmproj.
- **Symlink mmproj files can be stale.** Several model directories had `mmproj-F32.gguf` symlinked to a different model's mmproj (e.g. Carnice → Qwopus 27B). Always verify with `ls -la` and `readlink -f`. If the target is wrong, delete the symlink and download the correct file from the matching HF repo.
- **`serve <alias>` may report success but launch the wrong model** if a stale systemd service or leftover process holds the port. Always verify with `ps aux | grep <port>` that the actual command line matches the catalog entry.
- **The `serve` script reads the catalog fresh each time.** If you edit `models.yaml` while a serve command is running, the next invocation will pick up the changes. But the *currently running* process won't be affected — you must `serve stop <alias>` and re-launch.
- **GitHub repo creation requires explicit user permission.** Never create new repos or publish anything publicly without the user explicitly asking. Existing repos can be updated when the user directs work on them.

## mmproj Reference

Vision-capable models need the correct mmproj file. Architecture mismatch (`n_embd` difference) causes a hard crash at load time.

| Model family | n_embd | mmproj source | File |
|---|---|---|---|
| Qwen3.6-27B dense (Darwin Reason, Prism Eagle, Carwin, Qwopus, Qwable) | 5120 | Per-model `mmproj-F32.gguf` in GGUF dir | mmproj-F32.gguf |
| Qwen3.6-35B-A3B MoE (Carnice, Darwin APEX) | 2048 | `unsloth/Qwen3.6-35B-A3B-GGUF` | mmproj-BF16.gguf |
| Qwen2.5-Omni-3B | 2048 | Per-model dir | mmproj-F32.gguf |

**Check before launch:** `ls -la <model_dir>/mmproj*.gguf` — if it's a symlink, verify the target is the right architecture.

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

### 7. Never create new GitHub repos without explicit user permission
Existing repos can be updated when the user directs work on them. Publishing, creating, or making anything public always requires the user to say so first.

NVIDIA also sells a paid serverless tier with the same model IDs (e.g. $1.30/$2.60 for DeepSeek V4 Pro) under the same base URL. The same API key works for both, but the free tier has tighter rate limits.

### 7. The Omni*/Senter* training artifacts and OmniStep are excluded
OmniStep-SFT-8B was removed from the fleet catalog in v1.1.0 — it's no longer a fleet pick. The 12 other Omni*/Senter* training artifacts are user-made Darwin-merged models for training research. They live in the legacy catalog but not in the turbofit catalog. Add them back with `name <alias> <path>` if needed.

### 8. Hermes Desktop → Android APK via Capacitor
The Hermes Desktop app (`apps/desktop/`) is an Electron + React + Vite + Tailwind app. It can be wrapped for Android using Capacitor:

```bash
# 1. Install deps + Capacitor
cd ~/.hermes/hermes-agent && npm install --workspace apps/desktop
npm install --workspace apps/desktop @capacitor/core @capacitor/cli @capacitor/android

# 2. Init Capacitor (in apps/desktop/)
cd apps/desktop && npx cap init "Hermes" "com.nousresearch.hermes" --web-dir dist

# 3. Add Android platform
npx cap add android

# 4. Build the Vite renderer
npm run build

# 5. Sync to Android
npx cap sync android

# 6. Build APK (requires Android SDK)
export ANDROID_HOME=~/android-sdk
cd android && ./gradlew assembleDebug

# 7. Install on phone
adb install app/build/outputs/apk/debug/app-debug.apk
```

**Key insight:** The Desktop app's React renderer talks to the Hermes gateway via WebSocket — it doesn't need Electron's Node.js backend. On Android, it connects to the gateway running on the host machine. The gateway URL is configured at first launch.

**Android SDK setup (if not installed):**
```bash
mkdir -p ~/android-sdk/cmdline-tools
cd /tmp && curl -sSL -o cmdline-tools.zip \
  "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
unzip -q cmdline-tools.zip -d ~/android-sdk/cmdline-tools/
mv ~/android-sdk/cmdline-tools/cmdline-tools ~/android-sdk/cmdline-tools/latest
export ANDROID_HOME=~/android-sdk
yes | ~/android-sdk/cmdline-tools/latest/bin/sdkmanager --licenses
sdkmanager "platforms;android-34" "build-tools;34.0.0" "platform-tools"
```

## Extending turbofit

### Removing a model from the catalog

```bash
# Use Python to safely remove an entry from models.yaml
python3 -c "
import yaml
with open('$HOME/.config/turbofit/models.yaml') as f:
    cfg = yaml.safe_load(f)
if '<alias>' in cfg.get('models', {}):
    del cfg['models']['<alias>']
    print('REMOVED <alias>')
with open('$HOME/.config/turbofit/models.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f'Total models: {len(cfg[\"models\"])}')
"
```

After removal, also check if the model appears in the scaling ladder (`references/scaling-ladder.md` or `references/curated-lineup.md`) and remove/update those references.

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
- `references/api-model-rankings.md` — API main + aux rankings by volume performance, hardware tier mapping
- `southpaw-models` — curated lineup rationale (Darwin / Prism / Carnice / Carwin / Qwopus)
- `local-llm-fleet-management` — legacy catalog mechanics (now subsumed by turbofit)
- `llama-cpp` — llama.cpp build + GGUF discovery
- `omni-va-local-server` — wake-on-ping proxy (kept for Carnice slot)
- `evolutionary-radio` — where Darwin-merged checkpoints run

## Design rules (hardware-neutral, learned this session)

1. **No multi-GPU hardcoding.** Tensor split is user-configurable via catalog `extra_args: ['--tensor-split', 'X,Y']`. No preset assumes dual-GPU or specific VRAM amounts.
2. **tok/s varies by GPU.** Catalog `tok_s_target` values are measured locally — mark them as approximate. Use relative ranking (tier priority > vision bonus > speed bonus) rather than absolute claims.
3. **All models get vision.** Every local Qwen-family model can use the same `mmproj-F32.gguf` via symlink. Set `vision: true` and `mmproj:` in every catalog entry. One mmproj file serves the entire fleet.
4. **Model picks ARE the cascade.** The slot-ordered picks are the hand-chosen scale-down sequence — not generic examples. When building recommendations, preserve the actual model aliases and their tier assignments.