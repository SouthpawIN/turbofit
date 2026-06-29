# turbofit — opinionated unified LLM backend for Hermes Agent

Turbofit manages the entire lifecycle of LLMs with Hermes Agent: detecting your GPU, picking the best model, launching local servers, wiring API providers, managing daemons (systemd on Linux, PID files on Windows), scaling under VRAM pressure, tracking real-time pricing, integrating Mixture of Agents, and auto-updating a model database every day.

**End-user UX:** `serve auto main` → done. Works on Linux and Windows.

## What's New (v1.1)

- **🪟 Native Windows support** — no Docker, no WSL, no systemd. Auto-detects OS, manages daemons via PID files on Windows. Same `serve` command, both platforms.
- **🧠 Mixture of Agents (MoA) integration** — 5 presets that pair local + API models for multi-model reasoning. `serve moa recommend` picks the best preset based on live VRAM.
- **🔧 Scaling watcher fix** — auto-scaling now triggers on absolute per-GPU free VRAM (not just external pressure). Turbofit properly backs off when any process — including its own daemons — causes VRAM pressure. No more OOM crashes.

---

## Install

```bash
# Direct install from GitHub (SKILL.md at repo root)
hermes skills install https://github.com/SouthpawIN/turbofit

# Or use the tap for easier updates
hermes skills tap add SouthpawIN/turbofit
hermes skills install turbofit
```

After install, `source ~/.bashrc` (or open a new shell) to get the `serve` and `name` commands.

Update later with `hermes skills update turbofit`.

---

## Quick Start

```bash
# Let turbofit detect your hardware and pick the best setup
serve auto main

# Or force API-only mode (no GPU needed)
serve auto main --api

# Or restrict to free endpoints only
serve auto main --free

# Check what's running
serve list

# Check your GPU VRAM
serve vram

# Browse your model catalog
serve catalog

# Stop everything
serve stop-all
```

---

## Hardware Tiers (auto-detected)

`serve auto` probes `nvidia-smi` and picks a tier:

| Tier | VRAM | What happens |
|------|------|-------------|
| **Beefy** | ≥24GB | Local main + local aux (full dual-GPU setup) |
| **Modest** | 8-24GB | API main + free/cheap aux |
| **Thin** | <8GB or no GPU | API main + API aux (zero-cost option available) |

No NVIDIA GPU → defaults to Thin (API-only). Use `--api` to force API mode, `--free` for free endpoints only.

---

## Complete Command Reference

### Install / Update Launchers

```bash
serve install                    # Install llama.cpp from source
serve install <launcher>         # Install one: llama-cpp, ollama, vllm, sglang
serve update                     # Update llama.cpp (git pull + rebuild)
serve update <launcher|all>      # Update specific or all launchers
serve check                      # Show version status for all installed launchers
```

### Hardware / VRAM

```bash
serve fit <model> [ctx]          # Check if a model fits in VRAM (via llmfit, default ctx=65536)
serve vram                        # Live GPU VRAM probe (JSON output)
serve recommend                   # Scan all catalog entries, rank by fit:
                                  #   ctx ≥ 64K, tok/s ≥ 25, Q4, vision bonus, tier priority
```

### Catalog Management

```bash
serve register <alias> <path>    # Register a new model
           [--launcher llama-cpp|ollama|vllm|sglang]
           [--port N]
serve catalog                     # Browse all registered models (featured first, tier-ordered)
name <alias> <path>              # Shortcut for register (installed as bash function)
```

Each catalog entry supports:
- **Launcher:** llama-cpp, ollama, vllm, sglang
- **Presets:** 18 named flag bundles (nextn, draft-mtp, turbo4-kv, q8-kv, cpu-moe-4, no-mmap, mlock, parallel-4, etc.)
- **Binary pin:** atomic fork for TurboQuant+NextN, stock for legacy
- **Tier:** s / sf / sd / f / c (used by auto-picker for ranking)
- **Vision:** mmproj file for multimodal
- **Role:** main / aux / either

### Launch / Stop

```bash
serve <alias>                     # Launch a model (detached, shows backend/port/logs)
serve string <alias>              # Print the launch command without launching (dry run)
serve stop <alias>                # Stop a specific model
serve stop-all                    # Stop everything
serve list                        # List running servers + detect rogue llama-servers on any port
```

### Opinionated Auto (hardware-aware, API-aware)

```bash
serve auto main [--vision] [--api] [--free] [--ui tui|dashboard|gateway|desktop|herm]
serve auto aux  [--vision] [--api] [--free] [--ui ...]
serve downscale                   # Probe VRAM, walk the scaling ladder
AUTO_CTX=131072 serve auto main   # Override the context target
```

Flags:
- `--vision` — require vision capability
- `--api` — force API mode (no local GPU needed)
- `--free` — only free API endpoints
- `--ui` — start a Hermes frontend after wiring (tui, dashboard, gateway, desktop, herm)

### Hermes Config Wiring

```bash
serve main <alias> [--ui tui|dashboard|gateway|desktop|herm]   # Launch + set as main + start UI
serve aux  <alias> [--ui ...]                                    # Launch + set as aux
serve herm <alias>               # Launch + main + herm TUI
serve herm aux <alias>           # Launch + aux + herm TUI
serve herm                       # Auto-pick main + launch herm TUI
```

### NVIDIA NIM Free API

Turbofit ships a curated list of free NVIDIA NIM endpoints:

```bash
serve api list                           # Show curated NIM models with pricing/vision/ctx
serve api use <rank|api_id> [main|aux]    # Wire a NIM model into Hermes config
```

Free models (via `NVIDIA_API_KEY` from build.nvidia.com, ~1000 RPM, no credit card):
- DeepSeek V4 Pro (1M ctx, no vision)
- DeepSeek V4 Flash (1M ctx, no vision)
- MiniMax M3 (1M ctx, vision)
- Nemotron Ultra 550B (1M ctx, vision)

### Systemd Daemon Management

Turbofit can install models as systemd user services (Linux) or PID-managed daemons (Windows) with a **wake-on-ping proxy**. The proxy stays running (minimal memory), and the full model backend only loads when the first request arrives — freeing VRAM when idle.

**Linux:** systemd user services (`turbofit-<alias>.service`)
**Windows:** PID files in `~/.config/turbofit/daemons/` + `taskkill`/`tasklist` for process management

```bash
serve daemon install <alias> [--idle N]   # Generate + enable service (Linux) / PID daemon (Windows)
serve daemon remove <alias>              # Stop + disable + remove
serve daemon start <alias>               # Start proxy (backend wakes on ping)
serve daemon stop <alias>                # Stop daemon + kill backend (frees VRAM)
serve daemon restart <alias>             # Stop + start
serve daemon status [alias]              # Show status of one or all
serve daemon list                        # List all turbofit-managed daemons
serve daemon migrate <legacy> [alias]    # Migrate old omni-va/llama-* services to turbofit
```

### Mixture of Agents (MoA)

Turbofit integrates with [Hermes MoA](https://hermes-agent.nousresearch.com/docs/user-guide/features/mixture-of-agents) — reference models analyze first (no tools), then the aggregator synthesizes the final response with full tool access. MoA beats any single model on quality.

**5 presets out of the box:**

| Preset | References | Aggregator | Cost | Use Case |
|--------|-----------|------------|------|----------|
| `default` | Darwin + DeepSeek V4 Pro | GLM 5.2 | Low | Best quality, balanced cost |
| `local` | Carnice | Darwin | **$0** | Zero API cost, fully local |
| `reasoning` | Darwin + DeepSeek + Qwen 3.7 MAX | GLM 5.2 | Medium | Maximum reasoning power |
| `fast` | Carnice | DeepSeek V4 Flash | Minimal | Speed-optimized |
| `review` | Darwin + DeepSeek V4 Pro | GLM 5.2 | Low | Code review (low temp) |

```bash
serve moa list                    # List configured presets
serve moa recommend               # Hardware-aware preset recommendation (checks VRAM + running models)
serve moa status                  # Show active preset
serve moa presets                 # Full preset details
serve moa use <preset>            # Print activation command
serve moa shot <prompt>           # Print one-shot command
```

Activate in Hermes: `/model <preset> --provider moa`
One-shot: `/moa <prompt>`

### Fetch / Benchmark

```bash
serve fetch <alias>              # Download model from HuggingFace (uses hf_repo in catalog)
                                 # Only downloads Q4_K_M + mmproj — never wildcards
serve bench <alias>              # Run lm-eval-harness benchmark (launches model if needed)
serve bench compare_27b          # Run benchmark group (head-to-head comparison)
```

### Research / Cost Projections

```bash
# Run the research script manually (fetches live OpenRouter API pricing)
python3 ~/.hermes/skills/turbofit/scripts/research-models.py

# Check the latest report (includes your real usage data from Hermes Insights)
cat ~/.hermes/skills/turbofit/references/research-report.md

# Sync to GitHub (pushes to SouthpawIN/turbofit + SouthpawIN/sovth-config)
bash ~/.hermes/skills/turbofit/scripts/sync-github.sh
```

The research script:
1. Fetches live pricing from `https://openrouter.ai/api/v1/models` (339+ models)
2. Reads your **actual usage** from Hermes state.db (real input/output/cache tokens, real cache hit rate, real cost)
3. Projects monthly cost for each model based on **your actual usage patterns**
4. Projects pairing costs with aux offset (40-85% of tokens route to aux)
5. Reports cache savings for models that support cache reads

---

## Platform Support

### Linux (primary)
- Full systemd integration for daemon management
- Scaling watcher with auto-contraction/expansion
- Gateway proxy and status server
- All features supported

### Windows 11
- Native support — no Docker, no WSL required
- Requirements: [Git Bash](https://git-scm.com/downloads), `llama-server.exe` in PATH, NVIDIA drivers (for `nvidia-smi`)
- Daemon management via PID files (`~/.config/turbofit/daemons/`) + `taskkill`/`tasklist`
- Same `serve` command works identically
- Auto-appends `.exe` to model binaries
- Limitations: no scaling watcher, no gateway proxy (Linux-specific extras)

---

## Scaling Ladder

When VRAM is pressured, turbofit automatically backs off. The scaling watcher monitors per-GPU free VRAM every 30 seconds and walks a conservative contraction ladder:

**Contraction** (when VRAM is tight):
| Free VRAM | Action |
|-----------|--------|
| <6GB | Shrink context (262K → 131K → 65K) |
| <4GB | Expert offload (MoE → CPU) |
| <3GB | Swap to smaller model |
| <2GB | Stop aux daemons |
| <1GB | Stop main → API fallback |

**Expansion** (when VRAM recovers, with +4GB hysteresis):
| Free VRAM | Action |
|-----------|--------|
| >5GB | Restart main |
| >6GB | Restart aux |
| >7GB | Swap back to big model |
| >8GB | Restore experts to GPU |
| >10GB | Restore full context |

The watcher contracts on **absolute per-GPU free VRAM** — it doesn't matter whether the pressure comes from external apps (ComfyUI, games) or turbofit's own daemons (ACE-Step, another model loading). Turbofit always backs off to make room.

Manually trigger with `serve downscale`.

### Beefy (≥24GB, 7 steps)

| Step | State | Main | Aux | Context |
|------|-------|------|-----|---------|
| 1 | Ideal | 27-28B dense (Q4) | 35B MoE (3B active) | 1M |
| 2 | Mild pressure | 27-28B dense | 35B MoE (cpu-moe) | 1M |
| 3 | Moderate | 27-28B dense | 35B MoE | 512K |
| 4 | High | 27-28B dense | API vision (free) | 262K |
| 5 | Critical | 27B hybrid/Mamba | API vision (cheap) | 262K |
| 6 | Extreme | 35B MoE (3B active) | API vision (cheap) | 132K |
| 7 | API-only | API main | API vision | 1M |

Main is protected until Step 5. The ladder never kills a model mid-response.

### Modest (8-24GB, 5 steps) and Thin (<8GB, 4 steps)

See [`references/scaling-ladder.md`](references/scaling-ladder.md) for full details.

---

## Dynamic Model Database

The model database (`references/model-database.yaml`) auto-updates daily via a cron job:

1. **Research script** fetches live OpenRouter API data (pricing, cache rates, context, vision)
2. **Agent reviews** the report and updates the database
3. **GitHub sync** pushes to `SouthpawIN/turbofit`
4. **Users** get fresh data via `hermes skills update turbofit`

Each model entry includes:
- Pricing across all providers (Nous, OpenRouter, NIM, direct API)
- Cache read pricing (78-99% savings on cache hits for models that support it)
- Context window and supported context tiers
- Vision capability (from OpenRouter API's `architecture.input_modalities`)
- Benchmark scores when available
- Local model info (GGUF repo, quants, size, archetypes, mmproj)

---

## Catalog Schema

Register models in `~/.config/turbofit/models.yaml`:

```yaml
models:
  my-model:
    launcher: llama-cpp           # llama-cpp | ollama | vllm | sglang
    path: /abs/path/to/model.gguf # OR HF repo for vllm/sglang
    port: 11500                    # auto-assigned if absent
    ctx: 262144                    # 64K floor enforced
    gpu: 0                         # 0 | 1 (single-GPU target)
    mmproj: /path/to/mmproj.gguf  # vision projector (optional)
    presets: [nextn, turbo4-kv]   # named flag bundles (see below)
    extra_args: [--draft-block-size 3]  # raw flag list
    aliases: [short, alt]          # alternative names
    tier: s                        # s | sf | sd | f | c (used by serve auto)
    featured: true                 # shown first in catalog
    tok_s_target: 107              # measured throughput
    vision: true                   # has vision tower + mmproj
    size_gb: 16.0                  # disk footprint
    hf_repo: org/repo              # for serve fetch
    role: main                     # main | aux | either
```

### Named Flag Presets

Apply by listing in `presets:`. Multiple presets merge; later presets override earlier flags.

| Preset | Expands to |
|--------|------------|
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

Multi-GPU tensor split: use `extra_args: ['--tensor-split', 'X,Y']` — no hardcoded preset.

### Tier Ladder (used by `serve auto`)

**Main candidates (Tier S + SF only):**

| Tier | Meaning | Examples |
|------|---------|----------|
| `s` | Smartest | Darwin Reason, Darwin Apex-Compact, Prism Eagle |
| `sf` | Smart + fast | Carwin-MTP, Qwopus v2-MTP, Qwopus Coder-MTP, Carnice |

**Auxiliary only (Tier F + C — not recommended for main):**

| Tier | Meaning | Examples |
|------|---------|----------|
| `f` | Fast | Qwable MTP, Qwopus Abliterated-MTP |
| `c` | Cheap | Qwen legacy, devstral, step-flash, omni-3b |

`serve auto main` will only pick from tiers S and SF. Tier F and C models are reserved for the aux role, vision fallback, or lightweight tasks.

---

## Nous Tool Gateway

The Nous Tool Gateway (Firecrawl web search, FAL image generation, OpenAI TTS, Browser Use automation) is a **subscription feature** — it is active whenever the user has a Nous Portal subscription, regardless of which models are selected for main or aux.

Pairings in the matrix are tagged with routing indicators:
- 🟢 **NOUS** — both through Nous
- 🟡 **NOUS+OR** — main through Nous, aux through OpenRouter (10% credit bonus)
- 🟠 **NOUS+NIM** — main through Nous, aux through NIM (free)
- 🔵 **OR** — both through OpenRouter (10% bonus)
- ⚪ **NIM** — both through NIM (free)

---

## Registering Your Own Local Models

```bash
# Register a GGUF you've downloaded
serve register my-model /path/to/model.Q4_K_M.gguf --port 11500

# Or use the name shortcut
name my-model /path/to/model.Q4_K_M.gguf

# Then edit ~/.config/turbofit/models.yaml to set:
#   tier: s          # s | sf | sd | f | c
#   vision: true     # if it has an mmproj
#   role: main       # main | aux | either
#   size_gb: 16.0    # disk footprint
#   presets: [nextn, turbo4-kv, no-mmap]  # flag bundles
#   binary: /path/to/atomic-fork/llama-server  # if using TurboQuant
```

The `serve auto` picker scans your catalog and picks the best model based on tier, context, VRAM, and vision.

---

## What Turbofit Replaces

Turbofit consolidates three previous systems:
- **llama-launch** — old model launcher catalog (23 models migrated)
- **omni-va** — wake-on-ping proxy (subsumed into daemon system)
- **ad-hoc scripts** — `start-*-server.sh`, `model-server.sh` (13 dead launchers consolidated)

Use `serve daemon migrate` to convert legacy systemd services.

---

## File Structure

```
turbofit/
├── README.md                        # This file
├── SKILL.md                         # Full skill documentation
├── distribution.yaml                # Install manifest
├── references/
│   ├── model-database.yaml          # Dynamic source of truth (auto-updated)
│   ├── model-pricing.json           # Machine-readable live pricing
│   ├── research-report.md           # Latest research report
│   ├── api-pairing-matrix.md        # Main+aux pairings by price × context
│   ├── scaling-ladder.md            # All-tier scaling ladders
│   ├── curated-lineup.md            # Model archetypes + pairing rules
│   ├── api-model-rankings.md        # Individual model pricing
│   ├── api-tier-rankings.md         # Quick-reference tiers
│   └── binary-selection.md          # Atomic fork vs stock decision tree
├── scripts/
│   ├── serve                        # Main command (2100+ lines)
│   ├── research-models.py           # Daily research (OpenRouter API + Hermes Insights)
│   ├── sync-github.sh               # GitHub sync (turbofit + sovth-config)
│   ├── install.sh                   # Shell function installer
│   └── turbofit.sharco              # Shell shim
└── tools/                           # Benchmark/research utilities (not installed)
```

---

## See Also

- [SouthpawIN/sovth-config](https://github.com/SouthpawIN/sovth-config) — overarching config collection
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) — the agent framework turbofit is built for
