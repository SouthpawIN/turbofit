# turbofit — opinionated unified LLM backend for Hermes Agent

Turbofit manages the entire lifecycle of LLMs with Hermes Agent: detecting your GPU, picking the best model, launching local servers, wiring API providers, managing systemd daemons, scaling under VRAM pressure, tracking real-time pricing, and auto-updating a model database every day.

**End-user UX:** `serve auto main` → done.

## Install

```bash
# One-time: add the turbofit repo as a skill source
hermes skills tap add SouthpawIN/turbofit

# Install the skill (all 16 files — SKILL.md, references, scripts)
hermes skills install turbofit

# Update later when new models/pricing are added
hermes skills update turbofit
```

After install, `source ~/.bashrc` (or open a new shell) to get the `serve` and `name` commands.

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
serve check                       # Show version status for all installed launchers
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

Turbofit can install models as systemd user services with a **wake-on-ping proxy**. The proxy stays running (minimal memory), and the full model backend only loads when the first request arrives — freeing VRAM when idle.

```bash
serve daemon install <alias> [--idle N]   # Generate + enable systemd service
serve daemon remove <alias>              # Stop + disable + remove
serve daemon start <alias>               # Start proxy (backend wakes on ping)
serve daemon stop <alias>                # Stop daemon + kill backend (frees VRAM)
serve daemon restart <alias>             # Stop + start
serve daemon status [alias]              # Show status of one or all
serve daemon list                        # List all turbofit-managed daemons
serve daemon migrate <legacy> [alias]    # Migrate old omni-va/llama-* services to turbofit
```

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

## Scaling Ladder

When VRAM is pressured, `serve downscale` walks a conservative ladder:

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

### Modest (8-24GB, 5 steps) and Thin (<8GB, 4 steps) — see `references/scaling-ladder.md`

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

## Nous Tool Gateway

The Nous Tool Gateway (Firecrawl web search, FAL image generation, OpenAI TTS, Browser Use automation) is a **subscription feature** — it is active whenever the user has a Nous Portal subscription, regardless of which models are selected for main or aux.

Pairings in the matrix are tagged with routing indicators:
- 🟢 NOUS — both through Nous
- 🟡 NOUS+OR — main through Nous, aux through OpenRouter (10% credit bonus)
- 🟠 NOUS+NIM — main through Nous, aux through NIM (free)
- 🔵 OR — both through OpenRouter (10% bonus)
- ⚪ NIM — both through NIM (free)

---

## Registering Your Own Local Models

```bash
# Register a GGUF you've downloaded
serve register my-model /path/to/model.Q4_K_M.gguf --port 11500

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
├── skills/
│   └── turbofit/
│       ├── SKILL.md                 # Full skill documentation
│       ├── distribution.yaml        # Install manifest
│       ├── references/
│       │   ├── model-database.yaml         # Dynamic source of truth (auto-updated)
│       │   ├── model-pricing.json          # Machine-readable live pricing
│       │   ├── research-report.md          # Latest research report
│       │   ├── api-pairing-matrix.md        # Main+aux pairings by price × context
│       │   ├── scaling-ladder.md            # All-tier scaling ladders
│       │   ├── curated-lineup.md            # Model archetypes + pairing rules
│       │   ├── api-model-rankings.md        # Individual model pricing
│       │   ├── api-tier-rankings.md         # Quick-reference tiers
│       │   └── binary-selection.md          # Atomic fork vs stock decision tree
│       └── scripts/
│           ├── serve                        # Main command (2100+ lines)
│           ├── research-models.py           # Daily research (OpenRouter API + Hermes Insights)
│           ├── sync-github.sh               # GitHub sync (turbofit + sovth-config)
│           ├── install.sh                    # Shell function installer
│           └── turbofit.sharco              # Shell shim
```

---

## See Also

- [SouthpawIN/sovth-config](https://github.com/SouthpawIN/sovth-config) — overarching config collection
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) — the agent framework turbofit is built for
