# turbofit v5 — opinionated unified backend

Hardware-fit checker + multi-launcher orchestrator for Hermes-Agent. Uses `llmfit` to verify model fit, generates accurate launch strings for **llama.cpp / Ollama / vLLM / SGlang**, launches detached, wires Hermes config, and adapts to live VRAM pressure. Also bundles a curated NVIDIA NIM free-tier API list.

End-user UX: `serve auto main` → done.

For full docs: see [`SKILL.md`](../SKILL.md). For the scaling ladder: see [`scaling-ladder.md`](scaling-ladder.md). For the curated picks: see [`curated-lineup.md`](curated-lineup.md).

## Install

```bash
hermes skills install SouthpawIN/turbofit
~/.hermes/skills/turbofit/scripts/install.sh
source ~/.bashrc
```

## Supported backends

| Launcher | Binary | Use case |
|---|---|---|
| `llama-cpp` (default) | `llama-server` | GGUF files, custom builds, TurboQuant, NextN, MTP |
| `ollama` | `ollama serve` | Standard registry models, easy management |
| `vllm` | `vllm serve` | HuggingFace Safetensors, high-throughput |
| `sglang` | `python -m sglang.launch_server` | RadixAttention, low-latency |

Install any backend: `serve install <launcher>`

## UI choices (`--ui` flag)

| UI | Launches | When |
|---|---|---|
| `tui` (default) | `hermes --tui` | Terminal session |
| `dashboard` | `hermes dashboard` | Web UI in browser |
| `gateway` | `hermes gateway run` | Discord/Telegram/etc. bots |
| `desktop` | `hermes desktop` | Native desktop app |
| `herm` | `herm` (TUI) | Dashboard TUI (herm handles Hermes internally) |

## Commands

```bash
# Install / update
serve install [launcher]              # Install one (default: llama-cpp)
serve update [launcher|all]           # Update
serve check                           # Show version status

# Hardware / VRAM
serve fit <model> [ctx]               # llmfit fit check (default ctx=65536)
serve vram                            # Live GPU VRAM probe (JSON)
serve recommend                       # Scan all catalog entries, rank by fit (ctx≥64K, tok/s≥25)

# Catalog
serve register <alias> <path> [--launcher ...] [--port N]
serve catalog                         # Show registered (featured first, tier-ordered)

# Launch
serve <alias>                         # Launch detached, shows backend/port/logs
serve string <alias>                  # Print launch string (no launch)
serve stop <alias>                    # Stop running server
serve stop-all                        # Stop everything
serve list                            # List running + detect rogue servers

# Opinionated auto (v5)
serve auto main [--vision] [--ui ...] # Pick best main, launch, wire Hermes
serve auto aux [--ui ...]             # Pick best aux, launch, wire Hermes
serve downscale                       # Adapt to current VRAM pressure
AUTO_CTX=131072 serve auto main       # Raise ctx target

# Fetch / bench
serve fetch <alias>                   # Download missing model from HF
serve bench <alias>                   # lm-eval-harness benchmark (mmlu + gsm8k)

# Hermes routing
serve main <alias> [--ui ...]         # Launch + set main + start UI
serve aux <alias> [--ui ...]          # Launch + set aux + start UI
serve herm <alias>                    # Launch + main + herm TUI
serve herm aux <alias>                # Launch + aux + herm TUI

# NVIDIA NIM API (curated, 5 free models)
serve api list                        # Show curated lineup
serve api use <rank|api_id> [main|aux]# Wire API model into Hermes config
```

## Named flag presets

Apply via `presets:` in catalog. 18 presets total:

| Preset | Expands to |
|---|---|
| `nextn` / `nextn-tight` | `--spec-type nextn --draft-block-size 3/2` |
| `draft-mtp` / `draft-mtp-tight` | `--spec-type draft-mtp --draft-block-size 3/2` |
| `turbo4-kv` / `turbo3-kv` / `turbo2-kv` | `-ctk turboN -ctv turboN` |
| `q8-kv` / `q4-kv` | `-ctk q8_0/q4_0 -ctv q8_0/q4_0` |
| `cpu-moe-2` / `cpu-moe-4` / `cpu-moe-8` | `--n-cpu-moe N` (MoE expert offload) |
| `no-mmap` / `split-none` / `mlock` | `--no-mmap` / `--split-mode none` / `--mlock` |
| `parallel-4` / `parallel-2` / `parallel-1` | `--parallel N` |

Multi-GPU tensor split: use `extra_args: ['--tensor-split', 'X,Y']` in catalog.

## Curated lineup

The auto-picker walks this slot-ordered list from `~/.config/turbofit/curated.yaml`:

| Slot | Source | Tier ladder |
|---|---|---|
| `main_local` | `models.yaml` | s → sf → f → c |
| `main_api` | OpenRouter + NVIDIA NIM | s → sf → f → c |
| `aux_local` | `models.yaml` | sd → c |
| `aux_api` | OpenRouter + NVIDIA NIM | sd → c |

## License

MIT