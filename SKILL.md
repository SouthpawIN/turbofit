---
name: turbofit
description: "Hardware-fit checker + multi-launcher installer (llama.cpp / Ollama / vllm / SGlang) for Hermes-Agent. Uses llmfit to verify if a model fits your VRAM/RAM, generates the right launch string for the right backend, launches the server detached, and wires it into Hermes as main or auxiliary. Also bundles a curated NVIDIA NIM API list. Enforces a 64K context floor."
version: 4.0.0
author: SouthpawIN
license: MIT
tags: [llama.cpp, ollama, vllm, sglang, llmfit, gguf, hermes-agent, nvidia-nim]
metadata:
  hermes:
    related_skills: [local-llm-fleet-management, llama-cpp, gguf-quantization, southpaw-models]
---

# Turbofit

Four questions, one workflow:

1. **Will this model fit in my system memory?** → `serve fit <model>` (uses `llmfit`)
2. **Which backend should run it?** → `llama.cpp` (default), `ollama`, `vllm`, or `sglang`
3. **What's the launch string for that backend?** → `serve string <alias>` or `serve <alias>`
4. **Run it and wire it into Hermes?** → `serve main <alias> [--ui ...]` / `serve aux <alias> [--ui ...]`

Turbofit installs and updates `llama.cpp` from source, registers models in a YAML catalog with a per-model backend choice, generates accurate launch strings for each backend, launches detached, and rewrites `~/.hermes/config.yaml` to route Hermes to it. It also bundles a curated list of 5 NVIDIA NIM API models.

## ⚠ Context Floor: 64K (65536 tokens)

Hermes-Agent requires at least 64K context. Every launch string and server `serve` produces uses **`ctx_size: 65536` as the hard minimum**, regardless of what llmfit recommends. The script clamps `CTX = max(llmfit_value, 65536)` before printing.

## Install

```bash
# 1. Install the skill
hermes skills install SouthpawIN/turbofit

# 2. Wire serve/name into your shell
~/.hermes/skills/turbofit/scripts/install.sh
source ~/.bashrc

# 3. Install llama.cpp (or another backend)
serve install                  # installs llama.cpp
serve install ollama           # installs Ollama
serve install vllm             # installs vllm
serve install sglang           # installs SGlang

# 4. Verify
serve check
serve fit "Qwen/Qwen3-4B"
```

## Supported Launchers

| Name | Binary | Use when |
|------|--------|----------|
| `llama-cpp` (default) | `llama-server` | GGUF files, custom builds, TurboQuant |
| `ollama` | `ollama serve` | Standard ollama models, easy model management |
| `vllm` | `vllm serve` | HuggingFace HF/Safetensors, high throughput |
| `sglang` | `python -m sglang.launch_server` | RadixAttention, low-latency serving |

Future community projects can be added by extending `LAUNCHER_NAME`/`LAUNCHER_TEMPLATE`/`LAUNCHER_INSTALL` in `scripts/serve`.

## The `serve` command

```bash
# Install / update
serve install [launcher]                 Install a launcher (default: llama-cpp)
serve update [launcher|all]              Update a launcher or all
serve check                              Show version status for all installed

# Hardware fit
serve fit <model> [ctx]                  Run llmfit fit check (default ctx=65536)

# Catalog
serve register <alias> <path>            Register a model
           [--launcher llama-cpp|ollama|vllm|sglang]
           [--port N]
serve catalog                            Show registered aliases

# Launch
serve <alias>                            Launch (detached, shows backend/port/logs)
serve string <alias>                     Print launch string (no launch)
serve stop <alias>                       Stop a running server
serve list                               List running servers

# Hermes model routing + UI
serve main <alias> [--ui tui|dashboard|gateway|desktop|herm]
serve aux  <alias> [--ui ...]
serve herm <alias>                       Launch + main + herm TUI + hermes
serve herm aux <alias>                   Launch + aux + herm TUI + hermes

# NVIDIA NIM API (curated list)
serve api list                           Show the curated NVIDIA NIM list
serve api use <rank|api_id> [main|aux]   Wire a NIM model into Hermes config
```

## UI Choices (`--ui` flag)

| Value | Launches | Use when |
|-------|----------|----------|
| `tui` (default) | `hermes --tui` | Terminal session |
| `dashboard` | `hermes dashboard` | Web UI in browser |
| `gateway` | `hermes gateway run` | Discord/Telegram/etc. bot |
| `desktop` | `hermes desktop` | Native desktop app (when supported on your OS) |
| `herm` | `herm` (background) + `hermes --tui` | Both UIs |

Append `--gateway` to `serve main` / `serve aux` for the gateway variant.
Append `--ui <value>` for any specific UI.

## NVIDIA NIM API (curated list)

Bundled at `~/.local/share/turbofit/nvidia-nim-curated.yaml`. Verified from `build.nvidia.com` on 2026-06-22:

| Rank | Model | Vision | $/in | $/out | API ID | Recommended aux |
|------|-------|--------|------|-------|--------|----------------|
| 1 | DeepSeek V4 Pro | no | $1.30 | $2.60 | `deepseek-ai/deepseek-v4-pro` | MiniMax M3 |
| 2 | GLM 5.1 | no | $0.85 | $3.10 | `z-ai/glm-5.1` | MiniMax M3 |
| 3 | DeepSeek V4 Flash | no | $0.10 | $0.20 | `deepseek-ai/deepseek-v4-flash` | MiniMax M3 |
| 4 | MiniMax M3 | 👁 yes | $0.30 | $1.20 | `minimaxai/minimax-m3` | DeepSeek V4 Flash |
| 5 | Nemotron Ultra | 👁 yes | $0.60 | $3.60 | `nvidia/nemotron-3-ultra-550b-a55b` | MiniMax M3 |

```bash
serve api list                           # show the list
serve api use 1 main                     # rank 1 as main
serve api use 4 aux                      # rank 4 as aux (all 9 aux tasks)
serve api use minimaxai/minimax-m3 main  # use API ID directly
```

Requires `NVIDIA_API_KEY` in your environment (or `~/.hermes/.env`).

## Shell aliases (auto-installed)

After `install.sh`:

```bash
name <alias> <path> [--launcher ...]     Register a model alias
serve <alias>                            Launch + show backend/port/logs
serve main <alias> [--ui ...]            Launch + set main + start UI
serve aux <alias> [--ui ...]             Launch + set aux + start UI
serve herm <alias>                       Launch + main + herm + hermes
serve herm aux <alias>                   Launch + aux + herm + hermes
```

## How It Works

### Hardware fit check

`serve fit <model>` runs `llmfit plan` to estimate required VRAM/RAM for a model at a given context length. Returns a fit level:

| Level | Meaning |
|-------|---------|
| Perfect | Fully GPU-resident, recommended |
| Good | Minor headroom pressure, still fast |
| Marginal | CPU offload needed or tight fit |
| Too Tight | Won't run acceptably |

### Launch string per backend

The string changes by launcher (verified against each tool's docs):

```bash
# llama.cpp (default)
llama-server -m <path> --host 127.0.0.1 --port 8080 -ngl 99 -fa on -c 65536 --jinja -t 32

# Ollama
OLLAMA_HOST=127.0.0.1:8080 ollama serve

# vllm
vllm serve <path> --host 127.0.0.1 --port 8080 --max-model-len 65536 --gpu-memory-utilization 0.9 --served-model-name <alias>

# SGlang
python -m sglang.launch_server --model-path <path> --host 127.0.0.1 --port 8080 --context-length 65536 --served-model-name <alias> --tp 1
```

### Server lifecycle

`serve <alias>`:
1. Kills any existing server bound to the alias's port
2. Spawns the backend via `nohup ... &; disown` (detached, survives shell death)
3. Waits up to 120s for `/health` or `/v1/models` to respond
4. Writes PID to `~/.local/share/turbofit/pid/<alias>`
5. Prints backend, port, PID, log path on success

### Hermes-Agent config

`serve main` / `serve aux` rewrite `~/.hermes/config.yaml` before starting the UI, so the new config is picked up immediately:

```yaml
model:
  default: <alias>
  base_url: http://127.0.0.1:8080/v1
  api_key: not-needed
```

For `aux`, all 9 auxiliary tasks (vision, web_extract, compression, session_search, skills_hub, approval, mcp, title_generation, curator) are rewired.

## Catalog format

Models are registered in `~/.config/turbofit/models.yaml`:

```yaml
models:
  qwen-8b:
    launcher: llama-cpp
    path: /home/user/models/Qwen3-8B.Q4_K_M.gguf
    port: 8080
    ctx: 65536
  my-ollama-model:
    launcher: ollama
    path: qwen2.5:7b
    port: 8081
    ctx: 65536
  my-vllm-model:
    launcher: vllm
    path: Qwen/Qwen3-8B
    port: 8082
    ctx: 65536
```

Ports auto-increment (8080, 8081, 8082...) so multiple backends can coexist.

## Memory-Fit Reference

### Quick decision tree

| System | Best fit |
|--------|----------|
| 8 GB VRAM (RTX 3060/3070, M1/M2 base) | 7B-8B Q4_K_M, up to 13B Q3 |
| 12 GB VRAM (RTX 3060 12GB, 4070) | 13B Q4_K_M, 8B Q8_0 |
| 16 GB VRAM (RTX 4060 Ti 16, 4080) | 14B-15B Q4_K_M, 27B Q3 |
| 24 GB VRAM (RTX 3090/4090) | 27B Q4_K_M, up to 70B Q2 |
| 48 GB VRAM (A6000, RTX 6000 Ada) | 70B Q4_K_M, 120B Q3 |
| 80 GB VRAM (H100, A100) | 70B Q8_0, 120B+ Q4 |
| Apple Silicon unified memory | scales linearly (M2 Max 96GB ≈ 96GB VRAM) |
| No GPU | drop `-ngl 99`, expect 2-10 tok/s for 7B |

### 64K context VRAM budget (q8_0 KV cache)

| Model size | 64K KV cache | Total VRAM needed |
|------------|--------------|-------------------|
| 8B Q4_K_M  | ~2 GB        | ~7 GB             |
| 13B Q4_K_M | ~2 GB        | ~10 GB            |
| 27B Q4_K_M | ~2 GB        | ~19 GB            |
| 70B Q4_K_M | ~2 GB        | ~42 GB            |

## Pitfalls

### 1. `flash_attn` is tri-state, not a bare flag
`--flash-attn` (bare) errors out. Use `-fa on` / `-fa off` / `-fa auto`.

### 2. `--gpu` is not a valid llama.cpp flag
Use `-ngl N` to control GPU layers (or `--device CUDA0` + `--main-gpu 0` for device selection). `--gpu` will error.

### 3. Context below 64K breaks Hermes-Agent (HARD FLOOR)
Hermes-Agent requires `context_length >= 65536`. A launch string with `-c 8192` will load fine but Hermes will crash the moment it tries to use its full system prompt + tool registry + history.

`serve` clamps `ctx` to 65536 minimum regardless of what you pass.

### 4. YaRN scaling beyond native context
If the model natively supports less than 64K, use `--rope-scaling yarn --yarn-orig-ctx <native>` to extend:
- Qwen3 256K native → use as-is up to 256K
- Older models 8K native → extend with YaRN to 64K+ (`--rope-scaling yarn --yarn-orig-ctx 8192 -c 65536`)
- 1M context → `--rope-scaling yarn --yarn-orig-ctx 262144 -c 1048576` (extreme, monitor VRAM)

### 5. `serve <alias>` waits up to 120s for health
Large models take time to load. If `/health` or `/v1/models` doesn't respond within 120s, `serve <alias>` reports failure — but the process might still be alive. Check `~/.local/share/turbofit/logs/<alias>.log` for the real status.

### 6. Multiple backends on the same port
`serve <alias>` kills any existing process bound to the alias's port. If you registered two aliases to the same port manually, the second `serve` will kill the first.

### 7. Backend-specific quirks
- **llama.cpp**: requires build tools (cmake, gcc, CUDA). TurboQuant GGUF files need `llama.cpp-tq3` fork.
- **Ollama**: `ollama pull <model>` runs once before serving. Default model registry at `ollama.com/library`.
- **vllm**: requires `nvidia-smi` to see GPUs; needs `--tensor-parallel-size` for multi-GPU models.
- **SGlang**: requires `--tp N` for tensor parallelism; `--context-length` is the hard max, not the requested.

### 8. `serve install` requires build tools (for llama-cpp)
CMake, gcc/g++, CUDA toolkit (for GPU builds). If `cmake -B build` fails, install them first:
```bash
sudo apt install cmake build-essential    # Debian/Ubuntu
sudo dnf install cmake gcc-c++            # Fedora
brew install cmake                        # macOS (CPU only — CUDA needs NVIDIA SDK)
```

### 9. NVIDIA NIM requires an API key
Set `NVIDIA_API_KEY` in `~/.hermes/.env` or shell environment before `serve api use`.

### 10. Hermes Desktop is platform-dependent
`hermes desktop` works where the native desktop build is available. Linux install support varies by platform — verify on your distro.

## Extending with new launchers

To add a new backend (e.g. when you find a new community project):

1. Add entries to the four associative arrays in `scripts/serve`:
   ```bash
   [my-launcher]="My Launcher Name"
   [my-launcher]="my-launcher:install_method"
   [my-launcher]='my-launcher command with $MODEL_PATH $PORT $HOST $CTX $NGL'
   [my-launcher]='http://$HOST:$PORT/v1'
   ```
2. Add a case in `install_launcher()`, `update_all_launchers()`, and `launch_server()`.
3. Document in `USAGE` and `LAUNCHER_NAME` table.

## See Also

- `local-llm-fleet-management` — multi-model catalog + swap pattern
- `llama-cpp` — llama.cpp build + GGUF discovery
- `gguf-quantization` — GGUF format deep dive
- `southpaw-models` — local curated model picks (5-model lineup)
- `herm` — the Herm TUI dashboard (optional companion for `serve herm`)