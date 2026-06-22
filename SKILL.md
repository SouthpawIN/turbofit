---
name: turbofit
description: "Generate, install, and launch llama.cpp (llama-server) with the `serve` and `name` shell commands. Auto-installs llama.cpp from source, updates it when stale, uses llmfit to verify model fit in VRAM/RAM, launches the server detached, and wires it into Hermes-Agent as the main or auxiliary model. Enforces a 64K context floor."
version: 3.2.0
author: SouthpawIN
license: MIT
tags: [llama.cpp, llama-server, llmfit, gguf, inference, hermes-agent]
metadata:
  hermes:
    related_skills: [local-llm-fleet-management, llama-cpp, gguf-quantization]
---

# Turbofit

Three questions, one workflow:

1. **Will this model fit in my system memory?** → `serve fit <model>`
2. **Launch it on a port** → `serve <alias>` (or `name <alias> <path>` to register first)
3. **Launch it and run Hermes on it** → `serve main <alias>` / `serve aux <alias>` / `serve herm <alias>` / `serve herm aux <alias>`

Turbofit installs and updates `llama.cpp` from source, uses [`llmfit`](https://github.com/AlexsJones/llmfit) for fit analysis, launches `llama-server` detached (so it survives shell death), and rewrites `~/.hermes/config.yaml` to make the running server the main or auxiliary model for Hermes-Agent.

## ⚠ Context Floor: 64K (65536 tokens)

Hermes-Agent requires at least 64K context. Every launch string and server `serve` produces uses **`ctx_size: 65536` as the hard minimum**, regardless of what llmfit recommends. The script clamps `CTX = max(llmfit_value, 65536)` before printing.

- Quick checks below 64K are **disallowed**
- `--ctx` override is clamped to 65536 minimum
- VRAM headroom is budgeted assuming 64K minimum

## Install

```bash
# 1. Install the skill (one time)
hermes skills install SouthpawIN/turbofit

# 2. Wire serve/name into your shell
~/.hermes/skills/turbofit/scripts/install.sh
source ~/.bashrc   # or open a new shell

# 3. Install llama.cpp (auto-built from source)
serve install

# 4. (Optional) Verify
serve check
serve fit "Qwen/Qwen3-4B"
```

## The `serve` command

```bash
serve install                                 Install llama.cpp from source
serve update                                  Update llama.cpp to latest master
serve check                                   Show binary version + commit status
serve fit <model> [ctx]                       Run llmfit fit check

serve register <alias> <path>                 Register a model alias
serve catalog                                 Show registered aliases

serve <alias>                                 Launch the server (detached, shows port + logs)
serve string <alias>                          Print launch string (no launch)
serve stop <alias>                            Stop a running server
serve list                                    List running servers

serve main <alias>                            Launch + set as Hermes main + start hermes
serve aux <alias>                             Launch + set as Hermes aux + start hermes
serve herm <alias>                            Launch + set main + start herm TUI + hermes
serve herm aux <alias>                        Launch + set aux + start herm TUI + hermes
```

Append `--gateway` to `serve main` or `serve aux` to start the hermes gateway instead of the TUI:

```bash
serve main my-alias --gateway                 Launch + main model + start hermes gateway
serve aux my-alias --gateway                  Launch + aux model + start hermes gateway
```

## Shell aliases (auto-installed)

After `install.sh`, you get `serve` and `name` shell functions:

```bash
# Register a model alias
name <alias> <path-to-gguf-or-hf-repo>
# e.g.
name my-qwen ~/models/Qwen3-8B.Q4_K_M.gguf
name hermes-main ~/models/Darwin-Reason-28B.Q4_K_M.gguf

# Launch the server (detached) — shows port, PID, logs
serve <alias>

# Launch + start Hermes with this model set as main or aux
serve main <alias>                  # TUI with main model
serve main <alias> --gateway        # gateway with main model
serve aux <alias>                   # TUI with aux model
serve aux <alias> --gateway         # gateway with aux model

# Launch + start herm TUI + hermes
serve herm <alias>                  # herm + hermes TUI, model as main
serve herm aux <alias>              # herm + hermes TUI, model as aux (all 9 tasks)
```

`name` is a thin wrapper around `serve register`. `serve` passes through directly to the dispatcher, which detects whether the first argument is a registered alias, a known subcommand, or `main`/`aux`/`herm`.

## How It Works

### 1. llama.cpp management

`serve install` clones `ggerganov/llama.cpp` to `~/.local/src/llama.cpp`, builds with CUDA enabled, and installs to `~/.local/bin/llama-server`.

`serve update` pulls latest master, rebuilds, and re-installs. Safe to run repeatedly.

`serve check` shows the binary version, the local commit, the remote commit, and whether you're out of date.

### 2. llmfit fit analysis

`serve fit <model>` runs `llmfit plan` to estimate required VRAM/RAM for a model at a given context length. Returns a fit level:

| Level | Meaning |
|-------|---------|
| Perfect | Fully GPU-resident, recommended |
| Good | Minor headroom pressure, still fast |
| Marginal | CPU offload needed or tight fit |
| Too Tight | Won't run acceptably |

### 3. Launch string generation

`serve string <alias>` prints a copy-pasteable `llama-server` command:

```bash
llama-server \
  -m /path/to/Qwen3-8B.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -fa on \
  -c 65536 \
  --jinja \
  -t 32
```

The string is also written to `~/.local/share/turbofit/strings/<alias>.sh` so `serve <alias>` can re-execute it later.

### 4. Server lifecycle

`serve <alias>`:
1. Kills any existing server bound to the alias's port
2. Writes the launch string to a file
3. Spawns the server via `nohup bash -c "..."` (detached, survives shell death)
4. Waits up to 120s for `/health` to respond
5. Writes PID to `~/.local/share/turbofit/pid/<alias>`
6. Prints port, PID, and log path on success

Example output:

```
Launching qwen-8b (pid 12345) on :8080...
✓ qwen-8b ready

  Port:  :8080
  PID:   12345
  Logs:  /home/user/.local/share/turbofit/logs/qwen-8b.log
  Stop:  serve stop qwen-8b
```

`serve stop <alias>` kills the process by PID and cleans up state files.

`serve list` shows all running servers with their PIDs and ports.

### 5. Hermes-Agent config

The four "launch-and-run" commands:

| Command | What it does |
|---------|-------------|
| `serve main <alias>` | Launches server → sets Hermes-Agent main model → starts `hermes --tui` |
| `serve aux <alias>` | Launches server → sets Hermes-Agent aux model (all 9 tasks) → starts `hermes --tui` |
| `serve herm <alias>` | Launches server → sets main → starts `herm` (TUI) in background → starts `hermes --tui` |
| `serve herm aux <alias>` | Launches server → sets aux → starts `herm` in background → starts `hermes --tui` |

All four rewrite `~/.hermes/config.yaml` before launching Hermes, so the new config is picked up immediately:

```yaml
model:
  default: my-alias
  base_url: http://127.0.0.1:8080/v1
  api_key: not-needed
```

For `aux` variants, the `auxiliary:` section is rewired to point all 9 auxiliary tasks (vision, web_extract, compression, session_search, skills_hub, approval, mcp, title_generation, curator) at the same base_url + model name.

Append `--gateway` to `serve main` or `serve aux` to swap `hermes --tui` for `hermes gateway run`.

## Catalog format

Models are registered in `~/.config/turbofit/models.yaml`:

```yaml
models:
  qwen-8b:
    path: /home/user/models/Qwen3-8B.Q4_K_M.gguf
    port: 8080
  qwen-27b:
    path: /home/user/models/Qwen3-27B.Q4_K_M.gguf
    port: 8081
  hermes-main:
    path: /home/user/models/Darwin-Reason-28B.Q4_K_M.gguf
    port: 8082
```

Ports auto-increment (8080, 8081, 8082...) so multiple models can coexist.

## Optional Flags

Append any of these to `serve string <alias>` + manual edit, or pass them through the catalog YAML:

| Flag | Use case |
|------|----------|
| `--reasoning off` / `--reasoning-budget 0` | Disable reasoning for agentic/tool-use workloads |
| `--rope-scaling yarn --yarn-orig-ctx N` | Extend native context via YaRN (e.g. 8K → 64K) |
| `-ctk q8_0 -ctv q8_0` | Quantize KV cache (saves VRAM) |
| `--parallel N` | Number of concurrent slots |
| `--mlock` | Lock model in RAM (prevents swap) |
| `--no-mmap` | Disable memory-mapping (slower startup, faster inference) |
| `--no-warmup` | Skip model warmup (faster iteration during development) |
| `--batch-size 512 --ubatch-size 512` | Prompt processing batch sizing |

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

### 64K context VRAM budget

At 64K context with `q8_0` KV cache:

| Model size | 64K KV cache | Total VRAM needed |
|------------|--------------|-------------------|
| 8B Q4_K_M  | ~2 GB        | ~7 GB             |
| 13B Q4_K_M | ~2 GB        | ~10 GB            |
| 27B Q4_K_M | ~2 GB        | ~19 GB            |
| 70B Q4_K_M | ~2 GB        | ~42 GB            |

KV cache scales with context: 64K → ~2 GB, 128K → ~4 GB, 256K → ~8 GB, 1M → ~32 GB (per slot).

## API Integration

Point any OpenAI-compatible client at the running llama-server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Pitfalls

### 1. `flash_attn` is tri-state, not a bare flag
`--flash-attn` (bare) errors out. Use `-fa on` / `-fa off` / `-fa auto`.

### 2. `--gpu` is not a valid flag
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
Large models take time to load. If `/health` doesn't respond within 120s, `serve <alias>` reports failure — but the process might still be alive. Check `~/.local/share/turbofit/logs/<alias>.log` for the real status.

### 6. KV cache quantization trades quality for VRAM
`-ctk q4_0 -ctv q4_0` cuts KV cache size by ~50% vs q8_0, but degrades long-context recall. For 64K or less, q8_0 is the sweet spot.

### 7. Multiple servers on the same port
`serve <alias>` kills any existing process bound to the alias's port. If you registered two aliases to the same port manually, the second `serve` will kill the first.

### 8. CPU-only inference works but is slow
If you have no GPU, drop `-ngl 99` from the launch string manually. The model runs on CPU using RAM. Expect ~2-10 tok/s for 7B models.

### 9. `serve install` requires build tools
CMake, gcc/g++, CUDA toolkit (for GPU builds). If `cmake -B build` fails, install them first:
```bash
sudo apt install cmake build-essential    # Debian/Ubuntu
sudo dnf install cmake gcc-c++            # Fedora
brew install cmake                        # macOS (CPU only — CUDA needs NVIDIA SDK)
```

### 10. `serve herm` starts both herm and hermes
If `herm` isn't in PATH (e.g. you haven't run `bun add -g herm-tui`), `serve herm` silently skips it and starts hermes directly. Install herm for the full experience:
```bash
bun add -g herm-tui
```

## See Also

- `local-llm-fleet-management` — multi-model catalog + swap pattern
- `llama-cpp` — llama.cpp build + GGUF discovery
- `gguf-quantization` — GGUF format deep dive
- `herm` — the Herm TUI dashboard (optional companion for `serve herm`)