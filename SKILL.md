---
name: llmfit-turbohaul
description: "Generate optimized llama-server launch strings and turbohaul-manager manifests using llmfit hardware detection + TurboQuant flag doctrine. Use when setting up a new model in turbohaul, when someone asks 'what fits on my GPU', or when building a llama.cpp command from scratch."
version: 1.0.0
author: SouthpawIN
license: MIT
tags: [turbohaul, llmfit, llama.cpp, turboquant, gguf, inference, manifest]
metadata:
  hermes:
    related_skills: [local-llm-fleet-management, omni-va-local-server, llama-cpp, gguf-quantization]
---

# Turbohaul + llmfit

Two tools, one workflow: **llmfit** detects hardware and recommends models that fit → **turbohaul-manager** manages the inference sidecar with FIFO queuing, grace windows, and model hot-swap → the bridge generates an optimized `llama-server` command string or turbohaul manifest YAML.

## When to Use

- Setting up a new model in turbohaul-manager and need the right flags
- Someone asks "what model fits on my GPU" → run llmfit, then generate a launch string
- Building a turbohaul manifest from scratch for a model found via llmfit
- Testing a friend's inference engine and need a baseline launch string
- Comparing what runs on different hardware before recommending models

## Prerequisites

```bash
# llmfit (hardware detection + model recommendations)
which llmfit && llmfit --version    # should be >= 0.9.31
# Install: brew install AlexsJones/llmfit/llmfit  OR  curl -fsSL https://llmfit.axjns.dev/install.sh | sh

# turbohaul-manager (inference manager)
which turbohaul-manager             # or: docker ps | grep turbohaul
# Docker: ghcr.io/MrTrenchTrucker/turbohaul-manager:v0.2.3
```

## Step 1: Detect Hardware with llmfit

```bash
# Full system scan
llmfit system --json

# Top models that fit perfectly on this GPU
llmfit fit --perfect -n 10 --cli

# Top models ranked by overall score (JSON for scripting)
llmfit recommend --json --limit 5

# Hardware planning: "what do I need to run X?"
llmfit plan "Qwen/Qwen3-4B" --context 8192

# Simulate different hardware
llmfit --memory=24G --ram=64G --cpu-cores=8 fit --cli -n 5
```

**llmfit fit levels:**
| Level | Meaning |
|-------|---------|
| Perfect | Fully GPU-resident, recommended |
| Good | Minor headroom pressure, still fast |
| Marginal | CPU offload needed or tight fit |
| Too Tight | Won't run acceptably |

## Step 2: Build the llama.cpp Launch String

### TurboQuant Flag Doctrine (5 flags, enabled by default)

These are **spawn argv** arguments for turbohaul manifests. From turbohaul-manager's manifest.py (`SAFE_LLAMA_FLAGS` allowlist):

| Flag | Value | Purpose |
|------|-------|---------|
| `flash_attn` | `on` | Flash attention (critical for perf) |
| `no_context_shift` | `true` | Prevents context auto-shifting |
| `cache_reuse` | `256` | KV cache token reuse threshold |
| `slot_prompt_similarity` | `0.5` | Warm-slot match threshold |
| `no_perf` | `true` | Disables perf metrics overhead |

### Common Additional Flags

| Flag | Notes |
|------|-------|
| `-ngl 99` | All GPU layers (or `-ngl all`) |
| `-fa on` | Flash attention (alias for `flash_attn`) |
| `-ctk tq3_0` / `-ctv tq3_0` | TurboQuant KV cache (TQ fork only!) |
| `-c` | Context size from llmfit recommendation |
| `--jinja` | Enable Jinja chat templates |
| `--reasoning off` | Disable reasoning budget for agentic use |
| `--host 0.0.0.0 --port PORT` | Bind address |
| `-t N` | Thread count (llmfit detects CPU cores) |

### Generate Command String (from llmfit output)

```bash
# After llmfit recommend, extract the top model + quant
MODEL_JSON=$(llmfit recommend --json --limit 1)
MODEL_NAME=$(echo "$MODEL_JSON" | jq -r '.models[0].name')
MODEL_QUANT=$(echo "$MODEL_JSON" | jq -r '.models[0].recommended_quantization')
MODEL_CTX=$(echo "$MODEL_JSON" | jq -r '.models[0].recommended_context')

# Build the llama-server command
echo "llama-server \\"
echo "  -m /path/to/${MODEL_NAME}.${MODEL_QUANT}.gguf \\"
echo "  --host 127.0.0.1 --port 8080 \\"
echo "  -ngl 99 \\"
echo "  -fa on \\"
echo "  -c ${MODEL_CTX} \\"
echo "  --jinja \\"
echo "  -t $(nproc)"
```

### TurboQuant-Specific (TQ fork required)

If using TurboQuant GGUF files (TQ3_4S, TQ3_1S quantizations), the runtime fork is mandatory:

```bash
# TQ runtime: https://github.com/turbo-tan/llama.cpp-tq3
# Standard llama.cpp CANNOT load TQ GGUFs

CUDA_VISIBLE_DEVICES=0 /path/to/llama.cpp-tq3/build/bin/llama-server \
  -m /path/to/Model-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 \
  -fa on -ctk tq3_0 -ctv tq3_0 \
  -c 262144 --jinja
```

## Step 3: Generate a Turbohaul Manifest

Turbohaul manifests are YAML files registered via `PUT /api/manifests/{tag}`. The `llama_flags` dict maps to llama-server argv.

### Minimal Manifest

```yaml
tag: my-model:latest
model_path: /var/lib/turbohaul/blobs/sha256:<hash>
llama_flags:
  ctx_size: 8192
  n_gpu_layers: 99
  threads: 32
  batch_size: 512
  ubatch_size: 512
  flash_attn: "on"
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
keep_alive_default: 600
grace_seconds: 30
idle_hot_load_seconds: 600
parallel: 1
```

### TurboQuant Manifest (TQ3 GGUF)

```yaml
tag: qwen3-tq:latest
model_path: /var/lib/turbohaul/blobs/sha256:<hash>
llama_flags:
  ctx_size: 131072
  n_gpu_layers: 99
  threads: 32
  batch_size: 512
  ubatch_size: 512
  flash_attn: "on"
  cache_type_k: "tq3_0"
  cache_type_v: "tq3_0"
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
keep_alive_default: 600
grace_seconds: 30
idle_hot_load_seconds: 600
parallel: 1
```

### MoE Model Manifest (35B A3B style)

```yaml
tag: moe-35b:latest
model_path: /var/lib/turbohaul/blobs/sha256:<hash>
llama_flags:
  ctx_size: 262144
  n_gpu_layers: 99
  threads: 32
  batch_size: 512
  ubatch_size: 512
  flash_attn: "on"
  cache_type_k: "tq3_0"
  cache_type_v: "tq3_0"
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
  cpu_moe: true
  reasoning: "off"
  reasoning_budget: 0
keep_alive_default: 600
grace_seconds: 30
idle_hot_load_seconds: 600
parallel: 1
```

## Step 4: Register in Turbohaul

```bash
# Pull a model from HuggingFace into turbohaul's blob store
curl -X POST http://localhost:11401/api/pull-hf \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "YTan2000/Qwen3.6-35B-A3B-TQ3_4S", "filename": "*.gguf"}'

# Import a local GGUF file
curl -X POST http://localhost:11401/api/import \
  -F "file=@/path/to/model.gguf" \
  -F "tag=my-model:latest"

# Register/update a manifest (ETag for concurrency)
curl -X PUT http://localhost:11401/api/manifests/my-model:latest \
  -H "Content-Type: application/yaml" \
  -d @manifest.yaml

# List registered models
curl -s http://localhost:11401/api/tags | jq '.models[].name'

# Check live status
curl -s http://localhost:11401/status | jq
```

## llmfit → Turbohaul Quick Bridge Script

One-liner pipeline: scan → pick top model → generate launch string.

```bash
# Get the top perfectly-fitting model as JSON
TOP=$(llmfit fit --perfect -n 1 --json 2>/dev/null)
NAME=$(echo "$TOP" | jq -r '.models[0].name // empty')
QUANT=$(echo "$TOP" | jq -r '.models[0].recommended_quantization // "Q4_K_M"')
CTX=$(echo "$TOP" | jq -r '.models[0].recommended_context // 8192')
TPS=$(echo "$TOP" | jq -r '.models[0].estimated_tps // "unknown"')

if [ -z "$NAME" ]; then
  echo "No perfectly-fitting model found. Try: llmfit fit --min-fit good -n 5 --cli"
  exit 1
fi

echo "# Model: $NAME ($QUANT, ctx=$CTX, est. $TPS tok/s)"
echo "#"
echo "# Raw llama-server command:"
echo "llama-server -m ${NAME}.${QUANT}.gguf --host 0.0.0.0 --port 8080 -ngl 99 -fa on -c ${CTX} --jinja -t $(nproc)"
echo "#"
echo "# Turbohaul manifest (put to /api/manifests/${NAME,,}:latest):"
cat <<EOF
tag: ${NAME,,}:latest
llama_flags:
  ctx_size: ${CTX}
  n_gpu_layers: 99
  threads: $(nproc)
  flash_attn: "on"
  no_context_shift: true
  cache_reuse: 256
  slot_prompt_similarity: 0.5
  no_perf: true
  jinja: true
  cache_type_k: "q8_0"
  cache_type_v: "q8_0"
keep_alive_default: 600
grace_seconds: 30
parallel: 1
EOF
```

## API Integration with AI Agents

Point Hermes, LangChain, or any OpenAI-compatible client at turbohaul:

```yaml
# Hermes config.yaml
model:
  base_url: http://localhost:11401/v1
  api_key: dummy
  default: my-model:latest
```

Turbohaul's FIFO queue + grace windows handle multi-agent concurrency automatically. Same-model requests from the same `thread_id` get sub-second warm-slot reuse (ACTIVE_MATCH cascade).

## Turbohaul Flag Security Notes

Manifest flags go through a closed allowlist (`SAFE_LLAMA_FLAGS`, ~80 flags). Blocked categories:

- **Path/file injection**: `mmproj`, `lora*`, `grammar_file`, `log_file`, `model_draft`
- **SSRF/Network**: `model_url`, `hf_repo*`, `docker_repo`
- **Credentials**: `hf_token`, `api_key`, `ssl_*`
- **RCE**: `tools` (enables shell exec), `override_kv`, `binary_override`

Suffix guard catches future path/credential flags: anything ending in `_file`, `_path`, `_dir`, `_url`, `_repo`, `_key` is rejected unless explicitly allowlisted.

## Pitfalls

### 1. TurboQuant GGUFs need the TQ fork
Stock llama.cpp will NOT load TQ3_4S/TQ3_1S files. Get `llama.cpp-tq3` from `https://github.com/turbo-tan/llama.cpp-tq3`. Also applies to turbohaul — it supervises a llama-server subprocess, so that subprocess must be the TQ fork.

### 2. Manifest changes don't take effect on running sidecars
Updating a manifest via PUT does NOT restart the running `llama-server`. Force a cold-spawn:
- Set `keep_alive: 0` on the next request
- Wait for natural idle-hot teardown (default 600s)
- Restart the turbohaul container

### 3. `flash_attn` is tri-state, not a bare flag
Accepts `"on"`, `"off"`, `"auto"`, `true`, `false`. A bare `--flash-attn` on the command line errors out. In YAML manifests, use `flash_attn: "on"`.

### 4. Parallel > 1 requires `kv_unified: true`
If you set `parallel: 2` (or higher) in a manifest, `kv_unified: true` is mandatory. Also `ctx_size` must be evenly divisible by `parallel`, and per-slot context floor is 8192.

### 5. `n_gpu_layers` accepts int OR `"all"`
Both `-ngl 99` and `-ngl all` / `n_gpu_layers: "all"` work. The allowlist validates both forms.

### 6. State mount is required for Docker
Always bind-mount `-v $(pwd)/state:/var/lib/turbohaul`. Without it, manifests, state.sqlite, and blob store live in the container layer and die on `docker rm`.

## See Also

- `local-llm-fleet-management` — fleet catalog + llama-launch pattern
- `omni-va-local-server` — always-on VA server with VRAM cascade
- `hermes-agent` / `references/turboquant-setup.md` — TurboQuant model sizes, YaRN 1M scaling
- `gguf-quantization` — GGUF format deep dive
