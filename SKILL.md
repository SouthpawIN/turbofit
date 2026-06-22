---
name: turbofit
description: "Generate optimized llama.cpp (llama-server) launch strings using llmfit hardware detection. Answers 'will this model fit in my VRAM/RAM?' and produces a copy-pasteable command with sensible defaults. Enforces a 64K context floor. No manager dependency — direct llama.cpp launch."
version: 2.0.0
author: SouthpawIN
license: MIT
tags: [llama.cpp, llama-server, llmfit, gguf, inference, hardware-fit]
metadata:
  hermes:
    related_skills: [local-llm-fleet-management, llama-cpp, gguf-quantization]
---

# Turbofit

Two questions, one workflow:

1. **Will this model fit in my system memory?** → `llmfit plan` / `llmfit fit`
2. **What's the optimal llama-server launch string for it?** → bridge script below

Turbofit bridges [`llmfit`](https://github.com/AlexsJones/llmfit) (hardware scan + fit analysis) to `llama-server` (the inference engine from [llama.cpp](https://github.com/ggerganov/llama.cpp)). It produces a copy-pasteable command line with sensible defaults — no manager, no sidecar, no orchestration layer.

## ⚠ Context Floor: 64K (65536 tokens)

Hermes-Agent requires at least 64K context. Every launch string this skill produces uses **`ctx_size: 65536` as the hard minimum**, regardless of what llmfit recommends. If a model can fit more (128K, 256K, 1M via YaRN), the script scales up — never down below 64K.

- Quick checks below 64K are **disallowed** in generated strings
- The bridge script clamps `CTX = max(llmfit_value, 65536)`
- VRAM headroom is budgeted assuming 64K minimum

## When to Use

- "Will this model fit on my machine?" — run `llmfit fit` or `llmfit plan`
- "Give me a llama-server command for this model" — bridge script
- "What's the largest model I can run with 64K context?" — `llmfit fit --perfect -n 5`
- "How much VRAM do I need for model X at context Y?" — `llmfit plan`
- Comparing what runs on different hardware before recommending a model
- Testing a new model's compatibility with the standard llama.cpp runtime

## Prerequisites

```bash
# llmfit (hardware detection + model recommendations)
which llmfit && llmfit --version    # should be >= 0.9.31
# Install: brew install AlexsJones/llmfit/llmfit
#      OR: curl -fsSL https://llmfit.axjns.dev/install.sh | sh
#      OR: uv tool install -U llmfit

# llama.cpp build (you need the llama-server binary)
which llama-server
# Build from source: https://github.com/ggerganov/llama.cpp#build
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

## Step 2: Build the llama-server Launch String

### Default Flags (applied automatically)

| Flag | Value | Purpose |
|------|-------|---------|
| `-ngl 99` | all GPU layers | Offload everything to GPU (fall back to `-ngl` int if no GPU) |
| `-fa on` | Flash attention | Critical for performance |
| `--jinja` | Jinja templates | Required for modern chat templates |
| `-c` | from llmfit (clamped to 65536) | Context window |
| `-t` | `nproc` | Thread count (llmfit detects CPU cores) |
| `--host`, `--port` | configurable | Bind address (default 127.0.0.1:8080) |

### Generate Command String (from llmfit output)

```bash
# After llmfit recommend, extract the top model + quant
MODEL_JSON=$(llmfit recommend --json --limit 1)
MODEL_NAME=$(echo "$MODEL_JSON" | jq -r '.models[0].name')
MODEL_QUANT=$(echo "$MODEL_JSON" | jq -r '.models[0].recommended_quantization')
MODEL_CTX=$(echo "$MODEL_JSON" | jq -r '.models[0].recommended_context')

# 64K floor — never go below this
MODEL_CTX=$(( MODEL_CTX < 65536 ? 65536 : MODEL_CTX ))

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

### TurboQuant-Specific (optional, requires TQ runtime)

If using TurboQuant GGUF files (TQ3_4S, TQ3_1S), the llama.cpp TQ fork is mandatory:

```bash
# TQ runtime: https://github.com/turbo-tan/llama.cpp-tq3
# Standard llama.cpp CANNOT load TQ GGUFs
# 64K floor — scale up via YaRN if VRAM allows

/path/to/llama.cpp-tq3/build/bin/llama-server \
  -m /path/to/Model-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 \
  -fa on -ctk tq3_0 -ctv tq3_0 \
  -c 65536 --jinja   # 64K minimum, scale to 262144+ if VRAM allows
```

## Step 3: Common Optional Flags

| Flag | Use case |
|------|----------|
| `--reasoning off` / `--reasoning-budget 0` | Disable reasoning for agentic/tool-use workloads |
| `--rope-scaling yarn --yarn-orig-ctx N` | Extend native context via YaRN (e.g. 8K → 64K) |
| `-ctk q8_0 -ctv q8_0` | Quantize KV cache (saves VRAM) |
| `--parallel N` | Number of concurrent slots |
| `--cont-batching` | Continuous batching (default in most builds) |
| `--mlock` | Lock model in RAM (prevents swap) |
| `--no-mmap` | Disable memory-mapping (slower startup, faster inference) |
| `--batch-size 512 --ubatch-size 512` | Prompt processing batch sizing |
| `-sm row --split-mode layer` | Multi-GPU split mode |

## Quick Bridge Script

One-liner pipeline: scan → pick top model → generate launch string.

```bash
# Get the top perfectly-fitting model as JSON
TOP=$(llmfit fit --perfect -n 1 --json 2>/dev/null)
NAME=$(echo "$TOP" | jq -r '.models[0].name // empty')
QUANT=$(echo "$TOP" | jq -r '.models[0].recommended_quantization // "Q4_K_M"')
# 64K context floor — clamp whatever llmfit recommends
CTX=$(echo "$TOP" | jq -r '.models[0].recommended_context // 65536')
CTX=$(( CTX < 65536 ? 65536 : CTX ))
TPS=$(echo "$TOP" | jq -r '.models[0].estimated_tps // "unknown"')

if [ -z "$NAME" ]; then
  echo "No perfectly-fitting model found. Try: llmfit fit --min-fit good -n 5 --cli"
  exit 1
fi

echo "# Model: $NAME ($QUANT, ctx=$CTX, est. $TPS tok/s)"
echo "# (64K context floor enforced)"
echo "#"
echo "# Raw llama-server command:"
echo "llama-server -m ${NAME}.${QUANT}.gguf --host 127.0.0.1 --port 8080 -ngl 99 -fa on -c ${CTX} --jinja -t $(nproc)"
```

### Pipe to clipboard for easy paste

```bash
# Linux
llmfit fit --perfect -n 1 --json | <bridge-script-above> | tail -1 | xclip -selection clipboard

# macOS
llmfit fit --perfect -n 1 --json | <bridge-script-above> | tail -1 | pbcopy
```

## API Integration

Point any OpenAI-compatible client at the running llama-server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",  # llama-server doesn't require auth by default
)

response = client.chat.completions.create(
    model="local-model",  # ignored by llama-server
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

The `OpenAI`-compatible client works for Hermes, LangChain, LlamaIndex, LiteLLM, and the official OpenAI SDK.

## Memory-Fit Reference

### Quick decision tree

| System | Best fit |
|--------|----------|
| 8 GB VRAM (RTX 3060/3070, M1/M2 base) | 7B-8B Q4_K_M, up to 13B Q3 |
| 12 GB VRAM (RTX 3060 12GB, 4070) | 13B Q4_K_M, 8B Q8_0 |
| 16 GB VRAM (RTX 4060 Ti 16, 4080) | 14B-15B Q4_K_M, 27B Q3 |
| 24 GB VRAM (RTX 3090/4090) | 27B Q4_K_M, 35B A3B TQ3_4S, up to 70B Q2 |
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
| 35B A3B TQ3_4S | ~2 GB    | ~15 GB            |
| 70B Q4_K_M | ~2 GB        | ~42 GB            |

KV cache scales with context: 64K → ~2 GB, 128K → ~4 GB, 256K → ~8 GB, 1M → ~32 GB (per slot).

## Pitfalls

### 1. TurboQuant GGUFs need the TQ fork
Stock llama.cpp will NOT load TQ3_4S/TQ3_1S files. Get `llama.cpp-tq3` from `https://github.com/turbo-tan/llama.cpp-tq3` if you want to use TurboQuant quants.

### 2. `flash_attn` is tri-state, not a bare flag
`--flash-attn` (bare) errors out. Use `-fa on` / `-fa off` / `-fa auto`.

### 3. `--gpu` is not a valid flag
Use `-ngl N` to control GPU layers (or `--device CUDA0` + `--main-gpu 0` for device selection). `--gpu` will error.

### 4. `--batch-size` vs `--ubatch-size`
- `--batch-size` = prompt processing batch (logical batch)
- `--ubatch-size` = micro-batch (actual physical batch on GPU)
- `ubatch-size <= batch-size` is required
- Both default to 512; tune down if you OOM

### 5. Context below 64K breaks Hermes-Agent (HARD FLOOR)
Hermes-Agent requires `context_length >= 65536`. A launch string with `-c 8192` will load fine but Hermes will crash the moment it tries to use its full system prompt + tool registry + history.

**Symptoms of violation:**
- Hermes fails to initialize (`context_length too small`)
- Tool registry truncates mid-load
- `InvalidRequestError` on the first multi-turn message
- The model "works" for trivial prompts but dies on real workloads

**Rule:** every string this skill produces uses `-c 65536` minimum. The bridge script clamps `CTX = max(llmfit_value, 65536)` before printing.

### 6. YaRN scaling beyond native context
If the model natively supports less than 64K, use `--rope-scaling yarn --yarn-orig-ctx <native>` to extend:
- Qwen3 256K native → use as-is up to 256K
- Older models 8K native → extend with YaRN to 64K+ (`--rope-scaling yarn --yarn-orig-ctx 8192 -c 65536`)
- 1M context → `--rope-scaling yarn --yarn-orig-ctx 262144 -c 1048576` (extreme, monitor VRAM)

### 7. `--no-warmup` for fast iteration
Default llama-server warms up the model (loads + runs dummy prompt) before serving. Add `--no-warmup` during development for ~10× faster restarts when iterating on flags.

### 8. KV cache quantization trades quality for VRAM
`-ctk q4_0 -ctv q4_0` cuts KV cache size by ~50% vs q8_0, but degrades long-context recall. For 64K or less, q8_0 is the sweet spot.

### 9. CPU-only inference is slow but works
If you have no GPU (or `--memory` override shows `unified_memory: false`), drop `-ngl 99` entirely. The model runs on CPU using RAM. Expect ~2-10 tok/s for 7B models, much slower for larger ones. Use `llmfit --memory=0G fit` to see CPU-only recommendations.

## See Also

- `local-llm-fleet-management` — fleet catalog + llama-launch pattern
- `llama-cpp` — llama.cpp build + GGUF discovery
- `gguf-quantization` — GGUF format deep dive
