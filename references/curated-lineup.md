# Curated Model Lineup

Opinionated picks for main + aux roles, local + API. **Good intelligence is always available no matter what the user loads onto their GPUs.**

## Main Local Models

### S-tier: Top Picks

#### `darwin-28b-reason` ⭐ Top Local Pick
- **Why:** Smartest 27B dense (Darwin MRI-Trust merge, GPQA 89.39%), supports 1M context
- **VRAM:** ~22 GB
- **Spec decoding:** turbo4-kv, no speculative decoding
- **tok/s target:** 38
- **Context:** 1M (262K native + KV cache extension)

#### `prism-eagle-27b` 🚀 Speed/Size Pick
- **Why:** Fastest 27B (121 tok/s draft-mtp), hybrid Mamba-2+GDN architecture, smallest 27B-class
- **VRAM:** ~14 GB
- **Spec decoding:** draft-mtp + turbo4-kv
- **tok/s target:** 121
- **Context:** 256K

### SF-tier: Needs Benchmarking (all at Q4 quantization)

Three models compete here. Run `turbofit benchmark compare_27b` to rank:

#### `qwopus-27b-v2-mtp`
- **Why:** Qwopus 27B — needs benchmarking vs Qwable/Carwin/Darwin
- **VRAM:** ~17 GB
- **Spec decoding:** nextn + vision-mmproj
- **tok/s target:** 100
- **Context:** 256K

#### `qwable-27b-mtp`
- **Why:** Qwable 27B — needs benchmarking vs Qwopus/Carwin/Darwin
- **VRAM:** ~17 GB
- **Spec decoding:** nextn
- **tok/s target:** 100
- **Context:** 256K

#### `carwin-28b-mtp`
- **Why:** Carwin 28B — needs benchmarking vs Qwopus/Qwable/Darwin
- **VRAM:** ~17 GB
- **Spec decoding:** nextn + vision-mmproj
- **tok/s target:** 100
- **Context:** 256K

#### `darwin-apex-36b-i-compact`
- **Why:** Darwin Apex 36B Opus APEX (36B/3B-active MoE) — needs benchmarking vs Qwopus/Qwable/Carwin
- **VRAM:** ~17 GB
- **Spec decoding:** nextn + turbo4-kv
- **tok/s target:** 107
- **Context:** 256K

### API Fallback (main_api)

When local GPU is full or down:
- **deepseek-v4-pro** (OpenRouter) — Best reasoning + coding
- **glm-5.1** (OpenRouter) — Agentic + long-horizon reasoning
- **nemotron-ultra** (NVIDIA NIM) — 550B MoE, vision, free tier
- **kimi-k2.5** (OpenRouter) — Strong coding benchmarks
- **qwen-3.7-max** (OpenRouter) — Qwen frontier

## Auxiliary Local

### `carnice-apex-35a3b-compact` ⭐ Aux Choice in All Cases
- **Why:** Always-on 35A3B, Hermes data, vision, MTP, 1M ctx capable
- **VRAM:** ~17 GB
- **Spec decoding:** nextn + turbo4-kv
- **tok/s target:** 30
- **Context:** 1M

**Note:** Carnice 35A3B is the auxiliary choice in all cases. It serves as the always-on secondary model for tasks like Hermes agent support.

## Auxiliary API

When local aux is unavailable:
- **deepseek-v4-pro** (OpenRouter) — Best reasoning
- **glm-5.1** (OpenRouter) — Agentic reasoning
- **minimax-m3** (NVIDIA NIM) — Multimodal support
- **nemotron-ultra** (NVIDIA NIM) — 550B MoE

## Opionionated Polite VRAM Scaling Ladder

The system automatically adapts to VRAM pressure while preserving intelligence:

1. **Ideal** — Darwin 28B full @ 1M + Carnice 35A3B @ 1M auxiliary
2. **Mild pressure** — Offload MoE experts from Carnice to CPU
3. **Moderate pressure** — Drop context size of both (1M → 128K)
4. **High pressure** — Drop Carnice entirely (aux set to auto)
5. **Critical** — Swap to Prism Eagle 27B as Hermes model
6. **Extreme** — Swap to Darwin Apex 35A3B MoE as Hermes model
7. **Survival** — Offload experts of Darwin Apex

Each step down preserves maximum available compute before dropping.

See `scaling-ladder.md` for detailed step-by-step behavior and implementation.

## Benchmark Group: `compare_27b`

The following models are in the `benchmark_group: compare_27b` and should be benchmarked head-to-head at Q4 quantization:

- `qwopus-27b-v2-mtp`
- `qwable-27b-mtp`
- `carwin-28b-mtp`
- `darwin-apex-36b-i-compact`

Run: `turbofit benchmark compare_27b` to rank by tok/s and quality.
