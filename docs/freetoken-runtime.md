# FreeToken candidate runtime

TurboFit integrates [FreeToken](https://github.com/FlashML-org/FreeToken) as an **optional, evidence-gated candidate backend** for supported NVIDIA MoE checkpoints. It is not an Auto rung and does not change the active Qwen 3.8 Unleashed / Ornith / Bonsai model authority.

## Why it is worth integrating

FreeToken's bandwidth-adaptive MoE execution directly addresses a real TurboFit bottleneck: sparse experts that exceed VRAM but fit in host memory. It overlaps full-layer prefill transfers, caches routed experts in VRAM, calibrates CPU-vs-PCIe miss handling with `ft bench bw`, and resizes expert/KV pools without restarting. Its server already exposes OpenAI and Anthropic APIs, tool calling, reasoning parsing, `/health`, `/v1/models`, and `/v1/stats`.

The [FreeToken paper](https://arxiv.org/abs/2608.16157) reports 39.3 tok/s for Qwen3.6-35B-A3B NVFP4 on an RTX 4060 Laptop GPU with 8 GB VRAM **and 32 GiB LPDDR5 host memory**. The RTX 5090 results used systems with 180–192 GiB host RAM. These are model/runtime/hardware-specific measurements, not portable TurboFit scores.

## Hard boundaries

Pinned candidate:

- repository: `https://github.com/FlashML-org/FreeToken.git`
- revision: `0ab982f10905fa775962a4eddcb44caa50065251`
- version: `0.1.2`
- license: Apache-2.0

Current runtime requirements from FreeToken's own install contract:

- Linux x86_64;
- NVIDIA RTX CUDA GPU;
- NVIDIA driver 580 or newer;
- CUDA toolkit 13 or newer with matching `nvcc`;
- HF safetensors or converted FTW checkpoint;
- text-only serving (FreeToken currently drops multimodal input);
- a FreeToken-supported MoE architecture.

TurboFit's active Qwen 3.8 Unleashed and Ornith artifacts are not listed as supported FreeToken checkpoints at this revision. FreeToken therefore cannot replace the active native runtime today. It is integrated now so future supported MoE candidates can enter the normal artifact, context, tool-call, TPS, intelligence, pressure, and rollback campaigns without an ad-hoc side path.

## Install or verify

```bash
scripts/install-freetoken-runtime --check-only --json
scripts/install-freetoken-runtime --json
```

The installer refuses unsupported hosts before cloning or installing anything. A successful install remains `verified-candidate`, never `validated` or Auto-promoted.

## Candidate launch contract

TurboFit builds a loopback-only command equivalent to:

```bash
ft serve \
  --model <hf-or-ftw-path> \
  --host 127.0.0.1 \
  --port <owned-port> \
  --served-model-name <portable-alias> \
  --max-seq-len-override <context> \
  --memory-ratio <safe-ratio> \
  --moe-backend auto
```

Before promotion on any exact machine/model pair:

1. run `ft bench bw` on that machine;
2. capture exact model revision and artifact hashes;
3. pass normal and Jinja/Hermes tool-call requests at the required context ladder;
4. record TTFT, decode TPS, host RAM, per-device VRAM, PCIe topology, and cleanup;
5. prove owned-process drain, pressure contraction, recovery, and rollback;
6. run the standard TurboFit intelligence campaign;
7. only then add a candidate recipe and consider tier promotion.
