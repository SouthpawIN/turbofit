# Hardware-tier serving TPS

Measured with live `llama-server /completion` timings. Medians of three 128-token generations after one warmup. No fabricated scores.

- Host topology: **2x24 GB NVIDIA GeForce RTX 3090** (`2x24`, 48 GB total), driver from `nvidia-smi`, 386861 MiB RAM.
- This is **not** physical proof of an 8 GB, 16 GB, or 96 GB card. Larger hardware is not smaller-tier evidence.
- GPU0 was isolated (`CUDA_VISIBLE_DEVICES=0`). GPU1 was left to the in-flight MiniMax H3 job.
- Unleashed `UD-IQ3_XXS` / `UD-Q3_K_XL` GGUFs were **not on disk**; 16 GB and 24-95 GB bands used the available Qwen 3.8 27B Q4_K_M stand-in and are labeled as such.
- Captured: `2026-08-22T18:08:14.138242+00:00`

| Band | Model | Serving recipe | Decode tok/s median | Prefill tok/s median | Peak GPU0 MiB | Notes |
|---|---|---|---:|---:|---:|---|
| 8GB-class model | `bonsai-27b-q1` | `ceiling-ngl99-fa-b2048` | 63.82 | 65.87 | 4781 | live /completion timings |
| 8GB-class model | `bonsai-27b-q1` | `fit-on-fa-b2048` | 61.77 | 64.63 | 4781 | live /completion timings |
| 24-95GB-class model | `qwen3.8-27b-q4-k-m` | `ceiling-ngl99-fa-b2048` | 37.67 | 58.71 | 18737 | live /completion timings |
| 24-95GB-class model | `qwen3.8-27b-q4-k-m` | `fit-on-fa-b2048` | 37.44 | 58.79 | 18737 | live /completion timings |
| aux | `carwin-moe-nano` | `ceiling-ngl99-fa-b2048` | 132.46 | 137.31 | 11535 | live /completion timings |
| aux | `carwin-moe-nano` | `fit-on-fa-b2048` | 132.34 | 137.41 | 11535 | live /completion timings |

## Best measured decode on this host

- **8GB-class model**: `bonsai-27b-q1` / `ceiling-ngl99-fa-b2048` = **63.82 tok/s**
- **24-95GB-class model**: `qwen3.8-27b-q4-k-m` / `ceiling-ngl99-fa-b2048` = **37.67 tok/s**
- **aux**: `carwin-moe-nano` / `ceiling-ngl99-fa-b2048` = **132.46 tok/s**

Raw evidence: [`references/results/hardware-tier-tps-2x24-20260822.json`](references/results/hardware-tier-tps-2x24-20260822.json)

