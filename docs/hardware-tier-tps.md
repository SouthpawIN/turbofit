# Hardware-tier serving TPS

Live `llama-server /completion` medians. Three 128-token generations after warmup.
Contexts: **64K, 128K, 262K, 1M YaRN** (scale 4, orig 262144).
Host: dual RTX 3090 (`2x24`). Not smaller-card proof.
Unleashed artifacts only. Stock Qwen Q4_K_M is not used as a stand-in.
Updated: `2026-08-22T21:22:27.728816+00:00`

## Tier board

| Tier | Canonical model | 64K | 128K | 262K | 1M YaRN |
|---|---|---:|---:|---:|---:|
| 8 GB | Bonsai 27B Q1 | 58.75 | 58.13 | 58.48 | 9.41 |
| 8 GB MoE | Ornith 1.5 35A3B `--cpu-moe` mmap | 31.23 | 24.75 | 24.94 | 10.77 |
| 16 GB | Unleashed UD-IQ3_XXS | 41.35 | 40.76 | 40.71 | 8.62 |
| 24 GB | Unleashed UD-Q3_K_XL | 40.05 | 39.68 | 39.90 | 8.84 |
| 48 GB | Unleashed UD-Q3_K_XL | 40.05 | 39.68 | 39.90 | 8.84 |
| 64 GB | Unleashed UD-Q3_K_XL | 40.05 | 39.68 | 39.90 | 8.84 |
| 96 GB | Qwen 3.8 27B BF16 | — | — | — | — |
| 200 GB | Qwen 3.8 27B BF16 @ 1M | — | — | — | — |
| 300 GB | Qwen 3.8 27B BF16 @ 1M | — | — | — | — |
| aux | Carwin Nano `--cpu-moe` | 19.29 | 25.44 | 24.59 | 11.62 |

The 2.4 Fit List candidate for 8 GB is Maple Preview TQ2_0. The Bonsai row above is historical 2x24 serving evidence, not smaller-card proof and not Auto promotion. Ornith remains the 8 GB alternative when host RAM can hold offloaded experts. Never a 9B.

Raw: [`references/results/hardware-tier-tps-contexts-2x24.json`](../references/results/hardware-tier-tps-contexts-2x24.json)

