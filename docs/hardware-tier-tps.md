# Hardware-tier serving TPS

Live `llama-server /completion` medians. Three 128-token generations after warmup.
Contexts are Turbofit limits: **64K, 128K, 262K, 1M YaRN** (scale 4, orig 262144).
Host: dual RTX 3090 (`2x24`). Not smaller-card proof.
Unleashed artifacts: pinned `UD-Q3_K_XL` and `UD-IQ3_XXS`. Stock Qwen Q4_K_M is not used.
Captured: `2026-08-22T19:30:25.572825+00:00`

| Band | Model | Context | Decode tok/s | Prefill tok/s | GPU used MiB | Notes |
|---|---|---:|---:|---:|---|---|
| 8GB | `bonsai-27b-q1` | 65536 | 58.75 | 40.29 | 5687+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 8GB | `bonsai-27b-q1` | 131072 | 58.13 | 40.59 | 7159+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 8GB | `bonsai-27b-q1` | 262144 | 58.48 | 39.82 | 10103+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 8GB | `bonsai-27b-q1` | 1048576 | 9.41 | 18.86 | 8579+8487 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 16GB | `unleashed-ud-iq3-xxs` | 65536 | 41.35 | 37.85 | 11965+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 16GB | `unleashed-ud-iq3-xxs` | 131072 | 40.76 | 38.38 | 13437+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 16GB | `unleashed-ud-iq3-xxs` | 262144 | 40.71 | 37.93 | 16381+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 16GB | `unleashed-ud-iq3-xxs` | 1048576 | 8.62 | 18.55 | 11061+12283 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 24-95GB | `unleashed-ud-q3-k-xl` | 65536 | 40.05 | 35.18 | 13989+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 24-95GB | `unleashed-ud-q3-k-xl` | 131072 | 39.68 | 38.17 | 15461+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 24-95GB | `unleashed-ud-q3-k-xl` | 262144 | 39.90 | 37.91 | 18405+68 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |
| 24-95GB | `unleashed-ud-q3-k-xl` | 1048576 | 8.84 | 18.45 | 12203+13167 | live /completion; Unleashed Q3/IQ3; YaRN only at 1M |

## Best decode per band / context

- **8GB @ 64K**: Bonsai Q1 = **58.75 tok/s**
- **8GB @ 128K**: Bonsai Q1 = **58.13 tok/s**
- **8GB @ 262K**: Bonsai Q1 = **58.48 tok/s**
- **8GB @ 1M YaRN**: Bonsai Q1 = **9.41 tok/s**
- **16GB @ 64K**: Unleashed UD-IQ3_XXS = **41.35 tok/s**
- **16GB @ 128K**: Unleashed UD-IQ3_XXS = **40.76 tok/s**
- **16GB @ 262K**: Unleashed UD-IQ3_XXS = **40.71 tok/s**
- **16GB @ 1M YaRN**: Unleashed UD-IQ3_XXS = **8.62 tok/s**
- **24-95GB @ 64K**: Unleashed UD-Q3_K_XL = **40.05 tok/s**
- **24-95GB @ 128K**: Unleashed UD-Q3_K_XL = **39.68 tok/s**
- **24-95GB @ 262K**: Unleashed UD-Q3_K_XL = **39.90 tok/s**
- **24-95GB @ 1M YaRN**: Unleashed UD-Q3_K_XL = **8.84 tok/s**

Carwin aux 64K–1M still pending. Serve flags baked into the Unleashed recipe: `-ngl 99 -fa on -b 2048 -ub 512`, YaRN scale 4 at 1M.

Raw: [`references/results/hardware-tier-tps-contexts-2x24.json`](references/results/hardware-tier-tps-contexts-2x24.json)

