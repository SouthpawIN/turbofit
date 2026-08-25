# TurboFit Check and TurboFit List

Canonical product brief. This is what ships, not a research note.

## TurboFit List

The evidence-only winners at each physical hardware level. A level remains blank until its current-recipe physical, throughput, and intelligence campaigns produce a hash-bound winner. Canonical generated view: [`turbofit-list.md`](turbofit-list.md).

## TurboFit Check

The system scan-to-configuration process. It scans the user's hardware, compares it with exact-tier winners and portable-fit candidates, then recommends and configures Hermes-Agent main and auxiliary models for that device.

Low-VRAM dedicated Auto is **Maple Preview TQ2_0** at 64K/128K. Ornith 1.5 remains the host-RAM-dependent alternative. Apple Silicon uses **OrcaRouter Qwen3.8-27B-Uncensored MLX** (4/6/8-bit, never 2-bit) or official Maple MLX on `mlx-lm-deepgrove`. Integrated / unified (non-Apple) uses Maple at 8–15 GB and Ornith at 16–23 GB.

Must account for:

- dedicated memory
- integrated / unified memory
- CPU-only
- Linux, Windows, macOS
- NVIDIA CUDA, AMD ROCm, Vulkan, Metal

## Benchmarks drive recommendations

Models are ranked from real scores, not guesses:

- tok/s at 64K, 128K, 262K, and 1M YaRN
- DeepSWE
- TerminalBench v2.1
- LiveCodeBench
- MMLU
- Tau Banking, Food Truck
- other agentic, browser, coding, reasoning, and financial benches

Those scores are posted to the TurboFit GitHub. A user's local TurboFit can follow that live ladder.

## Engine audition

Check inventories and auditions these engines against the selected model pair:

| Engine | Maple TQ2_0 GGUF | Qwen 3.8 GGUF / HF | Notes |
|---|---|---|---|
| llama.cpp | Maple fork only | mainline | Cross-platform |
| Turbohaul Manager | only if it manages the Maple fork | preferred manager | Linux NVIDIA |
| MLX | `deepgrove/maple-preview-2bit-mlx` | OrcaRouter Uncensored 4/6/8-bit | Apple Silicon |
| SGLang | no | HF / FP8 recipes | Linux/WSL |
| vLLM | no | HF / FP8 / NVFP4 | Linux/WSL or vLLM-Metal |
| FreeToken | no | HF MoE recipes only | Linux x86_64 CUDA 13 / r580+ |

Canonical matrix: [`references/engine-serve-matrix.json`](../references/engine-serve-matrix.json). Desktop: **Audition engines**.


## Live fallback chain

The chain adapts on the fly to computer use:

1. disable aux (route aux work to main)
2. lower context
3. swap to a lower quant / model

When memory returns, it heals back up to the recommended configuration.

## Multimodal

TurboFit Check also recommends:

- Image: MiniMax H3 or LTX, single frame
- Video: MiniMax H3 or LTX
- Music: ACE-Step 1.5 (multiple sizes) and MiniMax Music 3
- STT/TTS/STS: Nemotron ASR, Parakeet, Soprano, Darwin TTS, KittenTTS
- plus the default Hermes-Agent multimodal pipelines

## Promo points

- Uncensored Qwen 3.8 27B Unleashed, 262K context, single 24 GB GPU
- Apple: OrcaRouter Uncensored MLX 4/6/8-bit
- The entire fallback ladder, explained
- Multimodal, explained

## Not Auto yet

`EschaLabs/Qwen3.8-27B-Escha-W2` (Asha mixed 2-bit, custom SGLang) is a research candidate only. One reviewer video is not TurboFit TPS. It does not replace Bonsai/Ornith at the bottom of the list until we have a pinned llama.cpp or TurboHaul recipe and 64K/128K/262K/1M numbers. OrcaRouter MLX 2-bit is banned (uploader: archival / quality collapse).
