# TurboFit Check / TurboFit List

Canonical product brief. This is what ships, not a research note.

## TurboFit List

The canonical models in the TurboFit Check chain.

## TurboFit Check

Scans the user's hardware, then recommends and configures Hermes-Agent main and auxiliary models for that device.

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

## Backend

TurboHaul Manager is the preferred backend. Other inference engines are fallbacks only when they are measurably better for that machine.

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
- The entire fallback ladder, explained
- Multimodal, explained
