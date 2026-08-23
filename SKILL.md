---
name: turbofit
description: "Operate the Turbofit adaptive local inference plugin."
version: 2.3.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, plugin, llama-cpp, adaptive-runtime]
---

# Turbofit

## 2.3 model authority

Main Auto chain: Bonsai 27B Q1 for the low-memory lane, Qwen 3.8 27B Unleashed UD-IQ3_XXS (16GB), UD-Q3_K_XL (24–95GB), and Qwen 3.8 27B 16-bit (96GB+) until an Unleashed FP16 GGUF is published. Below the proven 8GB tier, setup may expose Bonsai shared-main as a portable-fit lane with safe host spill, but it remains benchmark-required and cannot become Auto on that box until on-box validation passes. Auxiliary is Ornith 1.5 35A3B, optional Carwin Nano, or auto. API fallback is the five keyless Nous free models. New catalog entries remain candidates until current-recipe physical evidence promotes them.

Use this bundled plugin skill when configuring or inspecting Turbofit for Hermes Agent.

## Operator workflow

1. Call `turbofit_status` to inspect provider registration, gateway health, selected hardware profile, active rung, and stable routes.
2. Call `turbofit_configure` with `profile: auto` for hardware selection. Manual `hardware-*gb` profiles are accepted only when physical topology fits.
3. Set `primary: true` to use `custom:turbofit` with model `auto` as the main Hermes provider.
4. Set `fallback: true` to append Turbofit to the canonical `fallback_providers` chain; set it false to remove only Turbofit while preserving other fallbacks.
5. Set `publish_tailnet: true` to create private Tailscale Serve routes for the provider and dashboard; the returned HTTPS provider URL is registered automatically.
6. Set `install_sirvir: true` to install or update the canonical `SouthpawIN/sirvir` GitHub profile without replacing its memories or user state.
7. Set `install_freetoken: true` only on Linux x86_64 + NVIDIA driver 580+ + CUDA toolkit 13+ to install pinned FreeToken 0.1.2 as a text-only MoE **candidate**. It never changes Auto until exact on-box campaigns promote a supported model recipe.
8. Start a new Hermes session after provider changes.

The same controls are available in `hermes dashboard` under **Turbofit** and through `/turbofit status|tiers|setup`.

## Intelligence benchmarks

- `scripts/turbofit-catalog-campaign` proves native runtime fit and TPS; it does not produce intelligence scores.
- `scripts/turbofit-intelligence-campaign` runs the exact successful quantized production recipe through pinned DeepSWE and the Turbofit agentic main/auxiliary pair harness.
- Use `status`, `run-one`, or `run --limit N`; state is resumable in `references/intelligence-campaign-state.json`.
- Scores require both benchmark suites and immutable raw evidence. Never replace missing scores with catalog tiers, parameter counts, or vendor benchmark claims.
- `/turbofit tiers` and `scripts/turbofit-hardware-tiers` show every 8/16/24/48/64/96/200/300 GB class with pending versus measured intelligence and TPS.
- `scripts/turbofit-intelligence-campaign` benchmarks only the current machine's TurboFit List tournament candidates. `rebuild-scores` recomputes derived composites from raw suite counts; zero-call/token trials remain invalid infrastructure.
- `scripts/turbofit-promote-list-winner` promotes only an exact-tier candidate with current physical evidence, positive intelligence/TPS/balanced values, and matching recipe hashes. `scripts/turbofit-list` renders the global evidence-only List.
- Qwen 3.8 DFlash2 is a separate candidate runtime/artifact pair (`dflash2-llama.cpp`, `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`). Never attach that drafter to Bonsai. Bonsai uses its own released DSpark sidecar and Prism runtime until a dedicated Bonsai DFlash checkpoint exists.

## Portable memory allocation

- Hardware fingerprints classify memory as `dedicated`, `unified`, or `cpu` and reserve 5% of host RAM, bounded to 1–8 GiB.
- Dedicated systems may combine accelerator VRAM with host RAM through llama.cpp offload; contexts beyond the model's native window place KV cache in host RAM when at least 32 GiB is usable.
- Unified-memory systems count RAM once and suppress discrete multi-GPU split flags.
- CPU-only systems set model and draft GPU layers to zero.
- Native backend order is CUDA, ROCm, Vulkan, then CPU on Linux/Windows, and Metal on macOS. Use `scripts/install-native-runtimes --backend <backend>` for an explicit build.

## Invariants

- Stable model IDs are `auto`, `active:main`, and `active:aux`.
- FreeToken support is candidate-only: no active Qwen 3.8/Ornith replacement, no source TPS inheritance, and no Auto promotion without exact hardware/model evidence.
- External GPU processes are read-only pressure signals and are never terminated or signaled.
- The hardware recommendation remains the healing ceiling; transient pressure changes only the effective rung.
- Runtime activation and model lifecycle remain owned by `NativeRuntimeBackend`, which signals only PID-verified children it launched.
- Plain HTTP provider endpoints are limited to loopback or Tailscale addresses; all other endpoints require HTTPS.
- Every native llama.cpp command includes `--jinja`; DSpark variants include target, draft, projector, and draft-attention arguments.
