---
name: turbofit
description: "Operate the Turbofit adaptive local inference plugin."
version: 2.4.1
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, plugin, llama-cpp, adaptive-runtime]
---

# Turbofit

## 2.4 model authority

Two ladders converge on Qwen 3.8 Flash Next. Ladder 1 — dedicated VRAM (dense is fine): Bonsai 2 27B PTQ1_0/PQ2_0 at 8–11 GB → Underdog 27B PQ2_0 at 12–15 GB → Qwen 3.8 27B GSQ-RCO IQ3_S at 16–48 GB (262K from 24 GB) → Flash Next IQ3_XXS at 64 GB+. Ladder 2 — unified/CPU (MoE only): Maple Preview 20B-A1B at 8–15 GB → Cyber-Tiel Coder 35B-A3B UD-Q4_K_XL at 16 GB+ → Flash Next → DeepSeek V4.1 (1017) as the top rung (native engine pending). Fit math: model + KV ≤ tier. Cross-cutting: MTP on at every rung; IQ3_S is lossless — do not push a heavier quant past it for the power bill. `serve auto` builds the fit chain from these ladders and walks down on OOM/timeout — the Fit List IS the fallback chain.

Use this bundled plugin skill when configuring or inspecting Turbofit for Hermes Agent.

## Operator workflow

1. Call `turbofit_status` to inspect provider registration, gateway health, selected hardware profile, active rung, and stable routes.
2. Call `turbofit_configure` with `profile: auto` for hardware selection. Manual `hardware-*gb` profiles are accepted only when physical topology fits.
3. Set `primary: true` to use `custom:turbofit` with model `auto` as the main Hermes provider.
4. Set `fallback: true` to append Turbofit to the canonical `fallback_providers` chain; set it false to remove only Turbofit while preserving other fallbacks.
5. Set `publish_tailnet: true` to create private Tailscale Serve routes for the provider and dashboard; the returned HTTPS provider URL is registered automatically.
6. Set `install_turbosouth: true` (alias `install_turbosouth: true`) to install or update the canonical `SouthpawIN/turbosouth` GitHub-current profile without replacing its memories or user state — TurboSouth sends tested pull requests upstream to TurboFit.
7. Set `install_freetoken: true` only on Linux x86_64 + NVIDIA driver 580+ + CUDA toolkit 13+ to install pinned FreeToken 0.1.2 as a text-only MoE **candidate**. It never changes Auto until exact on-box campaigns promote a supported model recipe.
8. Start a new Hermes session after provider changes.

The same controls are available in Hermes Desktop under **Turbofit** and through `/turbofit status|update|shift|serve|tiers|setup`.

`/turbofit setup` refreshes Hermes Desktop. A refused `http://127.0.0.1:8091/v1/models` means the Turbofit runtime is down — not the Hermes messaging gateway. Diagnose that from TurboSouth or Desktop with `turbofit_status`. Setup downloads recommended models if they are missing.

## Intelligence benchmarks

- `scripts/turbofit-catalog-campaign` proves native runtime fit and TPS; it does not produce intelligence scores.
- `scripts/turbofit-intelligence-campaign` runs the exact successful quantized production recipe through pinned DeepSWE and the Turbofit agentic main/auxiliary pair harness.
- Use `status`, `run-one`, or `run --limit N`; state is resumable in `references/intelligence-campaign-state.json`.
- Scores require both benchmark suites and immutable raw evidence. Never replace missing scores with catalog tiers, parameter counts, or vendor benchmark claims.
- `/turbofit tiers` and `scripts/turbofit-hardware-tiers` show every 8/16/24/32/48/64/96/128/192/256/384 GB class with pending versus measured intelligence and TPS.
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
- FreeToken support is candidate-only: no active Qwen 3.8/Cyber-Tiel Coder replacement, no source TPS inheritance, and no Auto promotion without exact hardware/model evidence.
- External GPU processes are read-only pressure signals and are never terminated or signaled.
- The hardware recommendation remains the healing ceiling; transient pressure changes only the effective rung.
- Runtime activation and model lifecycle remain owned by `NativeRuntimeBackend`, which signals only PID-verified children it launched.
- Plain HTTP provider endpoints are limited to loopback or Tailscale addresses; all other endpoints require HTTPS.
- Every native llama.cpp command includes `--jinja`; DSpark variants include target, draft, projector, and draft-attention arguments.
