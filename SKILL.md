---
name: turbofit
description: "Operate the Turbofit adaptive local inference plugin."
version: 2.2.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, plugin, llama-cpp, adaptive-runtime]
---

# Turbofit

## 2.2 model authority

Qwen 3.8 27B replaces the retired GRM family through six pinned Q4/Q8/BF16 variants, with and without MTP, plus explicit vision projectors. Turbofit 2.2 also adds DeepSeek V4 Flash 0731 Q2 DwarfStar and pinned MiniMax Music 3, NVIDIA Parakeet TDT 0.6B v3, and Soprano TTS integration candidates. The exhaustive campaign contains 1,620 rows; new catalog entries remain candidates until current-recipe physical evidence promotes them.

Use this bundled plugin skill when configuring or inspecting Turbofit for Hermes Agent.

## Operator workflow

1. Call `turbofit_status` to inspect provider registration, gateway health, selected hardware profile, active rung, and stable routes.
2. Call `turbofit_configure` with `profile: auto` for hardware selection. Manual `hardware-*gb` profiles are accepted only when physical topology fits.
3. Set `primary: true` to use `custom:turbofit` with model `auto` as the main Hermes provider.
4. Set `fallback: true` to append Turbofit to the canonical `fallback_providers` chain; set it false to remove only Turbofit while preserving other fallbacks.
5. Set `publish_tailnet: true` to create private Tailscale Serve routes for the provider and dashboard; the returned HTTPS provider URL is registered automatically.
6. Set `install_sirvir: true` to install or update the bundled Turbofit customer-service profile without replacing its memories or user state.
7. Start a new Hermes session after provider changes.

The same controls are available in `hermes dashboard` under **Turbofit** and through `/turbofit status|tiers|setup`.

## Intelligence benchmarks

- `scripts/turbofit-catalog-campaign` proves native runtime fit and TPS; it does not produce intelligence scores.
- `scripts/turbofit-intelligence-campaign` runs the exact successful quantized production recipe through pinned DeepSWE and the Turbofit agentic main/auxiliary pair harness.
- Use `status`, `run-one`, or `run --limit N`; state is resumable in `references/intelligence-campaign-state.json`.
- Scores require both benchmark suites and immutable raw evidence. Never replace missing scores with catalog tiers, parameter counts, or vendor benchmark claims.
- `/turbofit tiers` and `scripts/turbofit-hardware-tiers` show every 8/16/24/48/64/96/200/300 GB class with pending versus measured intelligence and TPS.

## Invariants

- Stable model IDs are `auto`, `active:main`, and `active:aux`.
- External GPU processes are read-only pressure signals and are never terminated or signaled.
- The hardware recommendation remains the healing ceiling; transient pressure changes only the effective rung.
- Runtime activation and model lifecycle remain owned by `NativeRuntimeBackend`, which signals only PID-verified children it launched.
- Plain HTTP provider endpoints are limited to loopback or Tailscale addresses; all other endpoints require HTTPS.
- Every native llama.cpp command includes `--jinja`; DSpark variants include target, draft, projector, and draft-attention arguments.
