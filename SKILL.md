---
name: turbofit
description: "Operate the Turbofit adaptive local inference plugin."
version: 3.1.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, plugin, turbohaul, adaptive-runtime]
---

# Turbofit

Use this bundled plugin skill when configuring or inspecting Turbofit for Hermes Agent.

## Operator workflow

1. Call `turbofit_status` to inspect provider registration, gateway health, selected hardware profile, active rung, and stable routes.
2. Call `turbofit_configure` with `profile: auto` for hardware selection. Manual `hardware-*gb` profiles are accepted only when physical topology fits.
3. Set `primary: true` to use `custom:turbofit` with model `auto` as the main Hermes provider.
4. Set `fallback: true` to append Turbofit to the canonical `fallback_providers` chain; set it false to remove only Turbofit while preserving other fallbacks.
5. Set `publish_tailnet: true` to create private Tailscale Serve routes for the provider and dashboard; the returned HTTPS provider URL is registered automatically.
6. Set `install_sirvir: true` to install or update the bundled Turbofit customer-service profile without replacing its memories or user state.
7. Start a new Hermes session after provider changes.

The same controls are available in `hermes dashboard` under **Turbofit** and through `/turbofit status|setup`.

## Invariants

- Stable model IDs are `auto`, `active:main`, and `active:aux`.
- External GPU processes are read-only pressure signals and are never terminated or signaled.
- The hardware recommendation remains the healing ceiling; transient pressure changes only the effective rung.
- Runtime activation and model lifecycle remain owned by Turbohaul Manager.
- Plain HTTP provider endpoints are limited to loopback or Tailscale addresses; all other endpoints require HTTPS.
- `--jinja` is represented as `llama_server_flags.jinja: true` in compiled Turbohaul manifests.
