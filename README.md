# TurboFit

![Turbofit — unified backend, amber and mint aesthetic](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/turbofit-hero.png)

**One model provider. Every machine. The best local configuration the hardware can safely sustain.**

TurboFit is a first-class [Hermes Agent](https://github.com/NousResearch/hermes-agent) provider and adaptive local-inference runtime. It inventories your physical compute and total usable memory, then **populates your model list with every model that fits your hardware tier** — the single best sustainable fit on top (your `auto` model), the rest below as explicit choices. It recommends an evidence-backed **Fit Ladder**, launches a native backend, and exposes one OpenAI-compatible endpoint. Pick the `auto` model and it runs the largest model your machine can sustain — with a measured, reproducible fallback if a rung can't hold.

> **TurboFit is the whole project. TurboSouth is the dedicated service-tech bot** that helps you install it, land it on a machine that's never been tested, configure it, answer questions, and send tested fixes back upstream to TurboFit.

---

## Install

**One package.** Installing TurboFit brings **TurboSouth** along with it (and TurboSouth ships the **sovthpaw pet** as its default) — you install one thing, not three. You don't download TurboSouth, download the pet, and combine them yourself; the package does that for you.

The fastest path — **paste this into your Hermes** and it does the rest (enable, set as your provider on the `auto` model, install TurboSouth, select the pet):

> Install TurboFit from https://github.com/SouthpawIN/turbofit.git

Prefer a specific route? All five are supported.

| Option | How |
|--------|-----|
| **Hermes prompt** (recommended) | Paste the one-liner above — your Hermes installs, enables, configures, and selects the pet. |
| **Plugin** | `hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git` |
| **TurboSouth** (bot installs TurboFit) | `hermes profile install https://github.com/SouthpawIN/turbosouth.git` |
| **Hermes Desktop** | **Turbofit** sidebar → primary + fallback + **Install or update bundled TurboSouth** → Apply |
| **Manual** | `git clone https://github.com/SouthpawIN/turbofit.git` → follow `after-install.md` |

### After install (any path)
1. Quit Hermes Desktop (⌘Q) and open a new session.
2. `/turbofit status` → `/turbofit setup` — **this scans your machine and populates your model list** with every model that fits your hardware tier: the best sustainable fit on top (your `auto` model), the rest below as explicit choices.
3. The sovthpaw pet is selected for you (or run `hermes pets select sovthpaw` manually).

---

## The Fit — two ladders that converge on Flash Next (source of truth for the UI)
![TurboFit hero](https://raw.githubusercontent.com/SouthpawIN/turbofit/main/assets/turbofit-hero.jpg)

> **Two ladders, not one.** Dense doesn't run well on CPU / Unified memory, so the tiers split into **two fit ladders** — the **GPU-dedicated ladder is dense**, the **Unified (CPU / Mac) ladder is MoE** — and **both converge on Qwen 3.8 Flash Next**. Flash Next is an MoE, and it's the *only* rung where the dense ladder switches to MoE.
>
> `serve auto` scans the machine, builds the fit chain from the right ladder, picks the largest sustainable rung, and walks down on OOM/timeout — **the Fit List IS the fallback chain**.
>
> **Two cross-cutting rules for every rung:**
> - **MTP on whenever we can** — every one of these models has MTP (multi-token prediction) available in some form; turn it on at each rung.
> - **Mind the power bill (kWh / hour).** IQ3_S is lossless, so there's no reason to push a heavier quant (and a higher power draw) past it — 32 GB and 48 GB both stay on Qwen 3.8 27B GSQ-RCO @ IQ3_S.
>
> Deep detail: [fit-ladder](references/fit-ladder.md) · Canonical data in the product repo: `fit-lists.json` · `model-catalog.json`.

### Ladder 1 — Dedicated (GPU / VRAM, **dense**)

| VRAM | Rung | Quant | Context | Notes |
|------|------|-------|---------|-------|
| 8 GB | **Ternary Bonsai 2 27B** | PTQ1_0 / PQ2_0 | 131K | dense ternary; prism-llama.cpp |
| 12 GB | **Underdog 27B** | PQ2_0 | 131K | dense ternary; prism-llama.cpp |
| 16 GB | **Qwen 3.8 27B GSQ-RCO** | IQ3_S | 131K | near-lossless |
| 24 GB | **Qwen 3.8 27B GSQ-RCO** | IQ3_S | 262K | more KV headroom |
| 32 GB | **Qwen 3.8 27B GSQ-RCO** | IQ3_S (lossless) | 262K+ | don't up the power bill past lossless |
| 48 GB | **Qwen 3.8 27B GSQ-RCO** | IQ3_S + MTP | 262K+ | MTP on |
| 64 GB | **Qwen 3.8 Flash Next** | IQ3_XXS | 262K native | MoE — convergence |
| 96 GB | **Qwen 3.8 Flash Next** | IQ3_XXS | 1M w/ YaRN | beats Claude Opus 4.6 |
| 128 GB | **Qwen 3.8 Flash Next** | IQ3_XXS | 1M | max context |

### Ladder 2 — Unified (CPU / Apple Silicon, **MoE only — no dense**)

| RAM | Rung | Quant | Notes |
|-----|------|-------|-------|
| 8–15 GB | **Maple Preview 20B-A1B** | TQ2_0 | MoE; 218 tok/s on M4 |
| 16–23 GB | **Cyber-Tiel Coder 35B-A3B** | UD-IQ3_XXS | MoE, 3B active; faster than GSQ-RCO on CPU |
| 24–31 GB | **Cyber-Tiel Coder 35B-A3B** | UD-Q4_K_XL | MoE, 3B active |
| 32–47 GB | **Cyber-Tiel Coder 35B-A3B** | UD-Q5_K_XL | MoE, 3B active |
| 48–63 GB | **Cyber-Tiel Coder 35B-A3B** | UD-Q6_K_XL | MoE, 3B active |
| 64 GB+ | **Qwen 3.8 Flash Next** | IQ3_XXS | MoE — convergence (candidate-only on Apple, no macOS engine yet) |
| top | **DeepSeek V4.1 Flash** | FP8 / UD-Q4_K_XL | MoE, 763B; sits **above** Flash Next (engine pending) |

### DeepSeek V4.1 — the 1017 update (pending)

A native engine for **DeepSeek V4.1** (the **1017 update**) is in testing and measuring well. It's a larger model, so if it fits a machine it sits in the tier **above** Qwen 3.8 Flash Next — currently the top of the unified ladder. (Open test: the largest tier it can run on.)

### Live catalog (what `serve auto` ships today)

The currently-automated, downloadable set (`model-catalog.json`) — **62 models in 14 families**: `Ternary Bonsai 2 27B` (3 variants), `Underdog 27B`, `Qwen 3.8 27B` (11, incl. GSQ-RCO), `Qwen 3.8 Flash Next`, `Maple Preview 20B-A1B`, `Cyber-Tiel Coder 35B-A3B` (5), `Ornith 1.5 35A3B` (3), `Ternary Bonsai 27B` (12), `Bonsai 27B` (18), `Carwin Nano` (2), `GLM 5.2`, `Laguna S2.1` (2), `MiniMax M3`, `Darwin 180B RSI`. The two ladders above are the curated rungs `serve auto` promotes; the rest of the catalog is Model Zoo download-only.

### New models (canonical names)

New additions this release, by exact canonical name (full variant/role/state table in [`skills/turbofit/README.md`](https://github.com/SouthpawIN/turbofit/blob/main/skills/turbofit/README.md)):

- **Maple Preview 20B-A1B** — MoE main candidate (TQ2_0, 8 GB envelope)
- **Qwen 3.8 27B GSQ-RCO** — near-lossless dense main (IQ3_S / IQ3_XXS / IQ2_S / IQ2_XS)
- **Ornith 1.5 35A3B** — MoE auxiliary (Q4_K_M, oQ4e / oQ8e MTP)
- **MiniMax Music 3** — music generation (pinned integration candidate)
- **NVIDIA Parakeet TDT 0.6B v3** — speech-to-text (pinned integration candidate)
- **Soprano TTS** — text-to-speech (pinned integration candidate)

### API lineup + named local picks (aux)

Ranked by measured evidence, not heuristics:
- **API** (Modest/Thin machines): `DeepSeek V4 Pro` (NIM, free, top open SWE-bench), `Qwen 3.7 MAX` (paid, #1 MMLU), `MiniMax M3` (NIM, free, vision, 1 M ctx).
- **Named local picks** (MRI-Trust merges + MTP grafts): `Darwin 28B Reason` (89.4 % GPQA, smartest dense), `Prism Eagle 27B` (121 tok/s, fastest), `Carnice 35A3B` (small aux).

**Pairing rules:** a text-only main **must** pair with a vision aux (never two blind models) · the aux offsets 40–85 % of token cost · the aux needs 262 K+ context to compress the main without losing info · free NIM pairs beat everything on cost.

> Deep detail: [`references/fit-ladder.md`](references/fit-ladder.md) · [`references/model-lineup.md`](references/model-lineup.md) · the Fit UI: [`wiki/models/hardware-tiers.html`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/models/hardware-tiers.html) · [`wiki/models/leaderboard.html`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/models/leaderboard.html) · Canonical data: [`fit-lists.json`](https://github.com/SouthpawIN/turbofit/blob/main/fit-lists.json) · [`model-catalog.json`](https://github.com/SouthpawIN/turbofit/blob/main/model-catalog.json)

---

## TurboSouth — your TurboFit service tech

TurboSouth is the dedicated, self-updating customer-service bot for TurboFit. It's not a second model backend — it's the hands that get you from *"I have a machine"* to *"TurboFit is running the best model it can"* — and it keeps feeding improvements back upstream.

What TurboSouth helps with:
- **Install + setup** — resolves the right Auto-chain artifacts for your machine and gets the gateway up.
- **Landing on an untested machine** — evidence-based machine-fit comparison before committing to a rung. It shows the math; it never guesses.
- **Configure** — hardware profile, primary vs. fallback chain, Tailscale exposure.
- **Answer questions** — Q&A grounded in the wiki, with citations (not vibes).
- **Troubleshooting** — `status`, `smoke`, `health`, and gateway checks.
- **Send tested upstream PRs** — evidence-backed pull requests to TurboFit (never an autonomous publish).

What it does *not* do: autonomously publish PRs, manage credentials, kill external GPU processes, or bypass the native runtime lifecycle.

> **Reciprocal install** — TurboSouth installs TurboFit for you: `hermes profile install https://github.com/SouthpawIN/turbosouth.git` (Install TurboSouth, and it sets up TurboFit when it is missing). In-product: `install_turbosouth: true` (via `/turbofit` or the Desktop settings).

---

## Multimodal status

Local multimodal is the **hard frontier** — MoE + a vision tower + 300 K+ context doesn't fit one GPU today.

- **Vision aux** (image understanding) ships now: pair a text-only main with `MiniMax M3` or `qwen3-vl-8b-instruct-v2` (76.3 %).
- **TTS + STT** run through the same gateway.
- **Full local multimodal** (image/video/music on-device) is the 2-GPU job — `Qwen3-Next-80B-A3B` (3.4 tok/s, 300 K ctx) needs ~2× 40 GB; the small always-on multimodal is `Qwen2.5-Omni-3B` (49 tok/s, 3 B).

> Deep detail: [`references/multimodal.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/multimodal.md)

---

## Benchmarks (what we've measured, and the gaps)

- **External (discovery only):** DeepSWE v1.1 (113 tasks). The only local Turbofit candidate on it is `glm-5-2-2-788bpw` — 0.438 pass@1 → **78.8 bpw**, a 2× jump over the previous local baseline.
- **Local (the gap):** most local benchmark numbers are still missing. The only measured local numbers recorded: `GLM-5.1` at 1.75 tok/s on M1, `Qwen3-30B` 8.3 tok/s, `Qwen3.5-35B-A3B` 4.4 tok/s. The "all dashes" in the local tables is the **#1 evidence gap** — local candidates aren't yet benchmarked at Tier-S standard.
- **Pending:** DeepSpec (39.1 tok/s draft, 1.77× on 3090) and DeepSWE 2.0.

External results are discovery evidence only; a model is promoted to a local runtime profile only after a reproducible local run passes its gates.

> Deep detail: [`references/benchmark-evidence.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/benchmark-evidence.md)

---

## Tailscale — a built-in TurboFit backend

Tailscale is a **built-in backend**, not an add-on or a machine-specific setup: `/turbofit serve` exposes your gateway on a private tailnet with **no public ports**, and the model picker federates across it. Any user can enable it for their own devices — it's part of the product:

- **Gateway binding:** your machine runs the Hermes Gateway on `0.0.0.0:8091`; Tailscale exposes it on your private tailnet (runs as a local service on each OS — launchd on macOS, systemd on Linux).
- **Two gateways, don't confuse them:** the Turbofit model gateway on `127.0.0.1:8091` is separate from the Hermes messaging gateway. A refused `http://127.0.0.1:8091/v1/models` (on Windows: `WinError 10061`) means the Turbofit runtime is down — setup missing, not a firewall rule. Windows native path: [`docs/windows-native-install.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/windows-native-install.md).
- **Model-picker federation:** peers auto-discover over the tailnet — each machine advertises the models it can serve, so every machine's model list auto-populates with every other machine's models.
- **The end state:** private remote access + auto-fed model lists, zero public exposure. The wiki becomes the single source of truth, centralized over Tailscale across all devices, organized top-down by the wiki's own standards.

> Deep detail: [`references/tailscale.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/tailscale.md)

---

## What you get

- **Hardware scan → populated model list** — on install it reads your GPU/RAM and populates your model list with every model that fits your tier: the best sustainable fit on top (your `auto` model), the rest below as explicit choices.
- **Fit Ladder** — the two TurboFit-List ladders converging on Flash Next, with a measured reproducible fallback (the Fit List IS the fallback chain).
- **Tailscale (built-in backend)** — private remote access + auto-fed model lists across your devices (`/turbofit serve`, verify with `/turbofit smoke`). Funnel is never used.
- **Multimodal adapters** — vision aux, TTS, and STT through the same gateway (full on-device multimodal is the frontier).
- **Native engines** — CUDA, ROCm, Vulkan, Metal, and CPU backends chosen by your hardware.
- **One package** — installing TurboFit brings TurboSouth (and its default sovthpaw pet) along automatically.

![Turbofit settings in Hermes Desktop, including fallback routing, multimodal model selection, and hardware-fit recommendations](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/hermes-desktop-turbofit-settings.png)

---

## Troubleshooting & upstream

- **Troubleshooting** — VRAM OOM (walk the ladder, don't just restart), the local-benchmark gap, gateway reachability, blind-aux pairing. → [`references/troubleshooting.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/troubleshooting.md)
- **Upstream PR process** — TurboSouth converts recurring, evidence-backed support problems into PR *suggestions*; a real PR opens only with human approval, against the exact upstream, evidence attached. → [`references/upstream-pr-process.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/upstream-pr-process.md)

---

## Where the deep data lives

This entry is the overview. Everything else is one level down, and all of it lives in **this vault topic** — no knowledge is left only in a repo:

- **[`index.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/index.md)** — the full catalog: every page in the topic, plus the fit ladders generated from the canonical data.
- **Core doctrine** — [`fit-ladder`](references/fit-ladder.md) · [`model-lineup`](references/model-lineup.md) · [`curated-lineup`](references/curated-lineup.md) · [`scaling-ladder`](references/scaling-ladder.md) · [`binary-selection`](references/binary-selection.md) · [`product-roadmap`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/product-roadmap.md)
- **Benchmarks & evidence** — [`benchmark-evidence`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/benchmark-evidence.md) · [`benchmark-leaderboard`](references/benchmark-leaderboard.md) · [`external-benchmarks`](references/external-benchmarks.md) · [`research-report`](references/research-report.md) · [`model-intel`](references/model-intel.md) · [`api-model-rankings`](references/api-model-rankings.md) · [`api-tier-rankings`](references/api-tier-rankings.md) · [`api-pairing-matrix`](references/api-pairing-matrix.md)
- **Hardware & runtime** — [`hardware-tiers`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/hardware-tiers.md) · [`hardware-tier-tps`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/hardware-tier-tps.md) · [`runtime-backends`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/runtime-backends.md) · [`model-matrix`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/model-matrix.md) · [`model-metadata`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/model-metadata.md) · [`multimodal`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/multimodal.md) · [`salt-core-integration`](references/salt-core-integration.md) · [`freetoken-runtime`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/freetoken-runtime.md)
- **Installs & ops** — [`install-paths`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/install-paths.md) · [`windows-native-install`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/windows-native-install.md) · [`remote-hermes-model-server`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/remote-hermes-model-server.md) · [`apple-arm64-unmeasured-gap`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/apple-arm64-unmeasured-gap.md) · [`maple-small-device-validation`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/maple-small-device-validation.md) · [`turbofit-check`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/turbofit-check.md) · [`campaigns`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/campaigns.md) · [`workflows`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/workflows.md) · [`troubleshooting`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/troubleshooting.md) · [`tailscale`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/tailscale.md)
- **Ecosystem** — [`ecosystem`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/ecosystem.md) maps every repo (turbofit, turbosouth, SALT-core, model-momento, benchmark-infographics, turbohaul line) to its vault page; provenance for the 2026-10-28 absorption is in [`raw/sources/2026-10-28-github-consolidation.md`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/raw/sources/2026-10-28-github-consolidation.md)
- **History & contracts** — [`changelog`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/changelog.md) · [`skill-contract`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/skill-contract.md) · [`upstream-pr-process`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/references/upstream-pr-process.md)
- **Model pages & the fit UI** — `wiki/models/` (per-model pages) · [`wiki/models/hardware-tiers.html`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/models/hardware-tiers.html) · [`wiki/models/leaderboard.html`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/models/leaderboard.html)
- **[`ROADMAP.md`](ROADMAP.md)** — the product roadmap (vision, phases, open gaps)
- Raw sources: [`raw/inbox/`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/raw/inbox/) · [`raw/sources/`](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/raw/sources/)
- Canonical machine data (stays in the repo, read at runtime): [`fit-lists.json`](https://github.com/SouthpawIN/turbofit/blob/main/references/fit-lists.json) · [`model-catalog.json`](https://github.com/SouthpawIN/turbofit/blob/main/references/model-catalog.json)

---

## Visual reference

![Fit ladder — from low-VRAM rungs to the largest sustainable model](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/scaling-ladder.png)
![Provider integration](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/provider-integration.png)
![2.4 Spec-decode — Draft. Verify. Accept.](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/turbofit-2.4-spec-decode.png)

## License

MIT

---
> _This README is generated from the [TurboFit wiki](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/README.md) — the single source of truth. To change it, edit the wiki hub and re-run `tools/sync-wiki-readme.sh`._
