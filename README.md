# TurboFit

![Turbofit — unified backend, amber and mint aesthetic](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/turbofit-hero.png)

**One model provider. Every machine you own. The largest local model your hardware can actually hold — and it gets out of the way the moment something else needs the GPU.**

TurboFit is a first-class [Hermes Agent](https://github.com/NousResearch/hermes-agent) provider for local inference. Install it, set it as a provider, pick `auto`, and stop thinking about local models. It picks the model your machine fits, keeps it resident, and steps aside — context first, then model class, then the whole model — when a game, a render, or six browser tabs want the card. When the pressure is gone it climbs back on its own.

```bash
hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

---

## Start here: TurboSouth

**TurboSouth is TurboFit's support profile — an agent that installs TurboFit for you.** It inventories your hardware, compares it against the current catalog, installs the model that fits, proves it with a real completion, and fixes what breaks. When it finds a reusable bug it opens a tested PR against TurboFit.

```bash
git clone https://github.com/SouthpawIN/turbosouth.git
cd turbosouth && scripts/install
```

```text
Install TurboFit and verify a real local completion.
Configure auto for this machine with no cloud fallback.
Auto keeps picking the wrong model for my card — fix it.
```

TurboSouth is a *profile*, not a second product: one TurboFit implementation, one support surface, and the sovthpaw pet arrives with it. `hermes profile install https://github.com/SouthpawIN/turbosouth.git --name turbosouth --yes` works too — but Hermes does not act on a distribution's `dependencies:`, so run the plugin install **first** (or just use `scripts/install`, which does both and verifies it).

## Install

**1 — Let Hermes do it.** Paste this into a Hermes session:

```text
Install TurboFit from https://github.com/SouthpawIN/turbofit.git with
`hermes plugins install --enable`, then run `/turbofit setup`, then verify with
`/turbofit status` and a real completion through http://127.0.0.1:8091/v1/models.
Do not report success until /v1/models answers and one completion returns tokens.
```

**2 — Plugin only** (you already know your hardware):

```bash
hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git
hermes config set providers.turbofit.api http://127.0.0.1:8091/v1
```

**3 — TurboSouth** (recommended on a machine that has never run local models): see above.

**4 — From source**, if you are working on TurboFit itself:

```bash
git clone https://github.com/SouthpawIN/turbofit.git
cd turbofit && scripts/install.sh
```

**Verify — any path.** A refused `http://127.0.0.1:8091/v1/models` means the *TurboFit* gateway is down, not the Hermes messaging gateway. Then:

```bash
scripts/turbofit-runtime status        # profile, rung, routes
/turbofit status                       # dials + chain, in-session
scripts/turbofit-controller --once     # one bounded tick
curl -s http://127.0.0.1:8091/v1/models
```

## What it does at runtime

| Dial | Meaning |
|---|---|
| **Memory pressure** | used ÷ total per card; external apps are counted as untouchable, never as reclaimable |
| **Context** | current window vs native max, and the shed steps (262k → 131k → 65k → 32k) |
| **Fallback chain** | context first → smaller model → release residency; heal back up when the pressure clears |

Every rung is a Turbofile entry bound to an evidence digest. `external_gpu_priority: absolute` is enforced: a contended card with no cheaper rung makes TurboFit **yield** the model, and it will not reload while the pressure persists. Pressure can only lower the effective rung — never raise the ceiling your hardware was measured for.

```
hermes → Turbofit gateway :8091 → route state → local runtime
                                    ↑            (llama.cpp / Strata unit)
              adaptive controller ┘  probe → policy → reconcile → publish
```

## Models

Two ladders, never one. Dense on dedicated VRAM; Mixture-of-Experts on unified/CPU, because dense crawls there.

**Ladder 1 — dedicated VRAM**

| Band | Model | Context |
|---|---|---|
| 8–11 GB | Ternary Bonsai 2 27B (PTQ1_0 / PQ2_0) | 131k |
| 12–15 GB | Underdog 27B (PQ2_0) | 131k |
| 16–23 GB | Qwen 3.8 27B GSQ-RCO **IQ3_S** | 131k |
| 24–63 GB | Qwen 3.8 27B GSQ-RCO **IQ3_S** | 262k |
| 64 GB+ | Qwen 3.8 Flash Next (Strata) | 262k |

**Ladder 2 — unified / CPU / Apple Silicon (MoE only)**

| Band | Model | Context |
|---|---|---|
| 8–15 GB | Maple Preview 20B-A1B (TQ2_0) | 131k |
| 16–63 GB | Cyber-Tiel Coder 35B-A3B (UD-IQ3_XXS → Q6_K_XL by band) | 262k |
| 64 GB+ | Qwen 3.8 Flash Next (Strata) | 262k |

Rules that hold at every rung: MTP on wherever the model supports it; **IQ3_S is lossless**, so 32 GB and 48 GB both stay on it rather than paying for a heavier quant; DFlash2 drafters belong to Qwen only, Bonsai keeps its own Prism DSpark sidecar.

**Chain options** — `references/fallback-chain.json`, swappable per slot, all `candidate` until measured on the box in question:

- **Qwen 3.8 27B GSQ-RCO** — the promoted dense-27B default. *Unleashed is retired; GSQ-RCO replaced it.*
- **DeepSeek V4.1 Flash** — GSQ-RCO **and** a Strata build; 763B/16B MoE, so a 96 GB+ rung, not a 48 GB default.
- **Muse Glimmer 30B (Hermes-Agentic)** — agentic-tuned 30B, head-to-head candidate against GSQ-RCO.
- **LFM 24A2B** — 24B total / 2B active MoE candidate. **GROOT** — candidate, artifacts not yet pinned.

## Benchmarks

Measured, not estimated: real `/completion` calls. A rung with no number here is **unmeasured**, and the UI labels it that way.

**2× RTX 3090 (24 GB each), 2026-08-22 — 8k context, 128 generated tokens**

| Model | Decode | Prefill | Source |
|---|---|---|---|
| Bonsai 27B Q1_0 | **63.8 tok/s** | 65.9 tok/s | `references/results/hardware-tier-tps-2x24-20260822.json` |
| Qwen 3.8 27B Q4_K_M | **37.7 tok/s** | 58.7 tok/s | same |
| Carwin MoE Nano (aux) | **132.5 tok/s** | 137.3 tok/s | same |

**Long-context sweep, same box:** Bonsai 27B holds 58.5 tok/s at short context and 9.4 tok/s at the far end; Ornith 1.5 35A3B 31.2 → 10.8 tok/s; Carwin MoE Nano 24.6 → 11.6 tok/s. Context is not free — that is why the chain sheds it first.

**Cross-device, 2026-11-03:** a GPU-less node asked `auto`, the MacBook served `cyber-tiel-coder-35b-a3b` — 124.9 tok/s prefill, 27.5 tok/s decode, 237 s wall clock of which ~235 s was cold load, not inference.

**Not yet measured:** every IQ3_S rung, Flash Next, DeepSeek V4.1 Flash, Glimmer 30B, LFM 24A2B, GROOT, and every Apple Silicon tier. `scripts/turbofit-catalog-campaign` (fit + TPS) and `scripts/turbofit-intelligence-campaign` (pinned DeepSWE plus the agentic main/aux pair) are how a number gets here; `scripts/turbofit-promote-list-winner` refuses to promote without exact-tier evidence.

## Tailscale: use any model from any of your devices

Every TurboFit gateway serves its own models **and** advertises its peers'. That is how a phone runs Qwen Flash Next that lives on the machine with the GPU.

```bash
# on the node that has the model
tailscale serve --bg --https=443 http://127.0.0.1:8091
# on any other device on the tailnet
provider: custom:turbofit   base_url: https://<node>.your-tailnet.ts.net/v1   model: auto
```

- `auto` on a node with no GPU delegates to a healthy peer instead of failing.
- A peer's model is addressable as `peer.ts.net/model-name`; a bare `model-name` resolves to whichever peer serves it.
- A model you name explicitly is **never** silently answered by a hosted API — you get a retryable `503` naming it. Substitution is legal only for `auto`/`active`, and only when `TURBOFIT_ALLOW_API=true`.
- Plain HTTP is honoured only on loopback and Tailscale addresses; anything else needs HTTPS.

## Keeping it current

```bash
scripts/install-self-update            # fast-forward + restart on change, hourly
scripts/install-self-update --dry-run  # print the units first
```

Engine binaries are PM-pinned and never downloaded at boot. TurboFit tracks upstream llama.cpp (stock and atomic forks), Strata, MLX, Prism/DSpark and DFlash2 by version; `references/binary-selection.md` decides stock-vs-fork per model, not per machine.

## Docs

- Architecture, authority chain, honest debt list — [`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md)
- How TurboFit maps onto Hermes' built-in local runtime — [`docs/hermes-local-runtime-adapter.md`](docs/hermes-local-runtime-adapter.md)
- Chain data — [`references/fallback-chain.json`](references/fallback-chain.json) · ladders — [`references/fit-lists.json`](references/fit-lists.json) · catalog — [`references/model-catalog.json`](references/model-catalog.json)
- Binary selection — [`references/binary-selection.md`](references/binary-selection.md) · troubleshooting — [`docs/turbofit-check.md`](docs/turbofit-check.md)
- Changelog — [`CHANGELOG.md`](CHANGELOG.md) · roadmap — [`ROADMAP.md`](ROADMAP.md) · full wiki topic — [agent-wiki](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/index.md)

MIT.

---
> _This README is generated from the [TurboFit wiki](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/README.md) — the single source of truth. To change it, edit the wiki hub and re-run `tools/sync-wiki-readme.sh`._
