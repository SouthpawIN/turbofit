# TurboFit

![Turbofit — unified backend, amber and mint aesthetic](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/turbofit-hero.png)

**One model provider. Every machine you own. The largest local model your hardware can actually hold — and it gets out of the way the moment something else needs the GPU.**

TurboFit is a first-class [Hermes Agent](https://github.com/NousResearch/hermes-agent) provider for local inference. Install it, set it as a provider, pick `auto`, and stop thinking about local models. It picks the model your machine fits, keeps it resident, and steps aside — context first, then model class, then the whole model — when a game, a render, or six browser tabs want the card. When the pressure is gone it climbs back on its own.

```bash
hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

---

## TurboSouth

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

TurboSouth installs TurboFit when it is missing, then verifies it with a real completion. **Install TurboSouth** with `scripts/install` (it installs the plugin first and proves it with a real completion) or with `hermes profile install`. TurboSouth is a *profile*, not a second product: one TurboFit implementation, one support surface, and the sovthpaw pet arrives with it. `hermes profile install https://github.com/SouthpawIN/turbosouth.git --name turbosouth --yes` works too — but Hermes does not act on a distribution's `dependencies:`, so run the plugin install **first** (or just use `scripts/install`, which does both and verifies it).

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

Hermes scans a plugin's files the first time it installs it, so expect a couple of minutes with no output before `✓ Plugin turbofit enabled.` A script that gives up after 60 s will look like a hang; it is not.

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

**Auxiliary slots** — same catalog, same evidence rules: **MiniMax Music 3** (music), **NVIDIA Parakeet TDT 0.6B v3** (transcription), **Soprano TTS** (speech), with **Ornith 1.5 35A3B** and Maple Preview 20B-A1B eligible for the light text slots.

Rules that hold at every rung: MTP on wherever the model supports it; **IQ3_S is lossless**, so 32 GB and 48 GB both stay on it rather than paying for a heavier quant; DFlash2 drafters belong to Qwen only, Bonsai keeps its own Prism DSpark sidecar.

**Chain options** — `references/fallback-chain.json`, swappable per slot, all `candidate` until measured on the box in question:

- **Qwen 3.8 27B GSQ-RCO** — the promoted dense-27B default. *Unleashed is retired; GSQ-RCO replaced it.*
- **DeepSeek V4.1 Flash** — GSQ-RCO **and** a Strata build; 763B/16B MoE, so a 96 GB+ rung, not a 48 GB default.
- **Muse Glimmer 30B (Hermes-Agentic)** — agentic-tuned 30B, head-to-head candidate against GSQ-RCO.
- **LFM 24A2B** — 24B total / 2B active MoE candidate. **GROOT** — candidate, artifacts not yet pinned.

## Tailscale: use any model from any of your devices

Every TurboFit gateway serves its own models **and** advertises its peers'. That is how a phone runs Qwen Flash Next that lives on the machine with the GPU.

```bash
# on the node that has the model
tailscale serve --bg --https=443 http://127.0.0.1:8091
# on any other device on the tailnet
provider: custom:turbofit   base_url: https://<node>.your-tailnet.ts.net/v1   model: auto
```

Start the local endpoint with `/turbofit serve` (background: `scripts/turbofit-gateway.py`), prove it from the other device with `/turbofit smoke`, and point clients at the tailnet address. TurboFit never uses Funnel — nothing here needs public ingress, and a peer is only ever reachable across your own tailnet.

- `auto` on a node with no GPU delegates to a healthy peer instead of failing.
- A peer's model is addressable as `peer.ts.net/model-name`; a bare `model-name` resolves to whichever peer serves it.
- A model you name explicitly is **never** silently answered by a hosted API — you get a retryable `503` naming it. Substitution is legal only for `auto`/`active`, and only when `TURBOFIT_ALLOW_API=true`.
- Plain HTTP is honoured only on loopback and Tailscale addresses; anything else needs HTTPS.

## Built on Hermes' own runtime

TurboFit is a plugin *on top of* Hermes, not a parallel stack. Hermes already ships a managed llama.cpp runtime — PM-pinned engines, one supervised router-mode server, a context ladder with growth, a hardware budget with physics checks, MTP launch flags, idle unload. Those are adopted rather than re-implemented:

```bash
scripts/turbofit-hermes-bridge        # 10 of 10 primitives adopted from Hermes
```

What TurboFit adds, because Hermes has no equivalent: external pressure sensing, residency yield and restore, the two evidence-gated ladders, the federated gateway, and the benchmark campaigns. Hermes keeps deciding *how* a server lives; TurboFit decides *what* runs and *when it steps aside*.

Strata gets the same treatment — `scripts/turbofit-strata` prints the plan the engine will be launched with, from the 0.2.10 rules (8 GB cards start at 128K, DFlash2 by default, the all-cards split engine when two sm 8.6 cards have the RAM).

## Keeping it current

```bash
scripts/install-self-update            # fast-forward + restart on change, hourly
scripts/install-self-update --dry-run  # print the units first
```

Engine binaries are PM-pinned and never downloaded at boot. TurboFit tracks upstream llama.cpp (stock and atomic forks), Strata, MLX, Prism/DSpark and DFlash2 by version; `references/binary-selection.md` decides stock-vs-fork per model, not per machine.

## Troubleshooting

- **`WinError 10061` on Windows** — connection refused on `127.0.0.1:8091`: nothing is listening there. That port belongs to the TurboFit gateway, not the Hermes messaging gateway. Start it (`/turbofit setup`, or `scripts/turbofit-gateway.py`) and confirm with `curl http://127.0.0.1:8091/health`.
- **"no evidence-backed configuration fits"** — your topology has no measured rung yet. Run `scripts/turbofit-catalog-campaign`, or pick from `compatible_lanes` in the `/turbofit` report.
- **A model answers from the cloud** — only `auto`/`active` may cross to an API backend, and only with `TURBOFIT_ALLOW_API=true`. Naming a local model returns a retryable `503` instead of a substituted answer.
- **Windows specifics** — [`docs/windows-native-install.md`](docs/windows-native-install.md) covers the native install, the 8091 port check and the WinError 10061 path.
- **Strata feels slower after an update** — run `scripts/turbofit-strata`: the app (0.2.10) and its engine packages (0.1.56) version independently, and so do its drafters.

## Docs

- Architecture, authority chain, honest debt list — [`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md)
- How TurboFit maps onto Hermes' built-in local runtime — [`docs/hermes-local-runtime-adapter.md`](docs/hermes-local-runtime-adapter.md)
- Chain data — [`references/fallback-chain.json`](references/fallback-chain.json) · ladders — [`references/fit-lists.json`](references/fit-lists.json) · catalog — [`references/model-catalog.json`](references/model-catalog.json)
- Binary selection — [`references/binary-selection.md`](references/binary-selection.md) · troubleshooting — [`docs/turbofit-check.md`](docs/turbofit-check.md)
- Changelog — [`CHANGELOG.md`](CHANGELOG.md) · roadmap — [`ROADMAP.md`](ROADMAP.md) · full wiki topic — [agent-wiki](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/index.md)

MIT.

---
> _This README is generated from the [TurboFit wiki](https://github.com/SouthpawIN/agent-wiki/blob/wiki/wiki-topics/turbofit/README.md) — the single source of truth. To change it, edit the wiki hub and re-run `tools/sync-wiki-readme.sh`._
