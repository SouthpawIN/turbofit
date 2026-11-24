# TurboFit

![Turbofit — unified backend, amber and mint aesthetic](https://github.com/SouthpawIN/bot-wiki/blob/wiki/wiki-topics/turbofit/wiki/assets/turbofit-hero.png)

**One model provider. Every machine you own. The largest thing your hardware can actually hold — and it gets out of the way the moment something else needs the GPU.**

TurboFit is a first-class [Hermes Agent](https://github.com/NousResearch/hermes-agent) provider for local inference. Install it, pick `auto`, and stop thinking about local models. It scans your machine, downloads the model that fits, keeps it resident, and steps aside — context first, then model class, then the whole model — when a game, a render, or six browser tabs want the card. When the pressure is gone it climbs back on its own.

It is not a llama.cpp wrapper. TurboFit is a **multi-engine coordinator**: one catalog, one fit ladder, one gateway, and whichever runtime actually serves the checkpoint — llama.cpp, Strata, vLLM, or a media runtime for image, video, music and speech.

---

## Install

```bash
hermes plugins install --enable https://github.com/SouthpawIN/turbofit.git
```

Then, in a Hermes session:

```text
/turbofit setup
```

That is the whole path. Setup probes your hardware, picks the model, downloads it, starts the runtime, and points Hermes at it. Hermes scans a plugin's files the first time it installs it, so expect a couple of minutes with no output before `✓ Plugin turbofit enabled.` A script that gives up after 60 s will look like a hang; it is not.

**Other ways in**

| Path | Command | When |
|---|---|---|
| Let an agent do it | paste the prompt below into Hermes | you want it verified end to end |
| Plugin only | `hermes plugins install --enable …` then `hermes config set providers.turbofit.api http://127.0.0.1:8091/v1` | you already know your hardware |
| TurboSouth | `git clone https://github.com/SouthpawIN/turbosouth.git && cd turbosouth && scripts/install` | first local-model machine, or something is broken |
| From source | `git clone https://github.com/SouthpawIN/turbofit.git && cd turbofit && scripts/install.sh` | you are working on TurboFit itself |

```text
Install TurboFit from https://github.com/SouthpawIN/turbofit.git with
`hermes plugins install --enable`, then run `/turbofit setup`, then verify with
`/turbofit status` and a real completion through http://127.0.0.1:8091/v1/models.
Do not report success until /v1/models answers and one completion returns tokens.
```

## Use it

```bash
/turbofit setup                 # scan, recommend, download, configure, serve
/turbofit status                # the three dials and your fallback chain
/turbofit shift <model>         # change gear by hand
/turbofit intelligence          # re-rank by measured intelligence
/turbofit balanced | speed      # …or by the balance, or by tokens/second
```

From a shell, the same things:

```bash
scripts/turbofit-runtime status        # profile, rung, routes
scripts/turbofit-controller --once     # one bounded control tick
curl -s http://127.0.0.1:8091/v1/models
```

**The dashboard** is an instrument cluster: three round dials — memory pressure, context window against its native maximum, and measured tokens/second — above a **gear-shift tree** of your fallback chain. Each level is a numbered gate on a rail; the engaged one is lit, a level that cannot fit right now is dashed, and a level served by another machine carries a `tailnet` badge and costs no local VRAM. Every gear has a **shift** button. An unprobed host reads `—`, never a fake 0%.

**Your fallback chain is yours.** Put your own order in `~/.config/turbofit/fallback-chain.json` (or point `$TURBOFIT_CHAIN` at a file) and TurboFit serves that instead of the shipped default — no repo edit, and it survives updates. A level may name any smaller model on this machine, or any model served by another TurboFit node: give it an `endpoint` and optionally a `peer` name. That is how you keep working when your own GPUs are busy with something else.

**Serving on the tailnet is the default.** There is no separate "start sharing" step: setup publishes the provider and dashboard endpoints through Tailscale Serve and points Hermes at that tailnet URL, so any other TurboFit node can reach the model with no extra configuration. If Tailscale is missing or down, publication says why and the provider stays on its loopback URL — connected either way, never unconfigured. Opting out is one flag (`publish_tailnet: false`); the default is on, not forced.

## TurboSouth — when you want a hand

**TurboSouth is TurboFit's support profile: an agent that installs TurboFit for you and fixes it when it breaks.** It inventories your hardware, compares it against the current catalog, installs the model that fits, proves it with a real completion, and repairs what fails. When it finds a reusable bug it opens a tested PR against TurboFit.

```bash
git clone https://github.com/SouthpawIN/turbosouth.git
cd turbosouth && scripts/install
```

Talk to it in plain language:

```text
Install TurboFit and verify a real local completion.
Configure auto for this machine with no cloud fallback.
Auto keeps picking the wrong model for my card — fix it.
```

**Install TurboSouth** with `scripts/install` — it installs the plugin first and proves it with a real completion. TurboSouth installs TurboFit when it is missing, then verifies it with a real completion. TurboSouth is a *profile*, not a second product: one TurboFit implementation, one support surface, and the sovthpaw pet arrives with it. `hermes profile install https://github.com/SouthpawIN/turbosouth.git --name turbosouth --yes` also works — but Hermes does not act on a distribution's `dependencies:`, so run the plugin install **first**, or just use `scripts/install`, which does both and verifies the result.


---

## What it does at runtime

| Dial | Meaning |
|---|---|
| **Memory pressure** | used ÷ total per card; external apps are counted as untouchable, never as reclaimable |
| **Context** | current window vs native max, and the shed steps (262k → 131k → 65k), never below the **64K floor** |
| **Fallback chain** | context first → smaller model → release residency; heal back up when the pressure clears |

Every rung is a Turbofile entry bound to an evidence digest. `external_gpu_priority: absolute` is enforced: a contended card with no cheaper rung makes TurboFit **yield** the model, and it will not reload while the pressure persists. Pressure can only lower the effective rung — never raise the ceiling your hardware was measured for.

```
hermes → Turbofit gateway :8091 → route state → local runtime
                                    ↑            (llama.cpp / Strata / vLLM)
              adaptive controller ┘  probe → policy → reconcile → publish
```

## Engines

Serving paths are dispatched per model by `runtime_flavor` in its recipe. A model names its engine once; the ladder, the pressure policy and the gateway do not care which one answers.

| Engine | `runtime_flavor` | Serves | Why it exists |
|---|---|---|---|
| **llama.cpp** (mainline) | `mainline` | GGUF quantizations | the default for every quantized build |
| **llama.cpp** (Prism fork) | `prism` | ternary GGUF | mainline cannot load ternary weights |
| **Strata** | `strata` | large MoE on small VRAM | streams experts from RAM/SSD, so a 68 GB model runs on a 12–24 GB card. Shape engines: `strata-engine`, `strata-inkling`, `strata-hy4` |
| **vLLM** | `vllm` | raw HF / NVFP4 checkpoints | models with no GGUF, and **~10× faster cold prefill** on long prompts |
| **Basalt** | `basalt` | Qwen3.8 Flash Next IQ3_S | tuned speculative decoding; `--mmap-experts` and `--vram-elastic` |
| **media runtimes** | per-model `runtime` | image, video, music, TTS, STT | diffusers, ACE-Step, faster-whisper, PersonaPlex |

**Strata needs AVX2, not AVX-512.** AVX-512 is an accelerator, never a requirement — the engine says so itself: *"this CPU has no AVX-512: the expert kernels run on AVX-2"*. Any modern desktop qualifies. Its documented floor is 12 GB of VRAM, with 8 GB admitted as a slow lane.

**Pinned: Strata app 0.2.21, engines 0.1.67.** That release fixes Qwen *"layer 0 never rang"* on RTX 30-series and newer, fixes the long-conversation crash, and stops infinite engine restarts after three tries. It also makes Inkling dramatically faster after a long prompt — vendor-reported, versus 0.2.20 on the same machines:

| Card | Before → after |
|---|---|
| RTX 3060 Ti 8 GB, SQ-10 | 11 → **43 tok/s** |
| RTX 5070 12 GB, SQ-10 | 12.5 → **68 tok/s** |
| RTX 4090, SQ-20 after a 32K prompt | 20 → **92 tok/s** |
| RX 7900 XT, UD-IQ3_XXS | 4.9 → **66 tok/s** |

Agent sessions now reuse their prompt on every backend, Apple Silicon included, so a long session no longer re-reads the whole prompt each turn.

**Engine choice is workload-shaped, not one winner.** On the same 96 GB rig, vLLM reads a cold 22K prompt at ~21.8K tok/s against ~1.5K for Strata/Basalt — about 10×, which is ~3–4 s to first token instead of ~50 s on a cold 80K agentic prompt. Strata wins the warm multi-turn case through prompt reuse. Verified external profiles for both live in [`runtime-profiles/`](runtime-profiles/), each with its sweep receipts:

- **vLLM · Inkling 2 Mini · 1×96** — DSpark spec 7, FP8 KV, prefix caching: **246 tok/s** C1 mean, 429 tok/s at 8 users, tool calls 5/5. FP8 KV is what makes 262K fit on one card.
- **Basalt · Qwen3.8 Flash Next IQ3_S · 1×96** — spec 6 + relaxed drafting: **425 tok/s** C1, 583 tok/s at 8 users, **35.8 ms** cached-turn TTFT.

Two Basalt flags are worth knowing beyond the speed: `--mmap-experts` cuts resident memory from ~50 GB to ~13 GB with no decode penalty, and `--vram-elastic` lets the engine shrink its VRAM so another workload can use the card — the same yield-and-heal behaviour the pressure ladder implements.

*All figures above are vendor-reported or verified-external on the hardware named. They are not measurements on your machine; `/turbofit status` reports what your own host actually does.*

## Models and hardware tiers

**87 catalog models, 302 configuration rows, four contexts (64K / 128K / 262K / 1M).** Every entry pins a Hugging Face revision and a sha256, and `tools/check-artifact-existence` fails the build if a pinned artifact stops resolving upstream.

**64K is the floor.** Nothing below 65,536 tokens is ever recommended. A tier that cannot hold 64K with anything says so instead of offering a smaller window.

**Ladder 1 — dedicated VRAM** (AVX2 host; `auto` takes the first column)

**Inkling 2 Mini is the preferred model at every tier where it fits, with Qwen 3.8 Flash Next next in the chain.** It is a 26.4B MoE with 5.4B effective active parameters — smaller than the 35B-A3B class — that matches the 27B on the AA Index and beats it on DeepSWE, SWEBench Verified and BrowseComp. Its SQ-20 pack is **bit-exact lossless**, not a lossy quant.

| Band | System RAM | Default | Next in the chain | Context |
|---|---|---|---|---|
| 8–11 GB | 32 GiB | Maple Preview 20B-A1B | Cyber-Tiel Coder 35B-A3B · **Inkling SQ-10** | 64k |
| 12–15 GB | 64 GiB | **Inkling 2 Mini SQ-10** | **Qwen 3.8 Flash Next** | 64k / 256k |
| 16–23 GB | 64 GiB | **Inkling 2 Mini SQ-15** | **Qwen 3.8 Flash Next** · Qwen 3.8 27B GSQ-RCO | 128k / 256k |
| 24–63 GB | 64 GiB | **Inkling 2 Mini SQ-20 (lossless)** | **Qwen 3.8 Flash Next** · GLM 5.3 Flash Maya-XS | 256k |
| 64–95 GB | 128 GiB | **Inkling 2 Mini SQ-20 (lossless)** | **Qwen 3.8 Flash Next** · GLM 5.3 Flash Maya | 256k |
| 96 GB+ | 128 GiB | **Inkling 2 Mini SQ-20 (lossless)** | **Qwen 3.8 Flash Next** · Darwin 180B RSI | 256k |

**System RAM is part of the recommendation, not an afterthought.** A Strata MoE keeps its experts in host memory, so a band is only real when the RAM column is satisfied too. At 8 GB of VRAM with 32 GiB of RAM, Inkling SQ-10 runs hybrid; the same card with less RAM falls back to Maple. Rows that cannot get the RAM are refused *by name*, naming RAM as the constraint, never silently dropped.

**Ladder 2 — unified / CPU / Apple Silicon** (one shared pool)

| Pool | Default | Also offered |
|---|---|---|
| 8–15 GB | Maple Preview 20B-A1B | — |
| 16–23 GB | Cyber-Tiel Coder 35B-A3B UD-IQ3_XXS | Arete 27B Q3 |
| 24–31 GB | Qwen 3.8 27B GSQ-RCO IQ3_S | Arete 27B Q4 |
| 32–63 GB | **Inkling 2 Mini SQ-20 (lossless)** | Qwen 3.8 27B GSQ-RCO · Arete 27B Q4 |
| 64 GB+ | **Inkling 2 Mini SQ-20 (lossless)** | **Qwen 3.8 Flash Next** · GLM 5.3 Flash Maya-XS |

Strata ships a signed, notarised macOS arm64 Metal engine, so unified memory is a first-class Strata target rather than a llama.cpp-only path. SQ-20 is documented for 32 GiB+ of unified memory, which is where Inkling takes the lead on this ladder too.

**Small MoEs reach small cards.** Flint 40B-A2.5B and LFM2.5 24B-A2B offload measured expert layers to the host, so they are placed at 8–16 GB instead of being excluded: LFM2.5 sends 24 of 38 expert layers to the host at 8 GB, Flint 38 of 47. The tier table and the launcher use the same arithmetic, so a band that is advertised is a band that actually starts.

**The auxiliary keeps your VRAM.** Maple Preview 20B-A1B is the default auxiliary everywhere, and it runs with `-ngl 0`: weights stay memory-mapped on disk and the active experts stream into system RAM (32 GB+) or straight off the SSD below that. Since 40–85% of Hermes tokens go through the auxiliary, that placement is what leaves the whole card for the main model. Ask for GPU layers explicitly and it will take them; the default never claims a card.

**Quantisation policy.** Q4 is a ceiling, not a target: low quants (IQ2/IQ3/GSQ-RCO/UD-) are first-class, `Q8_0`/`F16` are allowed exceptions with a printed reason, and Q5/Q6 need a catalog justification. **IQ3_S is lossless**, so 32 GB and 48 GB stay on it rather than paying for a heavier quant. MTP is on wherever the model supports it; DFlash2 drafters belong to Qwen only, Bonsai keeps its own Prism DSpark sidecar.

**Multimodal — 17 entries, each with the runtime that serves it**

| Modality | Models |
|---|---|
| image | MiniMax H3 Image (diffusers) |
| video | MiniMax H3 (diffusers) |
| music | MiniMax Music 3, ACE-Step 1.5 2B / 4B |
| tts | Qwen3 TTS CustomVoice, Darwin TTS 1.7B, Soprano TTS, PersonaPlex (duplex), Edge TTS |
| stt | NVIDIA Parakeet TDT 0.6B v3, Nemotron ASR 0.6B, faster-whisper, PersonaPlex (duplex) |

`recommend_multimodal()` ranks every modality against the detected hardware and returns `fit`, an action and the reason.

## Tailscale: use any model from any of your devices

Every TurboFit gateway serves its own models **and** advertises its peers'. That is how a phone runs Qwen Flash Next that lives on the machine with the GPU.

```bash
# published automatically at setup; to do it by hand on the node that has the model:
tailscale serve --bg --https=443 http://127.0.0.1:8091
# on any other device on the tailnet:
provider: custom:turbofit   base_url: https://<node>.your-tailnet.ts.net/v1   model: auto
```

Start the local endpoint with `/turbofit serve` if it is not already running (background: `scripts/turbofit-gateway.py`), and prove it from the other device with `/turbofit smoke`. TurboFit never uses Funnel — nothing here needs public ingress, and a peer is only ever reachable across your own tailnet.

- `auto` on a node with no GPU delegates to a healthy peer instead of failing.
- A peer's model is addressable as `peer.ts.net/model-name`; a bare `model-name` resolves to whichever peer serves it.
- A model you name explicitly is **never** silently answered by a hosted API — you get a retryable `503` naming it. Substitution is legal only for `auto`/`active`, and only when `TURBOFIT_ALLOW_API=true`.
- Plain HTTP is honoured only on loopback and Tailscale addresses; anything else needs HTTPS.

## Built on Hermes' own runtime

TurboFit is a plugin *on top of* Hermes, not a parallel stack. Hermes already ships a managed llama.cpp runtime — PM-pinned engines, one supervised router-mode server, a context ladder with growth, a hardware budget with physics checks, MTP launch flags, idle unload. Those are adopted rather than re-implemented:

```bash
scripts/turbofit-hermes-bridge        # 10 of 10 primitives adopted from Hermes
```

What TurboFit adds, because Hermes has no equivalent: external pressure sensing, residency yield and restore, the evidence-gated ladders, the federated gateway, and the benchmark campaigns. Hermes keeps deciding *how* a server lives; TurboFit decides *what* runs and *when it steps aside*.

<!-- turbofit:generated:hermes-index:begin -->

## Intelligence — what actually runs, and how fast

The **Hermes Index** is the mean pass@1 across the four suites Hermes Agent runs — Hermes Bench, TerminalBench 4, TerminalBench Science and SkillsBench — reported with the cost of a task. Local inference has no dollar cost, so the cost column is **seconds per task** at a 1,000-token task, and the suite column always reads *measured / total*: a partial run is never a full index.

| Model | Tier | Context | Aux | Index | Suites | tok/s | s/task @1k | Measurement |
|---|---|---|---|---|---|---|---|---|
| `nanbeige-33b-iq3xs` IQ3_XS | 48 GB | 65,536 | — · `maple-preview-tq2` | 0.0 | 1/4 | 40.74 | 24.55 | screening |
| — | 8 GB, 12 GB, 16 GB, 24 GB, 32 GB, 64 GB, 96 GB, 128 GB, 192 GB, 256 GB, 384 GB | — | — | **no measurements yet** | 0/4 | — | — | — |

A suite that cannot be run here is reported as **unavailable** with its reason rather than as a zero, so `suites` shows exactly how much of the index a row carries. Hermes Bench is run by Nous and is not publicly reproducible; TerminalBench 4, TerminalBench Science and SkillsBench are wired through their own harnesses against a TurboFit-served local endpoint. `s/task @1k` is `1,000 ÷ tok/s`. The `no measurements yet` row lists every tier the campaign has not reached — those tiers stay visible instead of being dropped.

Superseded and not published: `qwen3-8-27b-gsq-rco-iq3s--auto--262k` — scored under the retired `deep-swe`/`agentic-pair` definition, which no longer contributes to any index.

Source: `references/intelligence-scores.json`; regenerate with `tools/turbofit-intel-leaderboard`. Do not hand-edit between the markers.

<!-- turbofit:generated:hermes-index:end -->

## Keeping it current

```bash
scripts/install-self-update            # fast-forward + restart on change, hourly
scripts/install-self-update --dry-run  # print the units first
```

Engine binaries are PM-pinned and never downloaded at boot. TurboFit tracks upstream llama.cpp (stock and atomic forks), Strata, MLX, Prism/DSpark and DFlash2 by version; `references/binary-selection.md` decides stock-vs-fork per model, not per machine.

## Troubleshooting

- **`WinError 10061` on Windows** — connection refused on `127.0.0.1:8091`: nothing is listening there. That port belongs to the **TurboFit gateway**, not the Hermes messaging gateway. Start it (`/turbofit setup`, or `scripts/turbofit-gateway.py`) and confirm with `curl http://127.0.0.1:8091/health`.
- **"no evidence-backed configuration fits"** — your topology has no measured rung yet. Run `scripts/turbofit-catalog-campaign`, or pick from `compatible_lanes` in the `/turbofit` report.
- **A model answers from the cloud** — only `auto`/`active` may cross to an API backend, and only with `TURBOFIT_ALLOW_API=true`. Naming a local model returns a retryable `503` instead of a substituted answer.
- **`vllm flavor needs a checkpoint directory`** — that family is served by vLLM and expects the HF checkpoint directory, not a `.gguf`. Check `references/artifact-manifest.json` for the snapshot digest.
- **Strata refuses to start** — run `scripts/turbofit-strata`. It needs AVX2 (not AVX-512), 12 GB VRAM recommended, and system RAM for the experts; the refusal names which of those is short.
- **Windows specifics** — [`docs/windows-native-install.md`](docs/windows-native-install.md) covers the native install, the 8091 port check and the WinError 10061 path.
- **Still stuck** — install TurboSouth (above) and describe the symptom; it diagnoses and fixes on the machine.

## Docs

- Architecture, authority chain, honest debt list — [`docs/PROJECT-OVERVIEW.md`](docs/PROJECT-OVERVIEW.md)
- How TurboFit maps onto Hermes' built-in local runtime — [`docs/hermes-local-runtime-adapter.md`](docs/hermes-local-runtime-adapter.md)
- Chain data — [`references/fallback-chain.json`](references/fallback-chain.json) · ladders — [`references/fit-lists.json`](references/fit-lists.json) · catalog — [`references/model-catalog.json`](references/model-catalog.json)
- Binary selection — [`references/binary-selection.md`](references/binary-selection.md) · troubleshooting — [`docs/turbofit-check.md`](docs/turbofit-check.md)
- Tailnet peer federation — [`docs/tailnet-federation.md`](docs/tailnet-federation.md)
- Changelog — [`CHANGELOG.md`](CHANGELOG.md) · roadmap — [`ROADMAP.md`](ROADMAP.md) · full wiki topic — [bot-wiki](https://github.com/SouthpawIN/bot-wiki/blob/wiki/wiki-topics/turbofit/index.md)

## Support the work

TurboFit is free, MIT, and always will be — nothing is paywalled and no runtime path checks whether you donated. If it saved you a cloud bill or a weekend of picking quants, these keep the benchmarks running.

[![Ko-fi](https://img.shields.io/badge/Ko--fi-tip-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white)](https://ko-fi.com/sovthpaw)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/sovthpaw)
[![Patreon](https://img.shields.io/badge/Patreon-join-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/sovthpaw)

- ☕ **Ko-fi** — one-off tip, no account needed, 0% platform fee
- 🧋 **Buy Me a Coffee** — one-off or monthly
- 🔧 **Patreon** — monthly support, and a one-on-one live call with me once a month

Star the repo if that's all you want to do — it genuinely helps other people find it.

MIT.

---
> _This README is generated from the [TurboFit wiki](https://github.com/SouthpawIN/bot-wiki/blob/wiki/wiki-topics/turbofit/README.md) — the single source of truth. To change it, edit the wiki hub and re-run `tools/sync-wiki-readme.sh`._
