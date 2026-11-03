# Changelog

## 2.4.1 — pressure yield, external-unit rungs, naming

- **Yield under external GPU pressure**: when a card is contended and no cheaper
  rung fits, policy returns `release` and residency is unloaded instead of
  squatting on the card (`external_gpu_priority: absolute` finally enforced on
  single-rung and top-rung profiles). Residency is restored on the same rung
  once the pressure clears; a yielded rung is never reloaded while contended.
- **Unavailable rungs no longer wedge the controller**: startup prunes rungs
  whose artifacts are missing and logs one `skipped_rung` line each. Previously
  one missing model file made every tick fail before pressure was measured.
- **Optional vision**: a missing projector GGUF serves the text model instead of
  failing the rung.
- **Externally managed rungs**: a portable rung may declare
  `launcher: systemd-unit` + `unit: <name>`. Lifecycle is
  `systemctl --user start|stop`, never a signal to a PID TurboFit did not
  launch. Strata can therefore be a real ladder rung that pressure can release.
- **Controller observability**: last tick published to
  `controller-status.json` (`degraded`, `last_error`, `error_count`); an
  identical tick error is logged once a minute instead of every interval.
- **Recipe aliases**: declared `aliases` plus alphanumeric normalization resolve
  hand-written spellings; unknown names fail with the closest candidates
  instead of a bare `unknown model family`.
- **Fit reporting**: a requirement recorded against a device index this box does
  not have is reported as host spill, not as a phantom card.
- **`/turbofit` on unmeasured hardware**: falls back to the best portable-fit
  lane and says so, instead of reporting "no evidence-backed configuration fits"
  while listing lanes that do.
- **TurboSouth rename**: the support profile and its repository are TurboSouth
  (`SouthpawIN/sirvir` → `SouthpawIN/turbosouth`; the old URL redirects). The
  profile id is now `turbosouth` (distribution 2.3.0), and every `sirvir`
  reference is gone from both repos. `TURBOSOUTH_DISTRIBUTION` stays overridable
  with `TURBOFIT_TURBOSOUTH_SOURCE`, and an unreachable distribution fails with
  one actionable sentence before `git clone`.

## 2.4.0 — Two-ladder TurboFit List, Strata engine, federated gateway

- Version surfaces are `2.4.0`.
- **Two-ladder fit doctrine**: dedicated-VRAM (dense) and unified/CPU (MoE) ladders converge on Qwen 3.8 Flash Next. Bonsai 2 27B PTQ1_0/PQ2_0 is the 8 GB rung; 24 GB reaches 262K; DeepSeek V4.1 (1017) is the top rung on the unified ladder (native engine + fit data pending). MTP on at every rung; IQ3_S is lossless, so no heavier quant past it (power bill). Canonical data in `references/fit-lists.json`, `references/fit-ladder.md`, and `references/model-lineup.md`.
- **Strata engine**: systemd user unit installed from the runtime installer; GSQ-RCO recipes and remote backend with pipeline tests.
- **Gateway federation**: local-alias routing plus automatic Tailscale peer discovery (10 s timeout); Tailscale auto-configure at install; gateway installed as a service on macOS and bound for tailnet reachability.
- **SALT-core engine integration**: machine-readable G1–G5 qualification gate with a committed gate report from a real checkout (36 tests).
- **Benchmark leaderboard**: deterministic best model + serve engines per tier from committed references; `/leaderboard` endpoint and dashboard card.
- **Engine registry**: data-driven engine-serve matrix with native-identity checks.
- **`hermes_fit` bridge**: built-in Hermes local-model recommendation — non-blocking, physics-predicted (no live model call, ~40 ms), deterministic, degrades to the allowlist + ladders.
- **hermes_cli discovery** wired (runtime + tests) with the physics-predicted TurboFit List.
- **sovthpaw mascot** bundled into the TurboSouth install (idempotent `hermes pets install s0uthpaw --select`).
- Auxiliary model streams from the hard drive at every tier.
- Cross-OS test suite made deterministic and fresh-install complete.
- Dependency floors aligned to Hermes pins: `fastapi>=0.133.1`, `huggingface_hub>=1.24.0`.
- Privilege auto-escalation removed — plugin security scan verdict is `safe`.
- `after-install.md` deterministic install + verify + recovery path for `not a command: turbofit`.
- Phase G: systemd linger enabled on Linux so user units survive logout/reboot (best-effort).

## 2.3.1 — Windows campaign gate fixes + remote-Mac docs

- Version surfaces are `2.3.1` (the Maple Fit List line stays 2.3; 2.4 is held).
- Windows GPU clear gate is baseline-relative: pre-existing driver residency no longer reads as a leaked Turbofit allocation (absolute safety floor preserved).
- Windows process-tree stop via `taskkill /T`, PowerShell GPU probing, AMD driver inventory via CIM.
- Maple Preview TQ2_0 added to the catalog as the 8 GB Fit List candidate (native 64K/128K).
- Documented the remote-Mac model-server topology (Turbofit on the Mac, Hermes in a Proxmox LXC).

## 2.3 — Maple small-device Fit List

- Version surfaces are `2.3.0`.
- Dedicated VRAM and integrated/RAM-only total memory are separate Fit List dimensions. 8 GB of VRAM is not 8 GB of RAM.
- Dedicated 8 GB main candidate is Maple Preview 20B-A1B TQ2_0. Ornith remains the alternative when host RAM can hold offloaded experts. Bonsai stays in the catalog but is no longer the 8 GB default.
- Shared total memory: Maple at 8–15 GB, Ornith 1.5 35A3B at 16–23 GB, Unleashed UD-Q3_K_XL at 24 GB+.
- Maple is native 64K/128K only and needs the Maple llama.cpp fork. Auto on 8 GB is Maple 128K → 64K.
- README is the whole-product 2.4 surface: Check, List, Desktop, slash, TurboSouth, Tailscale, engines, multimodal. Campaign depth stays in `docs/`.
- Posters use the existing gold/mint starfield language.
- Desktop shows a gold **New model recommended** card when Check suggests a replacement and the old weights are still installed: Keep both, Archive old model, or Delete old model.
- Auto on `hardware-8gb` is Maple TQ2_0 at 128K, contracting to Maple 64K. Bonsai is no longer the 8 GB Auto floor.
- Check now auditions llama.cpp, MLX, SGLang, vLLM, FreeToken, and Turbohaul Manager against the selected model pair. Maple GGUF is fork-only; vLLM/SGLang stay HF/FP8/NVFP4.
- Compatible lanes are Fit List mains only. 8 GB VRAM + any host RAM is Maple or Ornith. Host-spill of a dense 27B is not a recommendation.
- Default aux is Ornith oQ4e MTP at 40 GB+ total VRAM and oQ8e MTP at 64 GB+. Speed preference ranks Ornith over dense Qwen 27B. Unleashed stays the intelligence pick on 16/24 GB.
- Unleashed Qwen 27B now serves with DFlash2 (`n_max=7`) on the pinned z-lab fork. Ornith oQ is MTP. Spec decode is the default where a draft exists.
- Desktop opens with speed/VRAM/RAM graphs, the live scale-down/up ladder, subscriptions, and Nous keyless free models instead of fallback JSON.

## 2.3 — Unleashed / Ornith / Nous free fallback

### Setup / troubleshooting

- `/turbofit` now enables itself in every Hermes profile home (including TurboSouth). Desktop was reporting `not a quick/plugin/bundle/skill command: turbofit` because standalone plugins are opt-in and profile sessions do not see `~/.hermes/plugins`.
- `/turbofit update` pulls the plugin and TurboSouth onto the current device and refreshes the Desktop surface.
- `/turbofit shift up|down|<model>|intelligence|balanced|speed` walks the same measured ladder the healer uses.
- `/turbofit serve` publishes `:8091` on Tailscale Serve so other tailnet devices can use the same local model server. Funnel is never used.
- Setup downloads the Auto-chain artifacts for this machine if they are missing. TurboSouth owns install and setup, with a bootstrap fallback only until `:8091` answers.
- Private-LAN HTTP (`192.168.x`) is a valid Turbofit provider URL. Windows scheduled tasks accept Vulkan and `-GatewayHost 0.0.0.0` for remote Hermes.


- Documented the three gateways: Hermes messaging vs Turbofit `:8091` vs native model server. Plugin install and messaging-gateway restart do not start `:8091`.
- `/turbofit setup` refreshes Hermes Desktop. Headless Windows path is linked from README to `docs/windows-native-install.md`.
- TurboSouth GitHub install now uses `https://github.com/SouthpawIN/turbosouth.git` so Hermes 0.20.0 on Windows can resolve it.
- Support text now forbids diagnosing a dead `:8091` from inside TurboSouth.

### Recommendations

- Auto/Check will never recommend a 9B. Low-VRAM dedicated devices get Bonsai 27B and/or Ornith 1.5 35A3B. Ornith experts mmap from disk when host RAM is under 32 GB. Apple Silicon uses OrcaRouter Uncensored MLX 4/6/8-bit (never 2-bit). Integrated/unified non-Apple uses Ornith 1.5.

### Model runtime

- Re-ran the canonical Qwen 3.8 Q4 baseline/DFlash2 pair on the same dual-RTX-3090 fingerprint: 35.79 versus 45.60 tok/s, 1.273993× speedup, 70% draft acceptance, and 2,839 MiB additional aggregate peak GPU residency. Added a deterministic exporter that refuses mismatched hardware, context, prompt shape, raw hashes, or invalid acceptance counters before emitting self-hashed A/B evidence.
- Added a separate evidence record for the reported Spark/SGLang Qwen 3.8 27B NVFP4 + DFlash2 lane: 116 aggregate output tok/s at concurrency 10 and 262K configured context. It remains non-promotable until raw harness, exact topology, immutable image digest, workload/latency details, acceptance statistics, and a same-runtime baseline are attached.
- Added an evidence-gated Qwen 3.8 27B DFlash2 candidate: pinned Inco Q4_K_M draft artifact, pinned z-lab llama.cpp PR runtime, 64K/128K/262K/1M recipes, and a hard no-cross-family guard preserving Bonsai's own DSpark sidecar.
- Removed DeepSeek V4 Flash 0731 from the active catalog, recipes, downloads, and hardware-tier candidates.
- Main Auto chain is now Bonsai 27B at 8GB, Qwen 3.8 27B Unleashed UD-IQ3_XXS at 16GB, UD-Q3_K_XL at 24–95GB, and the existing Qwen 3.8 27B 16-bit recipe at 96GB+ until an Unleashed FP16 GGUF is published.
- Auxiliary authority is Ornith 1.5 35A3B, optional Carwin Nano, or auto. Ornith is MoE: scale-down offloads experts, then lowers context, then switches aux to auto, then a listed Unleashed quant, then Bonsai.
- Replaced the API fallback chain with the five looked-up Nous free/keyless models from the current Hermes-Agent catalog: `stealth/ox-alpha` (Ox Alpha, $0/$0 curated), `stepfun/step-3.7-flash:free`, `tencent/hy3:free`, `poolside/laguna-s-2.1:free`, `poolside/laguna-xs-2.1:free`. No Hermes-branded models. No NVIDIA NIM.

### Hermes plugin install

- Repaired the discovery/benchmark/List pipeline: current-hardware tournament priority, non-collapsing real-suite intelligence composites, model-call/token/route evidence gates, discovery-to-onboarding queue, exact-tier winner promotion, generated TurboFit List, and TurboFit Check scan-to-configuration terminology.
- Added pinned FreeToken 0.1.2 (`0ab982f…`) as an optional Linux x86_64 / NVIDIA driver 580+ / CUDA toolkit 13+ text-only MoE candidate, with setup controls, loopback command/client contracts, compatibility gating, and no Auto promotion before exact physical evidence.
- Turbofit's TurboSouth setup option now installs or updates the canonical `SouthpawIN/turbosouth` GitHub profile instead of copying a bundled snapshot; TurboSouth's bootstrap reciprocally installs Turbofit first when it is missing.
- Slimmed `distribution.yaml` so plugin install no longer ships benchmark result dumps as payload.

## 2.2 — Qwen 3.8 27B day-zero release

### Model runtime

- Replaced all six active GRM 2.6 27B variants with Qwen 3.8 27B Q4, Q8, and BF16 variants, each available with or without MTP.
- Pinned the official `Qwen/Qwen3.8-27B` upstream at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and the canonical `ggml-org/Qwen3.8-27B-GGUF` conversion at `0669b98607d47046c7c2b3f801011d54a08cfccf`.
- Added SHA-256-pinned main GGUFs, Q8/BF16 vision projectors, and Q4/Q8/BF16 MTP sidecars.
- Fixed the MTP compiler to emit `--model-draft` for separately published MTP artifacts; regression tests now verify draft and projector wiring.
- Added the executable DeepSeek V4 Flash 0731 `UD-Q2_K_XL` DwarfStar variant and expanded the exhaustive matrix to 1,620 rows.
- Made native allocation portable across dedicated VRAM + host RAM, unified-memory accelerators, Vulkan/ROCm/Metal, and CPU-only hosts. Large contexts explicitly place KV cache in usable host RAM when needed; shared pools are counted once; CPU/shared pressure no longer requires an NVIDIA inventory.
- Physically validated Qwen 3.8 27B Q4+MTP at a 1,048,576-token configured context on dual RTX 3090 plus host RAM: 0.51 tok/s, 81.82% MTP acceptance, 23,471/23,253 MiB peak VRAM.

### Catalog and release surfaces

- Removed GRM from active recipes, downloads, runtime profiles, successful-profile promotion state, campaign deferrals, and tier candidates while preserving historical result evidence as retired records.
- Added Qwen 3.8 27B to the 24, 48, 64, 96, and 200 GB physical tournament bands without inventing winners or inheriting stale GRM evidence.
- Added pinned MiniMax Music 3, Soprano TTS, and Parakeet TDT 0.6B v3 multimodal candidates with honest candidate status.
- Updated plugin, Dashboard, distribution, root skill, bundled skill, README, and changelog surfaces to 2.2.0.
- Generated a purpose-built Qwen 3.8 27B release commercial with grounded release statistics.

## 2.1 — DeepSeek 0731 and adaptive multimodal release

### Model runtime

- Replaced preview-era DeepSeek references with the official `deepseek-ai/DeepSeek-V4-Flash-0731` upstream checkpoint, pinned at commit `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Pinned the matching Unsloth `UD-Q8_K_XL` and `UD-Q4_K_XL` GGUFs and the extracted 0731 DSpark sidecar at commit `fbbb5b93fb787c21338159b0af3318bb3f4d9768`.
- Upgraded the native runtime to llama.cpp b10269, avoiding the documented b10259–b10268 DSpark loader regression window.
- Added SHA-pinned PrismML and ik_llama.cpp runtimes for custom Bonsai/Ternary and GLM 5.2 quantization/architecture support; production recipes now select the artifact-compatible runtime instead of misclassifying custom GGUFs as corrupt.
- Added explicit `draft-dspark`, three-token speculative depth, safe draft placement, bounded batch/u-batch prefill, Jinja tool calling, expert offload, and 64K/128K/262K/1M recipes.
- Generated all 1,584 valid main/add-on/auxiliary/context research combinations.

### Adaptation and recommendations

- Recommendations now fit against safely usable total memory, including host-memory spill, CPU-only systems, and unified-memory systems without double counting.
- Added intelligence, balanced, and speed preference modes.
- Standardized the promotion ladder: 128K → 30 tok/s → 262K → 50 tok/s → 1M → fastest.
- Added role-specific ports and strict readiness checks to prevent main/auxiliary false positives.
- Consolidated runtime classification to the exact eight required tiers: 8, 16, 24, 48, 64, 96, 200, and 300+ GB. Removed superseded 128/256/512 GB duplicates and marked every ladder evidence-gated until its current production/validation recipe SHA-256 is physically revalidated.

### Hermes integration

- Added first-class Hermes plugin tools and `/turbofit` plus `/turbofit setup` behavior.
- Added guided Dashboard and native Hermes Desktop management.
- Added primary provider, ordered credential-free fallback-chain, private Tailscale Serve, TurboSouth, native runtime, Lemonade, and multimodal setup controls.
- Rewrote TurboSouth as Turbofit customer service and PR-suggestion support.

### Multimodal

- Added hardware-aware image, video, music, TTS, and STT recommendations.
- Added MiniMax H3, ACE-Step 1.5, Parakeet, Nemotron ASR, Darwin TTS, Soprano, and Hermes-native options.
- Physically generated the six-style 2.1 promo locally through pinned MiniMax H3 INT8 streamed host offload and preserved prompts, seeds, clips, timings, narration, media probes, and checksums.
- Candidate adapters are labeled honestly and are never represented as built-in Hermes providers.

### Research integrity

- Artifact destinations are revision/SHA pinned and path-confined.
- Archived GLM, MiniMax M3, and Laguna configurations remain benchmarkable but cannot be promoted as current recommended winners.
- Physical evidence is retained only when its evidence file exists and the exact runtime/model checks pass.
- Added immutable per-attempt failure evidence, canonical production-recipe hashes, and automatic stale-evidence requeue when any runtime, artifact, context, topology, or launch argument changes.
- The exhaustive campaign is resumable and remains explicitly in progress until every valid row has current evidence.
- Added a separately resumable 1,584-configuration intelligence campaign with screening, promotion, and release levels.
- Pinned DeepSWE to commit `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9` and added a joint main/auxiliary agentic production harness.
- Replaced catalog-derived capability points with evidence-only DeepSWE/agentic geometric intelligence scores; missing scores remain pending.
- Invalidated four zero-call DeepSWE records caused by PIER container-to-host loopback routing; infrastructure-invalid trials can no longer become zero intelligence scores, and PIER now receives a container-reachable production endpoint.
- Added all-tier CLI, plugin, and Dashboard surfaces separating artifact storage, host RAM status, aggregate/per-device accelerator memory, topology, quantization/offload, inferred fit, physical fit, intelligence, and TPS.
