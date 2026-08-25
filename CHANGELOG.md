# Changelog

## 2.4 — Maple small-device Fit List

- Version surfaces are `2.4.0`.
- Dedicated VRAM and integrated/RAM-only total memory are separate Fit List dimensions. 8 GB of VRAM is not 8 GB of RAM.
- Dedicated 8 GB main candidate is Maple Preview 20B-A1B TQ2_0. Ornith remains the alternative when host RAM can hold offloaded experts. Bonsai stays in the catalog but is no longer the 8 GB default.
- Shared total memory: Maple at 8–15 GB, Ornith 1.5 35A3B at 16–23 GB, Unleashed UD-Q3_K_XL at 24 GB+.
- Maple is native 64K/128K only and needs the Maple llama.cpp fork. Auto on 8 GB is Maple 128K → 64K.
- README is the whole-product 2.4 surface: Check, List, Desktop, slash, Sirvir, Tailscale, engines, multimodal. Campaign depth stays in `docs/`.
- Posters use the existing gold/mint starfield language.
- Desktop shows a gold **New model recommended** card when Check suggests a replacement and the old weights are still installed: Keep both, Archive old model, or Delete old model.
- Auto on `hardware-8gb` is Maple TQ2_0 at 128K, contracting to Maple 64K. Bonsai is no longer the 8 GB Auto floor.
- Check now auditions llama.cpp, MLX, SGLang, vLLM, FreeToken, and Turbohaul Manager against the selected model pair. Maple GGUF is fork-only; vLLM/SGLang stay HF/FP8/NVFP4.
- Compatible lanes are Fit List mains only. 8 GB VRAM + any host RAM is Maple or Ornith. Host-spill of a dense 27B is not a recommendation.

## 2.3 — Unleashed / Ornith / Nous free fallback

### Setup / troubleshooting

- `/turbofit` now enables itself in every Hermes profile home (including Sirvir). Desktop was reporting `not a quick/plugin/bundle/skill command: turbofit` because standalone plugins are opt-in and profile sessions do not see `~/.hermes/plugins`.
- `/turbofit update` pulls the plugin and Sirvir onto the current device and refreshes the Desktop surface.
- `/turbofit shift up|down|<model>|intelligence|balanced|speed` walks the same measured ladder the healer uses.
- `/turbofit serve` publishes `:8091` on Tailscale Serve so other tailnet devices can use the same local model server. Funnel is never used.
- Setup downloads the Auto-chain artifacts for this machine if they are missing. Sirvir owns install and setup, with a bootstrap fallback only until `:8091` answers.
- Private-LAN HTTP (`192.168.x`) is a valid Turbofit provider URL. Windows scheduled tasks accept Vulkan and `-GatewayHost 0.0.0.0` for remote Hermes.


- Documented the three gateways: Hermes messaging vs Turbofit `:8091` vs native model server. Plugin install and messaging-gateway restart do not start `:8091`.
- `/turbofit setup` refreshes Hermes Desktop. Headless Windows path is linked from README to `docs/windows-native-install.md`.
- Sirvir GitHub install now uses `https://github.com/SouthpawIN/sirvir.git` so Hermes 0.20.0 on Windows can resolve it.
- Support text now forbids diagnosing a dead `:8091` from inside Sirvir.

### Recommendations

- Auto/Check will never recommend a 9B. Low-VRAM dedicated devices get Bonsai 27B and/or Ornith 1.5 35A3B. Ornith experts mmap from disk when host RAM is under 32 GB. Apple Silicon uses OrcaRouter Uncensored MLX 4/6/8-bit (never 2-bit). Integrated/unified non-Apple uses Ornith 1.5.
- MiniMax H3 is no longer a recommendation-only row. Selecting it writes `turbofit.h3-launch/v1` and other machines must pass `scripts/verify-h3-live --smoke`. VAE stays float32 on the generation device; decode latents are aligned. That is the failure that blocked the 2.3 promo.

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
- Turbofit's Sirvir setup option now installs or updates the canonical `SouthpawIN/sirvir` GitHub profile instead of copying a bundled snapshot; Sirvir's bootstrap reciprocally installs Turbofit first when it is missing.
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
- Added primary provider, ordered credential-free fallback-chain, private Tailscale Serve, Sirvir, native runtime, Lemonade, and multimodal setup controls.
- Rewrote Sirvir as Turbofit customer service and PR-suggestion support.

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
