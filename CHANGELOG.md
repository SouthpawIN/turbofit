# Changelog

## 2.3 — Unleashed / Ornith / Nous free fallback

### Recommendations

- Auto/Check will never recommend a 9B. Low-VRAM dedicated devices get Bonsai 27B and/or Ornith 1.5 35A3B. Ornith experts mmap from disk when host RAM is under 32 GB. Apple Silicon uses OrcaRouter Uncensored MLX 4/6/8-bit (never 2-bit). Integrated/unified non-Apple uses Ornith 1.5.

### Model runtime

- Removed DeepSeek V4 Flash 0731 from the active catalog, recipes, downloads, and hardware-tier candidates.
- Main Auto chain is now Bonsai 27B at 8GB, Qwen 3.8 27B Unleashed UD-IQ3_XXS at 16GB, UD-Q3_K_XL at 24–95GB, and the existing Qwen 3.8 27B 16-bit recipe at 96GB+ until an Unleashed FP16 GGUF is published.
- Auxiliary authority is Ornith 1.5 35A3B, optional Carwin Nano, or auto. Ornith is MoE: scale-down offloads experts, then lowers context, then switches aux to auto, then a listed Unleashed quant, then Bonsai.
- Replaced the API fallback chain with the five looked-up Nous free/keyless models from the current Hermes-Agent catalog: `stealth/ox-alpha` (Ox Alpha, $0/$0 curated), `stepfun/step-3.7-flash:free`, `tencent/hy3:free`, `poolside/laguna-s-2.1:free`, `poolside/laguna-xs-2.1:free`. No Hermes-branded models. No NVIDIA NIM.

### Hermes plugin install

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
