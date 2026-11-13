# Turbofit

## Current catalog — 89 entries, 86 main models, 4 engines

Serves each model on the runtime that can hold it: llama.cpp (mainline, and the Prism fork for ternary), **Strata** for large MoE on small VRAM, **vLLM** for raw checkpoints with no GGUF, and per-model media runtimes. Every entry pins a Hugging Face revision and a sha256.

| Model | Variants or role | State |
|---|---|---|
| **Qwen 3.6 35B-A3B** | UD-Q4_K_M + MTP, 64K/128K/262K | Active catalog candidate |
| **Qwen 3.5 35B-A3B** | UD-Q4_K_M + MTP, 64K/128K/262K | Active catalog candidate |
| **LFM2.5 24B-A2B** | Q4_K_M, 64K/128K | Active catalog candidate |
| **LFM3 Omni 2B** | Q4_K_M + vision projector | Active catalog candidate |
| **Nemotron Bolt 74B-A9B** | 24GB native, MoE + vision | Active catalog candidate |
| **Inkling 2 Small** | 366.5 GiB checkpoint, **vLLM**, 1M context | Engine-wired; needs vLLM installed |
| **Maple Preview 20B-A1B** | TQ2_0 8 GB main candidate | CPU 8 GiB envelope verified; exact 8 GB GPU pending |
| **Qwen 3.8 27B GSQ-RCO** | IQ3_S → Q8, with or without MTP | Active catalog candidate |
| **Ornith 1.5 35A3B** | Q4_K_M MoE auxiliary | Active auxiliary candidate |
| **Carwin Nano** | Optional MoE auxiliary | Optional auxiliary candidate |
| **MiniMax Music 3** | Music generation (diffusers-modular) | Pinned integration candidate |
| **NVIDIA Parakeet TDT 0.6B v3** | Speech-to-text | Pinned integration candidate |
| **Soprano TTS** | Text-to-speech | Pinned integration candidate |

Uncensored line, same rules: Arete 27B Uncensored V2, Qwen 3.8 27B Unleashed, Qwen 3.6 35B-A3B Abliterated, LFM2.5 24B-A2B Uncensored, Flint 40B Abliterated, Nemotron Bolt Abliterated, Flash Next Abliterated.

DeepSeek V4 Flash 0731 is retired. API fallback is the five keyless Nous free models. Catalog candidates are not benchmark winners until current-recipe physical and intelligence evidence passes.

The canonical model IDs, source revisions, artifacts, sizes, and hashes are in `references/model-catalog.json`, `references/artifact-manifest.json`, and `references/multimodal-models.json` at the repository root.

Turbofit is a first-class Hermes plugin and adaptive local-model runtime. It scans physical hardware and total usable memory, recommends evidence-backed main/auxiliary configurations, exposes one OpenAI-compatible provider, and manages setup through Hermes Desktop.

## Install

```bash
hermes plugins install --enable https://github.com/SouthpawIN/Turbofit.git
```

Restart Hermes Desktop or start a fresh **default-profile** CLI session so plugin registrations reload. `/turbofit setup` refreshes the Desktop Turbofit page; it does not start `:8091` by itself.

```text
/turbofit setup
```

Finish Apply on the Turbofit page, then verify `http://127.0.0.1:8091/v1/models` before opening TurboSouth. Do not diagnose a refused 8091 from inside TurboSouth, and do not restart the Hermes messaging gateway expecting the model endpoint to appear.

When Dashboard/slash is unavailable on Windows, use [`docs/windows-native-install.md`](../../docs/windows-native-install.md).

## Daily commands

```text
/turbofit                 # rescan + intelligence/balanced/speed recommendations
/turbofit update          # update Turbofit plugin + TurboSouth on this device
/turbofit shift up        # next smarter measured configuration
/turbofit shift down      # next lighter measured configuration
/turbofit shift <model>   # recommended combination for that model
/turbofit intelligence    # quality-first recommendation (scan)
/turbofit balanced        # context/balance recommendation (scan)
/turbofit speed           # throughput-first recommendation (scan)
/turbofit status          # provider, route, runtime, and gateway state
```

Repository operators can use:

```bash
scripts/turbofit-runtime list
scripts/turbofit-runtime set auto
scripts/turbofit-runtime status
scripts/turbofit-controller --once
curl -fsS http://127.0.0.1:8091/v1/models
```

## Setup surfaces

Dashboard and Desktop expose:

- physical hardware and total usable memory;
- auto/manual profile selection;
- primary provider and complete ordered fallback chain;
- runtime and native Desktop installation;
- multimodal recommendations and selections;
- benchmark campaign status and evidence.

## Research matrix

```bash
scripts/build-exhaustive-model-matrix
scripts/expand-multipart-artifacts
PYTHONPATH=src:. scripts/turbofit-catalog-campaign status
PYTHONPATH=src:. scripts/turbofit-catalog-campaign run
```

The generated campaign covers all valid pinned main/add-on combinations, every auxiliary option, and 64K/128K/262K/1M context. Candidate rows remain candidates until physical evidence passes.

## Verification

```bash
PYTHONPATH=src:. python3 -m pytest -q
node --check dashboard/dist/index.js
node --check desktop/plugin.js
scripts/release-check
```

Full architecture, setup, model zoo, evidence policy, and repository map: [README.md](README.md).

## Supported Models

- **Maple Preview 20B-A1B**
- **Qwen 3.8 27B GSQ-RCO**
- **Ornith 1.5 35A3B**
- **MiniMax Music 3**
- **NVIDIA Parakeet TDT 0.6B v3**
- **Soprano TTS**
