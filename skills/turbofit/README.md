# Turbofit

## 2.3 new models

| New model | Variants or role | State |
|---|---|---|
| **Qwen 3.8 27B Unleashed** | UD-IQ3_XXS, UD-Q3_K_XL | Active catalog candidates |
| **Ornith 1.5 35A3B** | Q4_K_M MoE auxiliary | Active auxiliary candidate |
| **Carwin Nano** | Optional MoE auxiliary | Optional auxiliary candidate |
| **Qwen 3.8 27B** | Q4, Q8, and BF16; each with or without MTP | Active catalog candidates |
| **MiniMax Music 3** | Music generation | Pinned integration candidate |
| **NVIDIA Parakeet TDT 0.6B v3** | Speech-to-text | Pinned integration candidate |
| **Soprano TTS** | Text-to-speech | Pinned integration candidate |

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

Finish Apply on the Turbofit page, then verify `http://127.0.0.1:8091/v1/models` before opening Sirvir. Do not diagnose a refused 8091 from inside Sirvir, and do not restart the Hermes messaging gateway expecting the model endpoint to appear.

When Dashboard/slash is unavailable on Windows, use [`docs/windows-native-install.md`](../../docs/windows-native-install.md).

## Daily commands

```text
/turbofit                 # rescan + intelligence/balanced/speed recommendations
/turbofit update          # update Turbofit plugin + Sirvir on this device
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

Full architecture, setup, model zoo, evidence policy, and repository map: [`../../README.md`](../../README.md).
