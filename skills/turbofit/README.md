# Turbofit

## 2.2 new models

| New model | Variants or role | State |
|---|---|---|
| **Qwen 3.8 27B** | Q4, Q8, and BF16; each with or without MTP; pinned vision projectors | 6 active catalog candidates replacing the 6 retired GRM entries |
| **DeepSeek V4 Flash 0731 Q2 DwarfStar** | Q2 main model + DSpark and expert offload | Active high-memory catalog candidate |
| **MiniMax Music 3** | Music generation | Pinned integration candidate |
| **NVIDIA Parakeet TDT 0.6B v3** | Speech-to-text | Pinned integration candidate |
| **Soprano TTS** | Text-to-speech | Pinned integration candidate |

Qwen 3.8 27B is now the active dense multimodal 27B family. No retired GRM recipe or artifact remains active. The 1,620-row campaign remains evidence-gated: a catalog candidate is not a benchmark winner until current-recipe physical and intelligence evidence passes.

The canonical model IDs, source revisions, artifacts, sizes, and hashes are in `references/model-catalog.json`, `references/artifact-manifest.json`, and `references/multimodal-models.json` at the repository root.

Turbofit is a first-class Hermes plugin and adaptive local-model runtime. It scans physical hardware and total usable memory, recommends evidence-backed main/auxiliary configurations, exposes one OpenAI-compatible provider, and manages setup through Hermes Dashboard and Hermes Desktop.

## Install

```bash
hermes plugins install --enable https://github.com/SouthpawIN/Turbofit.git
```

Restart the gateway, then launch guided setup:

```text
/turbofit setup
```

## Daily commands

```text
/turbofit                 # rescan + intelligence/balanced/speed recommendations
/turbofit intelligence    # quality-first recommendation
/turbofit balanced        # context/balance recommendation
/turbofit speed           # throughput-first recommendation
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
