# Runtime backends

Turbofit keeps provider routes stable while the process runtime is replaceable. The controller and Hermes provider do not depend on a specific accelerator vendor.

## Backend matrix

| Backend | Discovery | Process isolation | Installer | Live evidence in this repository |
|---|---|---|---|---|
| NVIDIA CUDA | `nvidia-smi` UUID and PCI topology | `CUDA_VISIBLE_DEVICES` | `scripts/install-dspark-runtime --backend cuda` | Required by Linux release acceptance |
| AMD ROCm | `rocm-smi` JSON UUID, PCI, and VRAM | `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` | `scripts/install-dspark-runtime --backend rocm` | Unit-tested; requires AMD hardware for a live acceptance record |
| Apple Metal | unified-memory inventory | native launchd service | `scripts/install-macos-native-service` | Installer and resource sampling are tested |
| Windows | user-scoped Scheduled Task | CUDA, Vulkan, or CPU | `scripts/install-windows-native-service.ps1` | Installer is statically tested; requires Windows for live acceptance |
| Lemonade | OpenAI-compatible `/api/v1` probe | pinned native loopback service | `scripts/install-lemonade-runtime` | `/api/v1/health` and `/api/v1/models` are checked live |
| CPU | host process | no accelerator visibility variable | `scripts/install-dspark-runtime --backend cpu` | Supported as a correctness/fallback path, not a performance ceiling |

## Pinned DSpark-capable llama.cpp

```bash
scripts/install-dspark-runtime install --backend cuda
scripts/install-dspark-runtime check --backend cuda
```

The installer checks out pinned upstream revision `1c3c9674de4d455f1e571bed808252af54932767` (llama.cpp `b10269`), the first recommended release after the `b10259`–`b10268` DeepSeek V4 DSpark loader regression window. It applies the repository's long-context patch, builds a static `llama-server`, and verifies both `draft-dspark` and `--jinja` in the executable.

ROCm uses the same source and verifier:

```bash
scripts/install-dspark-runtime install --backend rocm --amdgpu-targets gfx1100
```

Use the GFX target for the physical AMD GPU. The generated process environment never mixes CUDA and ROCm visibility variables.

## Qwen 3.8 Q4 plus DFlash2

The active Qwen 3.8 main, projector, and DFlash2 sidecar are revision-pinned and SHA-256-pinned in `runtime-profiles/downloads.json`:

```bash
scripts/download-models.py \
  --group qwen3-8-27b-q4-dflash2 \
  --base-dir "$HOME/Models/storage/gguf" \
  --receipt references/results/qwen38-download-verification.json
```

The launch recipe uses the pinned DFlash2 llama.cpp fork, Qwen Q4_K_M main model, matching vision projector, DFlash2 sidecar, `--spec-type draft-dflash`, `--spec-draft-n-max 7`, automatic CPU/GPU fit, and `--jinja`. Bonsai remains on its own family-matched DSpark sidecar; speculative sidecars are never shared across model families.

## Lemonade

The installer downloads the official Lemonade 11.5.1 embeddable release for Linux x64/ARM64, macOS ARM64, or Windows x64, verifies its platform-specific SHA-256, extracts it into the user's Turbofit runtime directory, and binds `lemond` to loopback:

```bash
PYTHONPATH=src:. scripts/install-lemonade-runtime install
PYTHONPATH=src:. scripts/install-lemonade-runtime check
```

The client adapter supports health, model listing, load, and unload through `http://127.0.0.1:13305/api/v1`. Existing runtime directories are verified and never silently replaced. Linux uses a user-scoped systemd service; other hosts use a user-owned native process. AMD users can install the ROCm backend through Lemonade itself:

```bash
lemonade backends install vllm:rocm
```

## Benchmarking and evidence

- `scripts/turbofit-catalog-campaign` runs the canonical generated main × auxiliary × context matrix (currently 516 rows), including all valid pinned Qwen 3.8, Ternary Bonsai, and Binary Bonsai add-on combinations.
- `scripts/turbofit-deepswe` prepares a pinned DeepSWE/Pier checkout and runs selected finalists against an OpenAI-compatible endpoint.
- `research/discover_external_benchmarks.py` binds imported DeepSWE evidence to the source artifact SHA-256. External evidence informs discovery but cannot promote a local runtime profile.
- `scripts/turbofit-completion-check` reports every executable completion gate without converting a failed or missing live run into a success claim.
