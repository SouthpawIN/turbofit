# Turbofit

![Turbofit — unified backend, amber and mint aesthetic](assets/turbofit-hero.png)

<p align="center">
  <a href="https://x.com/vSouthvPawv/status/2081193435172618461?s=20"><strong>▶ Watch the 53-second Turbofit overview</strong></a>
</p>

**A local-model provider for Hermes Agent that fits itself around the way you use your computer.**

Turbofit detects the machine's physical accelerator topology, selects a local main/auxiliary model ladder, downloads the required artifacts through [Turbohaul Manager](https://github.com/MrTrenchTrucker/turbohaul-manager), and exposes stable OpenAI-compatible model IDs to Hermes Agent.

When another program needs VRAM, Turbofit contracts one step at a time: it can remove a dedicated auxiliary residency, route auxiliary work through the main model, reduce context, or move to a smaller local model. After memory remains available long enough, it heals back toward the selected ceiling.

> **Current scope is local-only.** Turbofit does not select, configure, or fall back to API models. API orchestration is recorded under [Later development](#later-development), not presented as a current feature.

## What works now

- **Hermes provider:** one local OpenAI-compatible endpoint with stable `auto`, `active:main`, and `active:aux` IDs.
- **Hardware-aware auto selection:** canonical profiles from 8 GB through 300+ GB; no 48 GB runtime special case.
- **Manual profile selection:** pin the adaptive ceiling to a compatible hardware profile.
- **Managed model acquisition:** first activation pulls missing GGUF artifacts from pinned Hugging Face commits, verifies SHA-256, deduplicates shared blobs, installs Turbohaul manifests, and verifies the final tags before inference.
- **Adaptive local scaling:** contraction dwell, expansion dwell, hysteresis, cooldown, rollback, and flap quarantine.
- **External workload priority:** Turbofit never kills or signals games, editors, renderers, or other GPU consumers.
- **Verified publication:** a new route is published only after its local model rung loads and passes verification.
- **Bounded auxiliary lifecycle:** `active:aux` forwards SSE frames immediately, propagates client disconnect cancellation through Turbohaul to llama.cpp, disables hidden-thinking by default, and caps each generation at 4,096 tokens unless explicitly configured otherwise.
- **Portable configuration:** Turbofiles contain no credentials, machine-local paths, mutable process state, or embedded model binaries.

## Hardware tiers

![Turbofit auto-fit hardware ladder: one setting for every tier](assets/turbofit-tier-list.png)

The graphic shows the full product ladder: hardware tiers, context targets, auxiliary choices, pressure response, and the model families moving through promotion. The table below distinguishes the local ladders available now from higher-context and broader-model combinations still awaiting evidence.

Turbofit ships local-only profiles for these physical classes:

| Class | Canonical topology | Current local ladder |
|---:|---|---|
| 8 GB | `1x8` | Bonsai 27B 1-bit, 64K shared main/aux floor |
| 16 GB | `1x16` | Bonsai 27B 1-bit: 262K → 128K → 64K |
| 24 GB | `1x24` | GRM 2.6 Plus 128K → Bonsai 262K → 128K → 64K |
| 48 GB | `2x24` | GRM 2.6 Plus 262K shared main/aux → GRM 128K → Bonsai floors |
| 64 GB | `2x32` | same verified dual-model ladder with additional headroom |
| 96 GB | `4x24` | same verified dual-model ladder; unused cards remain available to other work |
| 200 GB | `2x100` | same verified dual-model ladder while larger candidates are promoted |
| 300+ GB | `3x100` | same verified dual-model ladder; unused cards remain available to other work |

The current dual-24 GB production ceiling routes both `active:main` and `active:aux` to the same GRM 2.6 Plus 262K runtime. Stage-v1 passed with 100% quality, 100% context retrieval, 3.823 effective end-to-end tok/s, and 12,369 / 15,437 MiB peak GPU use (`sha256:2dd320671f7f891ede49a22820754fa657335581ed1272a3444fe45af9426223`). A separate live 256-token probe reported 55.885 llama-server decode tok/s; effective throughput includes gateway and manager orchestration.

Profiles are selected from physical capacity, not transient free VRAM. Larger cards can satisfy a smaller per-card envelope when card count and topology shape remain compatible; `1x48` is still not treated as `2x24`.

The Bonsai floor was measured on the current benchmark host. The 8/16/64/96/200/300 class mappings are portable recommendations, not claims of completed benchmarking on every accelerator family. Activation remains fail-closed if Turbohaul cannot load or verify a selected rung.

## Selection and model downloads

```bash
# Show local profiles, rungs, and compatibility with this machine
scripts/turbofit-runtime list

# Let Turbofit choose from physical hardware
scripts/turbofit-runtime set auto

# Or select a compatible local profile explicitly
scripts/turbofit-runtime set hardware-16gb

# Inspect the persisted selection
scripts/turbofit-runtime status

# Run one controller reconciliation
scripts/turbofit-controller --once

# Optional persistent user service
scripts/install-controller-service --start
```

A new selection starts at its smallest local floor—not at an API fallback. Before that floor is published, the controller:

1. Resolves every model tag required by the rung.
2. Checks Turbohaul's installed tags and content digests.
3. Pulls each missing Hugging Face artifact from an exact commit.
4. Requires the downloaded SHA-256 to match the acquisition catalog.
5. Reuses a verified blob when several model tags share it.
6. Installs the context/runtime manifest for each tag.
7. Loads and verifies the selected local rung.
8. Atomically publishes the stable routes.

Acquisition recipes live in `runtime-profiles/acquisitions.json`. Model lifecycle authority remains in Turbohaul Manager; Turbofit does not create a second model store.

## Hermes Agent provider

Run the Turbofit gateway at `http://127.0.0.1:8091`, then configure one provider:

```yaml
custom_providers:
  - name: turbofit
    base_url: http://127.0.0.1:8091/v1
    api_key: not-needed
    api_mode: chat_completions
    models:
      auto: {}
      active:main: {}
      active:aux: {}

model:
  provider: custom:turbofit
  default: auto
```

Stable model IDs:

| ID | Meaning |
|---|---|
| `auto` | current selected main route |
| `active:main` | current main residency |
| `active:aux` | dedicated auxiliary when present, otherwise shared main |

The IDs stay constant while the controller changes the backing local model and context.

## Install the Hermes plugin

```bash
hermes plugins install SouthpawIN/turbofit --enable
```

Restart Hermes, then open the setup screen:

```bash
hermes dashboard
```

Select **Turbofit** in the dashboard. The plugin can scan/select a compatible
hardware profile, register the `custom:turbofit` endpoint, set `auto` as the
primary model, add or remove Turbofit from the canonical
`fallback_providers` chain, and publish both provider and dashboard endpoints
privately with Tailscale Serve. Tailnet publishing defaults to separate HTTPS
ports and never exposes a public Funnel route. It also registers `/turbofit`,
`turbofit_status`, and `turbofit_configure` for CLI and gateway sessions.

The setup screen installs **Sirvir** by default as a separate Hermes customer-
service profile. Sirvir helps users install, configure, use, and troubleshoot
Turbofit, and turns recurring support cases into evidence-backed pull request
suggestions. Updates replace only Sirvir's distribution-owned `SOUL.md`,
`AGENTS.md`, `config.yaml`, and manifest; profile memories and user state are
preserved. Disable the checkbox if the profile is not wanted.

Linux/WSL2 NVIDIA launches use CUDA. Apple Silicon is detected as one unified
Metal device; the portable 8/16/24 GB profiles compile Docker-only Bonsai
recipes to native `llama-server` processes with Metal enabled. Every generated
launch path includes `--jinja` for tool-call templates.

Development should happen from a Git checkout, not the installed plugin directory.

Current requirements:

- Python 3.11+
- Hermes Agent
- Turbohaul Manager v0.7
- PyYAML for YAML Turbofiles
- a supported local runtime/accelerator backend
- network access to the pinned Hugging Face artifacts on first acquisition

CUDA on Linux/WSL2 and Metal on Apple Silicon are implemented. NVIDIA remains the primary measured backend; Metal launch compilation is portable but still requires host-specific benchmark promotion before claiming equivalent performance. AMD/Intel backends remain discovery-only.

## Adaptive behavior

<p align="center">
  <img src="assets/turbofit-social-square.png" alt="AI that makes room: auto-selects main and auxiliary, steps down under VRAM load, and heals when memory returns" width="620">
</p>

A representative ladder is:

```text
local main + dedicated local auxiliary
→ local main shared with auxiliary work
→ smaller local model/context
→ minimum local floor
```

Contraction occurs only after a sustained deficit. Healing occurs one rung at a time only after the configured margin, dwell, hysteresis, cooldown, and flap controls pass.

A transition:

1. Blocks new auxiliary admission when leaving dedicated mode.
2. Drains active auxiliary streams.
3. Requests clean unload through Turbohaul.
4. Activates or acquires the target local rung.
5. Verifies the target.
6. Atomically publishes routes.
7. Restores and verifies the previous state on failure.

At the minimum local floor, Turbofit does not route to an API model. If no lower local rung fits, it holds the floor and reports the capacity condition.

## Configuration and evidence

| Path | Purpose |
|---|---|
| `runtime-profiles/*gb.yaml` | production hardware profiles |
| `runtime-profiles/acquisitions.json` | pinned sources, hashes, and Turbohaul tag recipes |
| `runtime-profiles/runtime-resolutions.json` | rung-to-model-tag resolution |
| `runtime-profiles/rung-requirements.json` | per-card VRAM requirements |
| `references/model-catalog.json` | requested model variants and capabilities |
| `references/configuration-matrix.json` | generated main × auxiliary × context candidate space |
| `benchmarks/suite.yaml` | promotion gates |
| `references/results/` | measured machine-readable evidence |

The matrix contains the requested 12 main variants × 4 auxiliary modes × 4 contexts: **192 research configurations**. Every row now compiles to a concrete, `--jinja`-enabled launch recipe; FP16, quantized, vision-projector, and DSpark artifacts resolve independently. A compilable row is not automatically a production recommendation. Production promotion requires artifact, runtime, performance, quality, and pressure/self-heal evidence.

The hybrid system-RAM catalog separately defines executable 64K, 128K, 262K, and 1M candidates for Laguna, MiniMax M3, and GLM 5.2. Higher-context rows are explicitly `configured-unmeasured`, use CPU/system-RAM policies (including MoE expert offload where supported), and cannot be promoted until real evidence exists.

The current source list includes:

- [GLM/GRM 5.2 2.788 bpw](https://huggingface.co/sokann/GLM-5.2-GGUF-2.788bpw)
- [MiniMax M3 GGUF](https://huggingface.co/unsloth/MiniMax-M3-GGUF)
- [Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1)
- [Laguna S 2.1 GGUF](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF)
- [GRM 2.6 Plus GGUF](https://huggingface.co/bartowski/OrionLLM_GRM-2.6-Plus-0628-GGUF)
- [Carwin MoE Nano GGUF](https://huggingface.co/isneezekittens/Carwin-MoE-Nano-GGUF)
- [Ternary Bonsai 27B GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
- [Bonsai 27B GGUF](https://huggingface.co/prism-ml/Bonsai-27B-gguf)

## Performance priorities

Candidate ranking follows:

```text
quality
→ reach 128K context
→ reach 20 tok/s
→ reach 262K context
→ reach 100 tok/s
→ reach 1M context
→ maximize speed
```

Measured claims remain attached to their exact artifact, runtime flags, context, host fingerprint, throughput, TTFT, and per-card VRAM evidence.

## Hybrid large-model bring-up

`runtime-profiles/hybrid-models.json` defines dual-24 GB GPU + system-RAM placements for Laguna S 2.1 Q4_K_M, MiniMax M3 MXFP4_MOE, and GLM 5.2 2.788 bpw. Every artifact is bound to an immutable Hugging Face revision, required SHA-256 identity, and exact file size. A benchmark pass validates only that exact artifact, runtime, flags, context, and host class; it does not automatically add the candidate to the production adaptation ladder.

| Candidate / exact placement | Context | Prompt | Server decode | Wall output | Evidence |
|---|---:|---:|---:|---:|---|
| Laguna S 2.1 Q4_K_M · all attention/routing on GPU · 46 CPU-MoE layers · 10 threads | 128K | 4.645 tok/s | **4.632 tok/s** | 4.131 tok/s | `sha256:62d45fb10a03ffbcfe1859195e741e2ee526d68f8e8270ed0b0fb276dbe87338` |
| MiniMax M3 Q4_K_M · all attention/routing on GPU · 56 CPU-MoE layers · 12 threads | 128K | 4.502 tok/s | **2.033 tok/s** | 1.455 tok/s | `sha256:a0e0404c3be45fa82e86d9d80d1d940b450afa25f7598db8dd8656c556b0c3e4` |
| GLM 5.2 2.788 bpw · ik_llama.cpp MLA/DSA · all experts on CPU · 14 threads | 128K | 2.471 tok/s | **1.180 tok/s** | 0.868 tok/s | `sha256:01d92667a57198559442a7d4eb258deaf4dc918b8b02d47fbe34a62793dd6ca4` |

All three 128K rows are production-style server validations. Laguna and MiniMax use `-ngl 999` so attention and routing stay on both GPUs while only the expert tensors needed to fit are left in system RAM. GLM keeps every expert layer in system RAM and uses `GGML_CUDA_NO_PINNED=1`; this avoids the prior 212.7 GiB pinned-host-memory failure. Row split was rejected by the architecture-specific Laguna and MiniMax runtimes, so the validated topology remains layer split.

The configuration checker reports both static `hardware_fits` and current `launch_ready`, so a machine is not called ready while another resident model still occupies required VRAM:

```bash
scripts/turbofit-hybrid-config list
scripts/turbofit-hybrid-config check glm-5-2-2-788bpw dual-24gb-64k
```

The evidence-first benchmark stage records raw responses, measured token usage, exact-answer quality checks, passkey context retrieval, effective end-to-end output throughput, host RAM, per-GPU VRAM, and an evidence SHA-256:

```bash
scripts/turbofit-benchmark-stage \
  --candidate <model-id> \
  --configuration dual-24gb-64k \
  --base-url http://127.0.0.1:<port>/v1 \
  --model <served-model> \
  --output references/results/<run>.json \
  --disable-thinking
```

## Verification

```bash
# Unit, integration, schema, profile, link, and simulated adaptation checks
scripts/release-check

# Adds live NVML, Turbohaul, stable-route, controlled-pressure,
# external-process survival, contraction, and healing checks
scripts/release-check --real
```

A simulated pass is not represented as real pressure evidence. The latest machine-readable acceptance record is stored at `references/results/adaptive-runtime-acceptance.json`.

## Safety invariants

<p align="center">
  <img src="assets/turbofit-story-9x16.png" alt="Your computer stays yours. Turbofit keeps local intelligence available while yielding resources to your work." width="430">
</p>

- External GPU processes are never terminated or signaled.
- Physical capacity selects the profile; transient availability selects the rung.
- All model lifecycle operations go through Turbohaul.
- Downloads use pinned revisions and required SHA-256 values.
- Missing or mismatched artifacts fail closed.
- New routes are not published before local verification.
- Credentials and machine-local paths never enter portable profiles.
- Research candidates never become production recommendations automatically.

## Remaining development

The following items remain outside the verified release boundary:

- host-specific Metal performance evidence and AMD/Intel runtime backends
- promotion evidence for every one of the 192 compiled model configurations
- Mixture-of-Agents presets
- pricing-aware opt-in API routing
- richer daemon/service management outside systemd

Model-source intelligence is refreshed daily by `.github/workflows/model-intel.yml`; discovery never bypasses the benchmark promotion gates.

## License

MIT License. Copyright (c) 2026 **sovthpaw (SouthpawIN)**. See [`LICENSE`](LICENSE).

## Repository map

```text
src/turbofit_runtime/       schemas, selection, acquisition, policy, controller
runtime-profiles/           production profiles and runtime catalogs
benchmarks/                 promotion suite
research/                   candidate-only discovery
scripts/turbofit-runtime    list/set/status selection CLI
scripts/turbofit-controller adaptive local controller
scripts/turbofit-gateway.py stable OpenAI-compatible gateway
references/results/         measured evidence
assets/turbofit-hero.png    README image
tests/                      unit and integration gates
```
