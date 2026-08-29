---
name: turbofit
description: "Hardware-aware adaptive Hermes runtime using portable Turbofiles, total usable memory, owned native llama.cpp residency, stable auto/active:main/active:aux routes, and evidence-backed promotion. Use for recommending, activating, inspecting, testing, or troubleshooting Turbofit runtimes."
version: 2.3.1
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, llama-cpp, llm, accelerator, cpu, adaptive-runtime, turbofile]
---

# Turbofit

## 2.4 model authority

Fit List keeps dedicated VRAM separate from integrated/RAM-only total memory. Dedicated: Maple Preview TQ2_0 at 8GB, Qwen 3.8 27B Unleashed UD-IQ3_XXS at 16GB, UD-Q3_K_XL at 24–95GB, and Qwen 3.8 27B 16-bit at 96GB+ until an Unleashed FP16 GGUF is published. Shared total memory: Maple at 8–15GB, Ornith 1.5 35A3B at 16–23GB, and Unleashed UD-Q3_K_XL at 24GB+. An 8GB GPU may also use Ornith when host RAM can hold offloaded experts. Maple remains Auto on dedicated 8GB at native 64K/128K. Auxiliary is Ornith, optional Carwin Nano, or auto.

## Use when

- Selecting an evidence-backed main/aux runtime for physical hardware
- Activating or inspecting a Turbofile profile
- Diagnosing pressure, contraction, expansion, routing, or native residency
- Benchmarking or promoting a model pair
- Updating candidate intelligence or generated wiki views

## Canonical workflow

Work from the Git repository, not an installed copy.

```bash
scripts/turbofit-runtime list
scripts/turbofit-runtime set auto
scripts/turbofit-runtime set <profile-id>
scripts/turbofit-runtime status
scripts/turbofit-controller --once
curl -fsS http://127.0.0.1:8091/v1/models
```

`set auto` chooses a canonical profile from immutable physical topology. `set
<profile-id>` validates a measured, natively resolvable manual combination.
Both begin at API safety and use the same adaptive controller to contract and
heal; manual selection changes only the healing ceiling.

Use only stable provider IDs: `auto`, `active:main`, and `active:aux`.

FreeToken 0.1.2 at revision `0ab982f10905fa775962a4eddcb44caa50065251` is an optional NVIDIA/CUDA-13 text-only MoE candidate. Install or probe with `scripts/install-freetoken-runtime`; never expose it as an Auto rung, inherit its published TPS, or replace active Qwen 3.8/Ornith authority until an exact supported model recipe passes the full physical and intelligence campaigns.

## Runtime authorities

1. Turbofile: portable recommendation and ordered rung policy.
2. Hardware fingerprint: physical topology, total usable memory, and per-device capacity; never transient free memory.
3. Pressure snapshot: ownership-aware transient capacity.
4. Pure policy: dwell/hysteresis/cooldown/flap decision.
5. Native runtime backend: sole local residency authority for owned processes.
6. Reconciler: drain, activate, verify, publish, rollback.
7. Gateway route state: backing targets for stable IDs.

Legacy `serve`, direct launchers, and scaling watcher are compatibility tools, not adaptive authorities.

## Portable memory allocation

Hardware fingerprints classify memory as `dedicated`, `unified`, or `cpu`. Turbofit reserves 5% of host RAM, bounded to 1–8 GiB, and never double-counts unified memory. Dedicated systems can combine VRAM and host RAM through llama.cpp offload; contexts beyond the model's native window move KV cache pressure to host RAM when at least 32 GiB is usable. Unified-memory systems suppress discrete split flags. CPU-only systems use pinned CPU runtimes with both model and draft GPU layers set to zero.

Backend order is CUDA → ROCm → Vulkan → CPU on Linux/Windows and Metal on macOS. Build or verify the current machine's pinned backend with `scripts/install-native-runtimes --backend cuda|rocm|metal|vulkan|cpu`.

## Non-negotiable safety

- Never kill or signal external accelerator/model processes.
- Signal only PID-and-command-verified processes owned by Turbofit.
- Count external memory as unavailable and managed residency as reclaimable.
- A temporary auxiliary admission redirect may precede drain; never publish a
  new target rung before verification.
- Restore and verify the previous rung after any failed transition.
- Never place paths, secrets, credentials, provider keys, or device indices in Turbofiles.
- Never treat research candidates or generated wiki text as production authority.
- Never mark benchmark success without a canonical promotion record.

## Profile/recommendation checks

```bash
PYTHONPATH=src python3 scripts/turbofit-runtime-recommend --fit-only --json
PYTHONPATH=src:. python3 -m pytest tests/test_runtime_profile.py tests/test_profile_io.py tests/test_hardware.py tests/test_recommend.py -q -o 'addopts='
```

Topology matters: `1x48` and `2x24` are different classes. Unmeasured classes keep API as the Auto safety rung while setup may expose separately labeled portable-fit local candidates for on-box validation.

**TurboFit Check** means the system scan-to-configuration process. **TurboFit List** means only the exact physical hardware-level winners generated by `scripts/turbofit-list`. Intelligence campaigns run only the current tier's tournament candidates; zero-call/token DeepSWE trials are invalid, and suite composites are rebuilt from real hash-bound pass counts rather than stale stored zeros.

Qwen 3.8 DFlash2 is a separate candidate using the pinned Inco Q4_K_M drafter and pinned z-lab llama.cpp PR runtime. Never reuse it for Bonsai. Bonsai keeps its own Prism DSpark sidecar/runtime until a dedicated Bonsai DFlash checkpoint exists.

## Pressure and adaptation checks

```bash
PYTHONPATH=src:. python3 -m pytest tests/test_pressure.py tests/test_pressure_probe.py tests/test_policy.py tests/test_reconciler.py tests/test_controller.py tests/test_runtime_service.py tests/integration -q -o 'addopts='
```

Expected contraction:

```text
dedicated aux → shared-main → smaller context/model → terminal API
```

Expected recovery walks one rung at a time toward the recommendation after margin and dwell.

## Release gates

```bash
scripts/release-check
scripts/release-check --real
```

The first command validates syntax, tests, profiles, links, and simulated transitions. The second additionally requires working accelerator telemetry, stable live routes, and controlled real pressure/recovery evidence. Do not claim release readiness if `--real` is blocked.

Acceptance evidence: `references/results/adaptive-runtime-acceptance.json`.

## Candidate intelligence

Collectors write only `research/candidates.json`:

```bash
PYTHONPATH=. python3 research/discover_huggingface.py
PYTHONPATH=. python3 research/discover_model_news.py --url <public-feed>
PYTHONPATH=. python3 research/discover_api_models.py --provider <name> --url <public-model-list>
```

No collector may modify runtime profiles, routes, or credentials. Live cron schedules/delivery require explicit user approval.

## Troubleshooting order

1. `scripts/turbofit-runtime status` and the hardware fingerprint in Dashboard/Desktop
2. The platform's available native inventory probe (CUDA, ROCm, Metal, Vulkan, or CPU)
3. Native runtime `/health`, `/v1/models`, and `/metrics`
4. Gateway `/v1/models`
5. Route-state freshness and stable IDs
6. Acceptance record blockers
7. Focused tests, then full `scripts/release-check`

If `http://127.0.0.1:8091/v1/models` is connection-refused, the Turbofit provider gateway is not running. That is setup missing. Do not restart the Hermes messaging gateway, do not start with a firewall hunt when nothing listens, and do not invoke Sirvir until the endpoint answers.

If the platform reports a driver/runtime mismatch, stop the real pressure test. Do not attempt blind driver reloads or disruptive accelerator work.

Full architecture and schema: `README.md`.
