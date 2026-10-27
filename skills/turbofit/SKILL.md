---
name: turbofit
description: "Hardware-aware adaptive Hermes runtime using portable Turbofiles, total usable memory, owned native llama.cpp residency, stable auto/active:main/active:aux routes, and evidence-backed promotion. Use for recommending, activating, inspecting, testing, or troubleshooting Turbofit runtimes."
version: 2.4.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, llama-cpp, llm, accelerator, cpu, adaptive-runtime, turbofile]
---

# Turbofit

## 2.4 model authority

Two ladders converge on Qwen 3.8 Flash Next. Ladder 1 — dedicated VRAM (dense is fine): Bonsai 2 27B PTQ1_0/PQ2_0 at 8–11 GB → Underdog 27B PQ2_0 at 12–15 GB → Qwen 3.8 27B GSQ-RCO IQ3_S at 16–48 GB (262K from 24 GB) → Flash Next IQ3_XXS at 64 GB+. Ladder 2 — unified/CPU (MoE only): Maple Preview 20B-A1B at 8–15 GB → Cyber-Tiel Coder 35B-A3B UD-Q4_K_XL at 16 GB+ → Flash Next → DeepSeek V4.1 (1017) as the top rung (native engine pending). Measured rung (2026-10-27, RTX 3080 10GB + 32GB RAM): Ornith 1.5 35B-A3B Q4_K_M on llama.cpp b11935, ngl 99 ncmoe 30 fa on → pp512 126 t/s, tg128 41 t/s, 21.9 GB GGUF resident, ctx 65536 — the live Windows main. Fit math: model + KV ≤ tier. Cross-cutting: MTP on at every rung; IQ3_S is lossless — do not push a heavier quant past it for the power bill. `serve auto` builds the fit chain from these ladders and walks down on OOM/timeout — the Fit List IS the fallback chain.

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

## Windows native install

Windows is a separate, user-scoped path: pinned llama.cpp runtime on loopback 8092, gateway on 8091, two limited user scheduled tasks. It wants a C++ build toolchain (CMake, Ninja, MSVC, plus the CUDA toolkit only for the `cuda` backend) — **never Ollama**: Turbofit owns its own `llama-server.exe`, and :8091 is the Turbofit gateway, not an Ollama proxy. Sequence: install the Hermes plugin, download a model artifact, then run the service installer — the plugin alone does not bind 8091, and restarting the messaging gateway will not create it. To avoid the toolchain, pass `-Binary` a prebuilt `llama-server.exe` taken from the **pinned revision's own release tag** (mandatory for Vulkan, which the runtime installer cannot build), accepting that a stock prebuilt lacks the repo's yarn long-context patch.

Turbofit needs **no account, subscription, or API key** — the plugin is a plain git install and the provider is `custom:turbofit` with model `auto`. On a fresh machine give the user a working agent first (Hermes' built-in Local Models provider also needs no account) and do the Turbofit native service last; its GGUF download is the long pole. Run **one** local stack at a time: Hermes' managed Local Models server (port 8080, started and stopped by the app) and Turbofit's owned residency (8092/8091) both want the GPU, and adaptive contraction only works when Turbofit owns the residency.

Two rules that cost a session to learn:

- **`hermes plugins install` finishes with the plugin disabled.** Run `hermes plugins enable turbofit` explicitly; do not gate on `hermes plugins list` containing the word `disabled`, because the listing format does not always show it. Its tools and `/turbofit` command only register after a full Desktop quit and a new session.
- **Verify the accelerator before believing a speed number.** `llama-server --list-devices` must name the GPU and the extracted folder must hold the backend DLL. A binary that answers `--help` but lists no device runs the model on CPU at roughly 1 tok/s, which surfaces later as "Turbofit is broken."

Dependency table, exact commands, prebuilt acquisition (pinned release tag, not `latest`), `-Model`/`-Context` caveats, LAN sharing, accelerator verification, and how to drive a by-hand install from a weak model: `references/windows-native-install.md`.

Measured on RTX 3080 10GB + 32GB RAM, Ornith-1.5-35B-A3B Q4_K_M, ngl 99 ncmoe 30 fa on (2026-10-27, live-server contention): llama.cpp b10269 pp512 83 t/s / tg128 42.8 t/s; b11935 pp512 126 t/s / tg128 41.3 t/s. Newer builds are a prompt-eval win, decode-neutral — prefer newer prebuilts.

## Sharing the gateway with a remote Hermes (Hermes Cloud, VPS, another box)

Hermes Cloud (portal.nousresearch.com/cloud) is a hosted always-on Hermes **agent** instance sized by VM tier — its RAM is for the agent process, never for hosting GGUFs. The integration direction that matters: route the remote agent's inference to the local Turbofit gateway.

- Expose the gateway: re-run the Windows service installer with `-GatewayHost 0.0.0.0` (or set `TURBOFIT_GATEWAY_HOST`; default `127.0.0.1`). The LAN-sharing recipe itself: `references/windows-native-install.md`.
- On the remote Hermes: `provider: custom`, `base_url: http://<reachable-ip>:8091/v1`, model id one of the stable route IDs (`auto`, `active:main`, `active:aux`). Never `custom` + a GGUF filename on the client.
- The gateway authenticates no inbound calls — anything beyond loopback must sit behind a tailnet or tunnel, never raw public. The CORS `browser_origin_allowed` check gates only browser origins; server-to-server calls pass.
- An always-on remote agent loses its main model whenever the rig sleeps — keep a portal-model fallback in the remote config, and say so before recommending the setup.
- Manage cloud instances from a local Hermes via the Portal MCP server: `hermes mcp add --url https://portal.nousresearch.com/mcp --auth oauth hermes-cloud` (list/start/stop/create/`update_env`, cost estimates).

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

On the Windows rig the gateway mirrors logs to `%LOCALAPPDATA%\hermes\logs\turbofit-gateway.log` (pythonw discards stderr, so without it deaths leave no trace); the `TurbofitGateway` keepalive task restarts it within 5 minutes, so check that log before assuming an outage was long.

If `http://127.0.0.1:8091/v1/models` is connection-refused, the Turbofit provider gateway is not running. That is setup missing. Do not restart the Hermes messaging gateway, do not start with a firewall hunt when nothing listens, and do not invoke TurboSouth until the endpoint answers.

If the platform reports a driver/runtime mismatch, stop the real pressure test. Do not attempt blind driver reloads or disruptive accelerator work.

Full architecture and schema: `README.md`.
