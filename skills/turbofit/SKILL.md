---
name: turbofit
description: "Hardware-aware adaptive Hermes runtime using portable Turbofiles, Turbohaul-owned local residency, stable auto/active:main/active:aux routes, evidence-backed promotion, and external-GPU-first pressure handling. Use for recommending, activating, inspecting, testing, or troubleshooting Turbofit runtimes."
version: 2.0.0
author: SouthpawIN + Nous Girl
license: MIT
tags: [hermes-agent, turbohaul, llm, gpu, adaptive-runtime, turbofile]
---

# Turbofit adaptive runtime

## Use when

- Selecting an evidence-backed main/aux runtime for physical hardware
- Activating or inspecting a Turbofile profile
- Diagnosing pressure, contraction, expansion, routing, or Turbohaul residency
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
<profile-id>` validates a measured, Turbohaul-resolvable manual combination.
Both begin at API safety and use the same adaptive controller to contract and
heal; manual selection changes only the healing ceiling.

Use only stable provider IDs: `auto`, `active:main`, and `active:aux`.

## Runtime authorities

1. Turbofile: portable recommendation and ordered rung policy.
2. Hardware fingerprint: physical topology/capacity; never current free VRAM.
3. Pressure snapshot: ownership-aware transient capacity.
4. Pure policy: dwell/hysteresis/cooldown/flap decision.
5. Turbohaul Manager v0.7: sole local residency authority.
6. Reconciler: drain, activate, verify, publish, rollback.
7. Gateway route state: backing targets for stable IDs.

Legacy `serve`, direct launchers, and scaling watcher are compatibility tools, not adaptive authorities.

## Non-negotiable safety

- Never kill or signal external GPU processes.
- Never directly signal model processes; use Turbohaul HTTP operations.
- Count external memory as unavailable and managed residency as reclaimable.
- A temporary auxiliary admission redirect may precede drain; never publish a
  new target rung before verification.
- Restore and verify the previous rung after any failed transition.
- Never place paths, secrets, credentials, provider keys, or GPU indices in Turbofiles.
- Never treat research candidates or generated wiki text as production authority.
- Never mark benchmark success without a canonical promotion record.

## Profile/recommendation checks

```bash
PYTHONPATH=src python3 scripts/turbofit-runtime-recommend --fit-only --json
PYTHONPATH=src:. python3 -m pytest tests/test_runtime_profile.py tests/test_profile_io.py tests/test_hardware.py tests/test_recommend.py -q -o 'addopts='
```

Topology matters: `1x48` and `2x24` are different classes. Unmeasured local classes remain API-only.

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

The first command validates syntax, tests, profiles, links, and simulated transitions. The second additionally requires working NVML, Turbohaul `/status`, stable live routes, and controlled real pressure/recovery evidence. Do not claim release readiness if `--real` is blocked.

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

1. `nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits`
2. Turbohaul Manager `/status` and `/api/tags`
3. Gateway `/v1/models`
4. Route-state freshness and stable IDs
5. Acceptance record blockers
6. Focused tests, then full `scripts/release-check`

If NVIDIA reports a driver/library mismatch, stop the real pressure test. Do not attempt blind module reloads or disruptive GPU work.

Full architecture and schema: `README.md`.
