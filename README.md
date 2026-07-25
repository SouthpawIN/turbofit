# Turbofit

Hardware-aware runtime selection and adaptive main/auxiliary routing for Hermes Agent.

Turbofit uses immutable, portable **Turbofiles** to describe an evidence-backed rung ladder. Local model lifecycle operations go through **Turbohaul Manager v0.7**. The OpenAI-compatible gateway exposes stable model IDs while backing routes change:

- `auto`
- `active:main`
- `active:aux`

External GPU consumers have absolute priority. Turbofit may reclaim only residency explicitly owned by Turbohaul; it never signals unrelated processes.

## Authority model

| Concern | Authority |
|---|---|
| Portable recommendation policy | Turbofile under `runtime-profiles/` |
| Physical hardware identity | `turbofit_runtime.hardware` |
| Current pressure classification | `turbofit_runtime.pressure` |
| Contraction/expansion decision | `turbofit_runtime.policy` |
| Local model residency | Turbohaul Manager v0.7 HTTP API |
| Verified rung transition | `turbofit_runtime.reconciler` |
| Stable provider routes | `scripts/turbofit-gateway.py` route-state reader |
| Research intake | `research/candidates.json` only |
| Human-readable views | Generated wiki sections; never runtime authority |

Legacy `serve`, direct launcher, PID-file, and scaling-watcher paths are compatibility tools only. They are not allowed to reconcile an adaptive Turbofile runtime.

## Install

```bash
# Add the tap, then install
hermes skills tap add SouthpawIN/turbofit
hermes skills install SouthpawIN/turbofit/skills/turbofit
```

For development, work from the Git repository rather than the installed skill directory.

Requirements:

- Python 3.11+
- PyYAML for YAML profiles; canonical JSON remains standard-library-only
- Turbohaul Manager v0.7 for local serving
- Hermes Agent for provider integration
- NVIDIA tooling only when using NVIDIA hardware

API credentials belong in Hermes/provider configuration. Turbofiles, research candidates, benchmark records, and route-state files reject or omit credentials.

### Use Turbofit as the Hermes provider

The gateway is one OpenAI-compatible provider. `auto` selects the tested hardware profile; `active:main` and `active:aux` remain stable while the controller changes backing models:

```yaml
custom_providers:
  - name: turbofit
    base_url: http://127.0.0.1:8091/v1
    api_mode: chat_completions

model:
  provider: custom:turbofit
  default: auto
```

Auxiliary tasks can use the same provider with model `active:aux`; vision should use `active:main` when the selected main model is multimodal. No second provider or route-specific base URL is required.

## Auto and manual runtime selection

```bash
# Inspect only combinations with measured VRAM requirements and resolvable
# Turbohaul manifests for this machine
scripts/turbofit-runtime list

# Hardware-auto mode: select the canonical class from physical topology
scripts/turbofit-runtime set auto

# Manual mode: pin the healing ceiling to one compatible combination
scripts/turbofit-runtime set <profile-id>

# Inspect the persisted request, then run one bounded controller tick
scripts/turbofit-runtime status
scripts/turbofit-controller --once

# Optional: install the persistent user service; starting remains explicit
scripts/install-controller-service
scripts/install-controller-service --start

# 4. Use stable gateway IDs
curl http://127.0.0.1:8091/v1/models
```

Both modes start from the terminal API rung. Auto chooses a canonical physical-hardware profile; manual validates the requested combination against the same physical topology. Current availability is evaluated later by one shared pressure policy, so neither mode bypasses contraction, hysteresis, rollback, or healing controls.

## Turbofile contract

Schema: `turbofit.runtime/v1`

A profile contains exactly:

```text
schema, id, revision, hardware, policy, roles, rungs
```

Core guarantees:

- Frozen dataclasses and immutable ordered rungs
- Strict unknown-field rejection
- Topology-aware hardware classes (`1x48` is not `2x24`)
- Content-addressed manifest/evidence references (`sha256:<64 lowercase hex>`)
- No local paths, credentials, GPU placement, or mutable machine state
- Dedicated local rungs require main and auxiliary manifests
- Shared-main rungs preserve one main residency and omit an auxiliary manifest
- The final rung is API-only and defines both role policies

Eight class profiles are supplied for 8, 16, 24, 48, 64, 96, 200, and 300 GB. A class without measured local evidence remains API-only rather than advertising an unproven local winner.

## Complete candidate configuration catalog

`references/model-catalog.json` records the 12 requested main variants, their verified Hugging Face sources, capabilities, and runtime features. `references/configuration-matrix.json` is generated from it and contains every main × auxiliary × context combination: **12 × 4 × 4 = 192 candidates** across 64K, 128K, 262K, and 1M.

```bash
scripts/generate-configuration-matrix
```

The candidate matrix is complete, but it is not a claim that every row fits every machine. A row enters an automatic Turbofile only after artifact, runtime, performance, quality, and pressure/self-heal evidence passes. The existing `references/main-aux-matrix.json` remains the live benchmark campaign ledger so prior measured evidence is not rewritten or relabeled.

### Published auto ceilings through 48 GB

| Physical class | Auto ceiling | Local roles |
|---|---|---|
| 1x8 GB | API policy, 64K route ceiling | API main + auxiliary |
| 1x16 GB | API policy, 128K route ceiling | API main + auxiliary |
| 1x24 GB | GRM 2.6 Plus, 128K | one shared main residency |
| 2x24 GB | GRM 2.6 Plus, 128K on GPU 1 + Carwin Nano 1-bit Bonsai, 262K on GPU 0 | dedicated main + auxiliary |

The 48 GB composition uses two measured `split_mode=none` residents pinned to separate cards. The previous dual-1M draft is intentionally not the auto recommendation: its real Turbohaul admission/load gate exceeded a 24 GiB card, so publishing it would overclaim local support.

## Adaptive behavior

Contraction is one rung at a time after configured dwell:

```text
dedicated main+aux
→ shared-main auxiliary
→ smaller model/context rung(s)
→ terminal API policy
```

Expansion/self-healing walks back toward the physical recommendation only after margin, dwell, hysteresis, cooldown, and flap controls pass.

A transition:

1. Blocks new auxiliary admission when leaving dedicated mode.
2. Drains active auxiliary streams.
3. Requests clean unload through Turbohaul.
4. Leaves process escalation to Turbohaul; Turbofit never sends process signals.
5. Activates the target rung or API policies.
6. Verifies the target.
7. Atomically publishes stable routes.
8. Restores and verifies the previous state on any failure.

The admission guard may temporarily redirect new auxiliary work before drain. A new target rung is never published before target verification, and the previous route is restored on failure.

## Benchmark and promotion gates

The canonical suite in `benchmarks/suite.yaml` requires these ordered stages:

1. artifact
2. runtime
3. performance
4. quality
5. pressure-self-heal

Promotion records include artifact hashes, host fingerprint, observed context, throughput, TTFT, per-card VRAM and power, quality score, raw-result identity, and per-stage evidence identity.

`matrix-benchmark.py --mark-success` fails closed unless `--promotion-record` passes the canonical suite.

## Candidate intelligence

Collectors are public-read-only and write only candidate status:

```bash
PYTHONPATH=. python3 research/discover_huggingface.py
PYTHONPATH=. python3 research/discover_model_news.py --url <public-rss-or-atom-url>
PYTHONPATH=. python3 research/discover_api_models.py --provider <name> --url <public-models-url>
```

They never edit production profiles or routes and never auto-promote discoveries. See `docs/cron/model-intelligence.md`. Live jobs require explicit schedule and delivery approval.

## Generated wiki

`src/turbofit_runtime/wiki.py` publishes bounded deterministic sections to the Turbofit README/checklist in the Hermes wiki. Publication validates evidence paths and candidate state first. A second publication is idempotent.

## Verification and release

```bash
# Syntax, unit, integration, profiles, links, simulated adaptation
scripts/release-check

# Adds real NVML, Turbohaul, stable-route, bounded 3 GiB pressure,
# contraction, pressure-process survival, and recovery gates
scripts/release-check --real
```

A release is blocked unless the real gate passes. A simulated pass is not evidence of real pressure handling.

The latest local real preflight record is stored at:

```text
references/results/adaptive-runtime-acceptance.json
```

## Current host blocker

At the latest controlled attempt, simulated contraction/recovery, Turbohaul Manager v0.7 `/status`, and all three live stable gateway routes passed. The real pressure gate correctly stopped before GPU allocation because:

- NVIDIA userspace library `580.173` did not match loaded kernel module `580.159.03`.
- The controller unit is installed but intentionally remains inactive until NVML is healthy.

No external process was signaled or modified, and no GPU allocation was attempted. Once NVML is healthy, the controlled gate allocates only its own bounded CUDA buffer, verifies that buffer survives contraction, terminates only that acceptance-owned process, and then verifies healing to the selected ceiling.

## Repository map

```text
src/turbofit_runtime/       portable schema, hardware, policy, client, reconciler
runtime-profiles/           canonical class and migrated profiles
benchmarks/                 canonical promotion suite
research/                   candidate-only intelligence collectors
scripts/turbofit-runtime    list/set/status selection entry point
scripts/turbofit-controller persistent adaptive controller
scripts/turbofit-gateway.py stable OpenAI-compatible gateway
scripts/adaptive-acceptance controlled acceptance runner
references/results/         machine-readable acceptance evidence
tests/                      unit and integration gates
```

## Safety invariants

- External GPU consumers are never terminated.
- Recommendations never depend on transient free VRAM.
- Local lifecycle changes go through Turbohaul.
- Stable route IDs do not expose transient model identities.
- Credentials never enter portable artifacts.
- Candidate discovery never equals promotion.
- Generated documentation never becomes runtime authority.
