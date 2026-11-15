# Tier placement: VRAM band *and* system RAM

A model that exceeds a VRAM band is not automatically unusable. On an MoE
checkpoint llama.cpp can hold the expert tensors in system RAM (`--n-cpu-moe`,
see `docs/expert-offload.md`) while attention, KV cache, dense weights and the
vision projector stay on the GPU. The launch path has known this since the
expert-offload work; tier placement did not, so it refused models the engine can
actually run on a large-RAM host.

Placement is therefore two-dimensional: **the tier's VRAM band** and **the
system RAM budget**. Every resolved configuration reports one of two placements:

| `placement` | meaning | what is reported |
|---|---|---|
| `gpu` | weights + KV fit the band | `vram_gib = total`, `ram_gib = 0` |
| `offload` | exceeds the band, fits with experts in RAM | `ram_gib` (experts actually moved to RAM), `vram_gib` (bytes still on GPUs), `n_cpu_moe`, `expert_layers`, `reason` |

`offload` rows carry `evidence: "measured-expert-footprint"`; `gpu` rows carry
`evidence: "weights-and-kv-fit-vram"`. The two are never merged into one list:
`references/hardware-tier-tournaments.json` has `candidates` (full-GPU) and
`offload_candidates` (offloaded), and the report has a matching split, so
"runs entirely on the GPU" and "runs with 67 GiB of experts in RAM" cannot be
mistaken for each other.

## The arithmetic is the launcher's, not a parallel model

The offload branch calls `turbofit_runtime.experts.plan_expert_offload` — the
same function the recipe resolver uses to emit `--n-cpu-moe` — with:

* `total_bytes` = weights + KV cache at the requested context,
* `expert_bytes` / `expert_layers` = measured from the artifact's GGUF tensor
  table (`references/model-footprints.json`, `expert_measurement`),
* `vram_bytes` = the tier's band plus the documented reporting tolerance,
* `ram_bytes` = the RAM budget below.

Its 1536 MiB VRAM and 2048 MiB RAM reserves apply unchanged, so a placement is
conservative in the same way a real launch is. A model is only offered as
`offload` when an *actual* measurement says how many expert bytes must move; a
MoE model whose header has not been measured is left out rather than guessed at
(the generator prints the unmeasured ids).

Regenerate the measurement with:

```
PYTHONPATH=src python scripts/generate-model-footprints --measure-experts
```

which reads each pinned artifact's GGUF header over HTTP Range (a few MiB per
shard, no weights) and records `expert_bytes`, `expert_layers` and the pinned
revision it came from.

## The RAM assumption

The committed artefacts cannot encode one machine's RAM, so they state a
budget:

* default: **128 GiB of available system RAM** (`DEFAULT_RAM_GIB` in
  `turbofit_runtime.tier_fit`), documented in
  `references/hardware-tier-tournaments.json → ram_assumption` and in
  `references/hardware-tier-report.json → ram_budget`;
* override for one run:
  `scripts/generate-hardware-tier-tournaments --ram-gb 377` or
  `scripts/turbofit-hardware-tiers --ram-gb 377`;
* override for every caller: `TURBOFIT_RAM_GIB=377`.

The number is an *available* budget — the same quantity
`experts.available_ram_bytes()` reads from `/proc/meminfo` `MemAvailable` — not
an installed total. `turbofit-hardware-tiers` with no `--ram-gb` uses the host's
real `MemAvailable`, so a user asking "what runs on this box?" gets their own
answer instead of the published assumption. Nothing in the repo contains a
specific owner's RAM figure.

## Where it shows up

* `references/hardware-tier-tournaments.json` — `candidates`,
  `offload_candidates`, `ram_assumption`.
* `references/hardware-tier-report.json` — `ram_budget`, per-tier
  `offload_candidates` with `expert_offload.{ram_gib,vram_gib,n_cpu_moe}`.
* `scripts/turbofit-hardware-tiers` — prints both lists per tier.
* `turbofit_runtime.tier_fit.select_placements()` — the shared selection used by
  the generators and the tests.
