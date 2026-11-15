# Expert offload: running an oversized MoE in system RAM

An MoE model can be much larger than the cards it runs on, because llama.cpp can
keep a chosen number of layers' expert weights in system RAM (`--n-cpu-moe N`)
and leave attention, KV and dense weights on the GPU. Choosing `N` is pure
arithmetic:

```
usable VRAM = free VRAM - reserve
per layer   = expert bytes / expert layers
layers on GPU = (usable VRAM - non-expert bytes) / per layer
n_cpu_moe   = expert layers - layers on GPU
```

## The numbers come from the file, not a table

`turbofit_runtime.gguf_footprint` reads the GGUF header (metadata + tensor
descriptors, a few KiB per shard) and measures:

- `total_bytes` — every shard of the artifact on disk,
- `expert_bytes` — the sum of tensors named `blk.N.ffn_{gate,up,down}_exps`
  (llama.cpp's spelling; older `ffn_*_expert` and `gpt-oss`-style
  `ffn_*_experts` spellings are matched too), excluding router weights
  (`ffn_gate_inp`) and shared experts (`ffn_*_shexp`), which stay on the GPU,
- `expert_layers` — distinct `blk.N` indices carrying expert tensors, falling
  back to the declared block count when the tensors carry no layer index.

Tensor sizes are derived from the descriptor offsets rather than a
quantization table, so IQ\*/TQ\*/ternary quants are measured exactly. Results are
cached under `~/.cache/turbofit/gguf-footprints` keyed by path, size and mtime,
so a re-quantized or replaced file re-measures automatically. Set
`TURBOFIT_FOOTPRINT_CACHE` to relocate the cache.

## Enabling it

A recipe (or a context override) opts in:

```json
{ "auto_cpu_moe": true }
```

Resolution then measures *this host*:

- **VRAM**: the sum of `memory.free` from `nvidia-smi` (a card already serving
  another model is not counted as capacity), falling back to the fingerprint's
  device total only when no sample is available,
- **RAM**: `/proc/meminfo` `MemAvailable`,
- **reserves**: 1536 MiB of VRAM and 2048 MiB of RAM by default; a recipe can
  override the VRAM one with `offload_vram_reserve_mb` (it has to cover KV cache
  and compute buffers, which the planner does not model: Laguna S2.1 sets 12288
  so 64K context fits on two cards).

The resulting `--n-cpu-moe N` is appended to the launch command and exposed as
`ResolvedComponent.n_cpu_moe` / `.offload_reason` for logs and evidence.

## Multi-GPU: the split has to be byte-balanced

`--n-cpu-moe` moves the *first* N layers' experts to RAM, so the weights that
stay on the GPUs pile up in the later layers. An even layer split then hands one
card far more than its share: measured on two 24 GiB cards, `--n-cpu-moe 28` with
the default split wanted ~23 GiB on CUDA0 against ~14 GiB on CUDA1, so the
load died on CUDA0 while CUDA1 sat half empty.

When auto planning runs on more than one device, the resolver therefore also
emits a `--tensor-split` whose cut points are chosen from the measured per-layer
bytes (contiguous ranges, non-layer tensors counted against the first device,
minimising the heaviest device's share). For the same model that is `37,11`, and
the two cards come up at ~22.9 GiB / ~16.8 GiB used.

Behaviour that does not change:

- a model that fits VRAM plans `n_cpu_moe = 0` and gets no flag;
- an explicit `n_cpu_moe` in a recipe (or context override) is authoritative and
  skips measurement entirely;
- Strata, vLLM and other runtimes never see the flag — they place experts
  themselves. A recipe that declares `n_cpu_moe` on such a flavor is rejected
  rather than emitting a flag the engine cannot parse;
- an artifact that is not on disk yet is a planning state, so resolution does
  not fail over it (the engine fails on the missing file with its own message).

When the model *is* measurable and cannot fit — non-expert weights exceed usable
VRAM, or the offloaded experts exceed usable RAM — resolution raises
`ExpertOffloadInfeasible` carrying the planner's reason, e.g.
`non-expert weights alone need 30.3 GiB but only 22.5 GiB of VRAM is usable`,
instead of launching an engine that will OOM.

## What this does not model

KV cache is not in the planner's budget: the defaults assume it fits inside the
1.5 GiB reserve. On very long contexts raise `offload_vram_reserve_mb` so the
planner leaves room, or the engine will allocate into a card that is already
full. Unified-memory hosts (Metal) and pure-CPU backends skip auto planning —
their VRAM *is* RAM, so there is nothing to offload to.

## Catalog entries using it

`Laguna S2.1`, `MiniMax M3` and `Qwen 3.8 Flash Next Abliterated` carry
`auto_cpu_moe: true`: each is larger than a 48 GiB desktop, so the flag is
computed per host instead of hardcoded.
