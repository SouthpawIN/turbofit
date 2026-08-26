# Sparse-MoE Expert Offload (Qwen3.8-Flash-Next, GLM-5.3-Flash)

Large sparse-MoE checkpoints are dominated by **routed expert weights** that are
only sparsely read per token. Those are the cheapest tensors to keep in system
RAM while attention, shared experts, embeddings and the PLE n-gram table stay
resident on the accelerator.

## Portable, hardware-derived sizing (NOT host-baked)

`turbofit_runtime.offload.moe_expert_offload_layers()` computes `-ncmoe N` from
the *machine*, so the same recipe works on a 16 GB laptop and a 192 GB unified
box. Recipes declare the model's **shape** only:

```json
"moe_offload": {
  "total_layers": 48,
  "expert_bytes_per_layer": 2190433321,
  "dense_bytes": 6442450944,
  "compute_reserve_mb": 2048
}
```

`recipes.py` calls the solver when `n_cpu_moe` is not explicitly set, using
`total_vram_mb` (or `total_usable_memory_mb` on unified memory) and
`host_usable_memory_mb`. An explicit `n_cpu_moe` still wins, and a machine that
cannot hold the model raises rather than silently thrashing.

Sizing across tiers for the three Qwen3.8-Flash-Next quants:

| Tier | IQ1_S (68 GB) | Q3_K_XL (84 GB) | Q4_K_XL (104 GB) |
|---|---|---|---|
| 8 GB / 32 GB RAM | rejected | rejected | rejected |
| 16 GB / 64 GB RAM | `-ncmoe 42` | rejected | rejected |
| 24 GB / 128 GB RAM | `-ncmoe 36` | `-ncmoe 39` | `-ncmoe 41` |
| 48 GB / 377 GB RAM | `-ncmoe 18` | `-ncmoe 24` | `-ncmoe 29` |
| 96 GB / 377 GB RAM | `-ncmoe 0` | `-ncmoe 0` | `-ncmoe 5` |
| 192 GB unified | `-ncmoe 0` | `-ncmoe 0` | `-ncmoe 0` |

## Qwen3.8-Flash-Next — VERIFIED WORKING

- **Arch**: `qwen4exp`, 125B total / 6B active, 512 experts (10 routed + 1 shared),
  48 layers, 262K native context, PLE n-gram table (`per_layer_token_embd`,
  160 × 320M rows), hyper-connections.
- **Runtime**: llama.cpp **PR #27742** (`model: add Qwen3.8-Flash-Next (qwen4exp)`).
  PR #27739 is a *different, incompatible* PR — it expects per-layer
  `blk.N.ple_ngram_embd` instead of the shipped shared `per_layer_token_embd`.
- **Local build**: `~/projects/llama.cpp-latest/build-cuda/bin/llama-server` @ `pr-27742`
- **Required local patch**: `graph_max_nodes` in `src/llama-context.cpp` must include
  `LLM_ARCH_QWEN4EXP` in the 32×-tensors branch, otherwise the 4-branch
  hyper-connection graph overflows with `GGML_ASSERT(obj_new)`.

### Measured on 2× RTX 3090 (48 GB VRAM) + 377 GB RAM

| Quant | Size | Load | Prompt | Decode |
|---|---|---|---|---|
| UD-IQ1_S | 68 GB | 7 s | 16.07 tok/s | 8.26 tok/s |
| UD-Q3_K_XL | 84 GB | 3 m 36 s | — | — |
| UD-Q4_K_XL | 104 GB | ~7 min | 10.75 tok/s | 8.68 tok/s |

Launch (Q4_K_XL, this box):
```
llama-server -m <shard-00001-of-00004>.gguf --jinja \
  -ngl 99 -ncmoe 36 -c 8192 -fa on --parallel 1
```

### Pitfalls (all hit and confirmed)

- **`-ncmoe 0` offloads nothing.** The flag means "first N *layers*", so 0 = all
  experts stay on GPU = OOM. Use the layer count, not 0.
- **`--load-mode none` costs 24× throughput** on split GGUFs (0.36 tok/s vs
  8.68 tok/s) because each token re-reads from disk. Leave mmap enabled.
- **`--mlock` fails on 100 GB+ models** with `RLIMIT_MEMLOCK` unless ulimit is
  raised; it stalls the load instead of erroring cleanly.
- **`--fit on` conflicts with explicit `-ngl`** (`n_gpu_layers already set by
  user, abort`). Pick one.
- **Split GGUFs load from shard 1 automatically** — do not concatenate shards.
- **Free the GPUs first.** A leftover Turbofit backend holding 22 GB caused the
  compute-buffer OOM that looked like a weights problem
  (`systemctl --user stop turbofit-controller`).

## GLM-5.3-Flash — runtime ready, awaiting weights

- **Arch**: `glm5_next` / `Glm5NextForConditionalGeneration`, 321.3B hybrid
  linear/sparse MoE. 45-layer trunk + MTP block at index 45, 288 routed experts
  (8 per token) + 1 shared, hidden 4096, 1M context, vision tower, FP8 source.
  34 KDA (Kimi Delta Attention) layers + 11 DSA layers at indices 3, 7 … 43.
- **Runtime**: llama.cpp **PR #27754** (`unslothai:glm5next/upstream`) — includes
  the vision tower and a `conversion/glm5next.py` converter with MTP export.
  PR #27752 is a competing text-only draft.
- **Local build**: `~/projects/llama.cpp-glm5/build-cuda/bin/llama-server` (BUILT,
  `LLM_ARCH_GLM5NEXT` present).
- **No public GGUF exists yet.** Every candidate repo
  (`unsloth/`, `AtomicChat/`, `aj9o9/`, `vcruz305/`, `MaliAir/`,
  `DevQuasar/zai-org.GLM-5.3-Flash-GGUF`) contains only READMEs/images; the sole
  real artifact is a 1.1 GB `mmproj` in the DevQuasar repo. Converting locally
  from `zai-org/GLM-5.3-Flash` (328 GB FP8) instead.
