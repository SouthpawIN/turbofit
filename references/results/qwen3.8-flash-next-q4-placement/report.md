# Qwen3.8-Flash-Next UD-Q4_K_XL — dual-24GB / high-RAM optimization

## Hardware class

- Two discrete NVIDIA GPUs with at least 24 GiB each (48 GiB aggregate)
- At least 144 GiB system RAM for 64K–262K; 192 GiB for the 1M candidate
- Dual-socket NUMA is supported; this physical run used 2× RTX 3090 and 377 GiB RAM
- The full 103.69 GiB Q4 checkpoint remains mmap-backed in host RAM/page cache

## Runtime decision

Stock/static llama.cpp expert placement is the only working Q4 runtime today.
`-ncmoe N` is layer-granular: every expert in the first N layers stays in RAM;
it does not transfer only the 10 selected experts per token.

FreeToken implements the desired selected-expert design (host source-of-truth,
shared VRAM LRU cache keyed by `(layer, expert)`, PCIe fills on misses, concurrent
CPU execution), but Qwen3.8-Flash-Next support is still upstream feature request
#214. This host also has CUDA toolkit 12.0 while Turbofit's pinned FreeToken 0.1.2
contract requires CUDA 13+.

The vLLM day-zero image supports the architecture and PLE table CPU offload, but
its validated NVFP4 checkpoint is 130 GB and its minimum configurations use
large datacenter GPUs. It does not provide the selected-expert cache needed for
this 48 GB class.

## Measured expert placement

Exact model: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`be60321be4a38386344b529db0153382a2d31a5b`, UD-Q4_K_XL, four shards.

| CPU expert layers | Result | Prefill | Decode |
|---:|---|---:|---:|
| 29 | load failure | — | — |
| 32 | load failure | — | — |
| 34 | load failure | — | — |
| 35 | pass | 37.41 | 7.68 |
| 36 | pass/winner | 44.36 | 9.63 |

`35` is slower despite one more GPU-resident expert layer. The extra residency
creates device/interconnect pressure. Maximum residency is not maximum speed.

Validated n=3 short-context medians for `-ncmoe 36`:

| CPU/NUMA policy | Threads | Prefill median | Decode median |
|---|---:|---:|---:|
| default | 16 | 40.72 | 7.48 |
| NUMA distribute | 10 | **47.26** | **8.82** |
| NUMA distribute | 12 | 42.65 | 8.88 |
| NUMA distribute | 14 | 46.44 | 8.66 |

Ten threads is the balanced winner: the 12-thread decode gain is 0.7%, while
10 threads has 10.8% higher prefill.

llama.cpp automatic placement also beats explicit `--tensor-split 1,1`:

| Split | Prefill median | Decode median |
|---|---:|---:|
| automatic | **47.26** | **8.82** |
| explicit 1:1 | 42.72 | 8.37 |

## Context-adaptive launch policy

| Context | `-ncmoe` | KV | Status | Server decode | Minimum free VRAM |
|---|---:|---|---|---:|---:|
| 64K | 36 | GPU F16 | validated startup + generation | 8.02 tok/s | 1.63 GiB |
| 128K | 37 | GPU F16 | validated allocation | short-bench median 8.27 tok/s | 2.01 GiB |
| 262K native | 39 | GPU F16 | validated startup + generation | 8.02 tok/s | 2.76 GiB |
| 1M YaRN | 38 | host F16 | real 1M slot + short generation; full fill not tested | 4.50 tok/s | 2.79 GiB |

Quantized Qwen4Exp KV is not usable in the current branch. `q8_0` triggers the
assertion at `src/models/qwen4exp.cpp:544`. Host F16 KV is therefore used only
for 1M; at short context it reduces decode from 8.82 to 4.72 tok/s.

The 1M server required a private one-line fix in `tools/server/server-context.cpp`:
do not cap `n_ctx_slot` to training context when an explicit RoPE scaling type is
active. After rebuilding, the server reported `n_ctx_slot = 1048576`. A full
one-million-token fill/retrieval benchmark remains required before release
promotion, so the 1M profile is `configured-unmeasured`.

## Speculative decoding status

No compatible DFlash2 checkpoint exists for Qwen3.8-Flash-Next/Qwen4Exp. The
published `incoai/` and `z-lab/` Qwen3.8-27B-DFlash2 checkpoints target the dense
27B model and are incompatible with Flash-Next's hidden size, QSA/GDN state,
hyper-connections, expert layout, and output projection.

The closest exact-target alternatives are native MTP heads:

| Artifact | Size | Runtime boundary |
|---|---:|---|
| `jlkivey/Qwen3.8-Flash-Next-MTP-PR27836-GGUF` Q8_0 | 4.14 GB | llama.cpp PR #27836 only; PR is draft/unstable and tested on Metal |
| `quimmedes/Qwen3.8-Flash-Next-MTP-GGUF` Q4_K_M | 2.79 GB | cafe-llama.cpp only; incompatible tensor layout |
| `ashbash/Qwen3.8-Flash-Next-MTP-Drafter-GGUF` Q4_K_L | 1.94 GB | gmlx/Apple Silicon; explicitly unsupported by llama.cpp |

PR #27836 reports 30–40% code-generation gains but roughly neutral prose, with
about 3.5 GiB extra residency for the Q8 head. It needs a separate physical
campaign because that VRAM cost changes every expert-placement rung above.
No draft was downloaded or promoted.

## Commands

### 64K

```bash
llama-server -m <first-Q4-shard> --jinja \
  -ngl 99 -ncmoe 36 -c 65536 -fa on --parallel 1 \
  --numa distribute -t 10 --load-mode mmap
```

### 128K

```bash
llama-server -m <first-Q4-shard> --jinja \
  -ngl 99 -ncmoe 37 -c 131072 -fa on --parallel 1 \
  --numa distribute -t 10 --load-mode mmap
```

### 262K native

```bash
llama-server -m <first-Q4-shard> --jinja \
  -ngl 99 -ncmoe 39 -c 262144 -fa on --parallel 1 \
  --numa distribute -t 10 --load-mode mmap
```

### 1M candidate

```bash
llama-server -m <first-Q4-shard> --jinja \
  -ngl 99 -ncmoe 38 -c 1048576 --no-kv-offload -fa on --parallel 1 \
  --numa distribute -t 10 --load-mode mmap \
  --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144
```

## Evidence

Machine-readable evidence: `summary.json`

SHA-256: `58563a2f5905e61ceed4c2bbe640a1d9cea670aee76bb973be93708717d71d83`

Raw llama-bench JSON and failure logs are in this directory. The hybrid runtime
catalog references this evidence hash directly.
