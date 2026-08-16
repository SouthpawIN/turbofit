# Qwen 3.8 27B + DSpark hardware benchmark

**Status:** physical evidence is complete only for the available `2× RTX 3090 24 GB` host (48 GB aggregate VRAM). Results for other hardware tiers are intentionally blank rather than estimated.

## Tested configuration

- Main: `Qwen 3.8 27B Q4_K_M + Q4 MTP`
- Auxiliary: `Binary Bonsai 27B Q1_0 + Q4 DSpark`
- Runtime: pinned Turbofit llama.cpp CUDA runtimes
- Host: Linux x86_64, `2× NVIDIA GeForce RTX 3090 24 GB`, CUDA compute capability 8.6
- Driver: `580.159.03`
- Hardware fingerprint: `sha256:aa8f29e3c3917a04f4301806f0b0f7d33e7267d283e1dfb96264e0ef313d926b`
- Prompt: 26-token campaign probe; 128 generated tokens per route
- KV cache: Q4_0 K/V; one slot per server

## Qwen 1M host-offload validation

After portable memory planning was enabled, the Qwen shared-main lane completed a real 1,048,576-token-context launch on the same host:

- Configuration: `Qwen 3.8 27B Q4_K_M + MTP`, shared-main (`auto` auxiliary)
- Allocation: both 24 GB GPUs for model residency plus system RAM for KV via `--no-kv-offload`
- Exact reported context: **1,048,576 tokens**
- Decode: **0.8173 tok/s**, n=1
- Prompt: **2.7409 tok/s**, n=1
- MTP acceptance: **90/110 = 81.82%**
- Peak accelerator memory: **16,113 / 21,767 MiB**
- Raw evidence SHA-256: `bb70236d169f101351b77101d39c7feae4264b6dad418c72043620fabf6ecd5f`

This proves functional 1M context through combined VRAM + host RAM. It does not turn the dedicated DSpark-pair 1M failure below into a pass; that pair remains a separate two-server allocation problem.

## 48 GB results

| Context | Main decode | Aux decode | Main prompt | Aux prompt | Peak VRAM (GPU 0 / GPU 1) | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 64K | **62.59 tok/s median, n=3** | **65.42 tok/s median, n=3** | 66.03 tok/s median | 71.88 tok/s median | 8,445 / 22,085 MiB | PASS |
| 128K | 2.23 tok/s, n=1 | 63.27 tok/s, n=1 | 4.75 tok/s | 58.87 tok/s | 9,917 / 22,593 MiB | PASS, exploratory |
| 262K | 0.78 tok/s, n=1 | 60.62 tok/s, n=1 | 1.99 tok/s | 54.47 tok/s | 12,861 / 22,543 MiB | PASS, exploratory |
| 1M | — | — | — | — | 9,460 / 21,882 MiB observed before failure | FAIL: inference exceeded gateway timeout; auxiliary also attempted an impossible 2,220,051.90 MiB CUDA allocation before capping to its 262K training context |

At 64K, Qwen MTP accepted `91/106` drafted tokens in all three runs: **85.85% acceptance**.

### 64K raw runs

| Main decode | Aux decode | Main prompt | Aux prompt | SHA-256 |
|---:|---:|---:|---:|---|
| 61.8211 | 66.5124 | 69.7045 | 56.2042 | `b3aff15e2f247e814184bd2d9fb30816d14ca58c36572de273817662c209c817` |
| 62.5925 | 64.9995 | 66.0286 | 76.4495 | `0341a2e69a7e484bdeba74d54da1586e7de6a48162ef8ef7f7a70b9a69e17a23` |
| 63.6278 | 65.4208 | 62.4104 | 71.8818 | `0020dc36c37796bc33c9932ef4b4395fbacc32429226c97de02baec4a35b2a76` |

## Hardware-tier coverage

| Turbofit tier | Qwen 3.8 + DSpark physical evidence | Reason |
|---:|---|---|
| 8 GB | Not tested / not a Qwen tier candidate | Qwen 3.8 27B does not fit the tier |
| 16 GB | Not tested / not a Qwen tier candidate | Qwen 3.8 27B does not fit the tier |
| 24 GB | Not exact-topology tested | Qwen Q4+MTP alone peaked at 22,151 MiB, but the DSpark auxiliary requires a second accelerator or offload; the available machine fingerprints as 48 GB |
| 48 GB | **Tested** | Real `2×24 GB` physical run above |
| 64 GB | Not tested | No 64 GB physical host attached |
| 96 GB | Not tested | No 96 GB physical host attached |
| 200 GB | Not tested | No 200 GB physical host attached |
| 300 GB | Not a current Qwen tier candidate | Current 300 GB tournament is DeepSeek V4 Flash only |

## Interpretation

- The 64K lane is viable and fast on dual 3090s: both routes sustain roughly 63–65 tok/s decode.
- Qwen's MTP path remains highly effective at 64K (85.85% draft acceptance).
- Qwen decode collapses at 128K and 262K on this 48 GB topology because the runtime must trade GPU residency for large-context memory; these contexts pass functionally but are not production-speed winners here.
- The 1M configuration is not viable under the present recipe and five-minute gateway timeout.
- No numbers in this report are projected or topology-emulated.
