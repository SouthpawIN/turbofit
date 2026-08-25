# Maple Preview small-device validation

Status: **Auto on `hardware-8gb` at native 64K and 128K**

## Pinned inputs

| Item | Value |
|---|---|
| Model | `stamsam/maple-preview-gguf` |
| Model revision | `0afcb98771d39ab4de25252ee006ea9f9eab1920` |
| Artifact | `maple-tq2_0.gguf` |
| Artifact size | `5,454,482,432` bytes |
| Artifact SHA-256 | `09d219202562dbd17722dc8e3273527a021182ab7f892c2a06aac459a8f3a090` |
| Runtime | `stamsam/llama.cpp` |
| Runtime revision | `9ee03eec62d088a117ab916bbe489e7a3872a21f` |
| Supported contexts | native 64K and 128K only |

Mainline llama.cpp cannot load this architecture or its custom TQ2_0 type.

## Measured results

| Envelope | Context | Peak | Prefill | Decode | Result |
|---|---:|---:|---:|---:|---|
| CPU-only, hard `MemoryMax=8G`, `MemorySwapMax=0`, no GPU, no mmap | 64K | 5.368 GiB | — | 8.165 tok/s | pass |
| CPU-only, hard `MemoryMax=8G`, `MemorySwapMax=0`, no GPU, no mmap | 128K | 5.581 GiB | — | 11.918 tok/s | pass |
| CPU-only quality/tool suite | 128K | 5.623 GiB | — | — | 3/3 tool calls; arithmetic pass |
| CUDA, all layers offloaded, RTX 3090 surrogate | 128K | 6,115 MiB delta | 256.06 tok/s | 82.988 tok/s | pass |

Prompt reuse was observed with `255` cached prompt tokens.

## What this proves

- The Maple process fits below 8 GiB at native 64K and 128K with swap disabled.
- That process envelope also fits inside 16–23 GB RAM-only budgets with substantially more headroom.
- The CUDA working-set delta is below 8 GiB at 128K.
- Basic coherent arithmetic and Hermes-style function calls work.
- Maple must not receive 262K or 1M rows; those exceed the model's native 131,072-token window.

## What this does not prove

- A physical 8 GB GPU has not been tested. The CUDA measurement came from a 24 GB RTX 3090 and is only a residency surrogate.
- A physical 8 GB total-RAM machine has not been tested. The CPU test used an 8 GiB/no-swap cgroup on a larger host; OS and driver behavior on an actual small machine may differ.
- No systematic intelligence benchmark, perplexity comparison, or exact-tier campaign has promoted Maple over Bonsai or Ornith.
- The fork-only runtime is an operational compatibility cost.

## Fit List decision

### Dedicated VRAM

| Capacity | Fit List model |
|---:|---|
| 8 GB | Maple Preview TQ2_0 candidate; Ornith alternative when host RAM can hold offloaded experts |
| 16 GB | Qwen 3.8 27B Unleashed UD-IQ3_XXS |
| 24–95 GB | Qwen 3.8 27B Unleashed UD-Q3_K_XL |
| 96 GB+ | Qwen 3.8 27B 16-bit until Unleashed FP16 exists |

### Integrated / RAM-only total memory

| Capacity | Fit List model |
|---:|---|
| 8–15 GB | Maple Preview TQ2_0 candidate |
| 16–23 GB | Ornith 1.5 35A3B |
| 24 GB+ | Qwen 3.8 27B Unleashed UD-Q3_K_XL |

The Fit List defines eligible candidates. Live topology, free capacity, external pressure, engine compatibility, and exact current-recipe evidence still drive contraction and healing. Existing Auto profiles remain on their last physically proven rungs until Maple passes an exact 8 GB-device campaign.

## Promotion blockers

1. Run the same native 64K/128K server campaign on a physical 8 GB GPU.
2. Run the CPU-only campaign on actual 8 GB and 16 GB RAM-only or unified-memory machines.
3. Record current-recipe throughput, latency, tool-call, cache, and intelligence evidence with immutable hashes.
4. Promote only if Maple beats the incumbent for the same exact topology and operational constraints.

Raw machine-readable evidence: [`../references/results/maple-preview-small-device-evidence.json`](../references/results/maple-preview-small-device-evidence.json)

Infographic: [`../assets/turbofit-maple-fit-list.png`](../assets/turbofit-maple-fit-list.png)
