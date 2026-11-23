# Model & engine research — what we are leaving on the table

Researched against primary sources. Every claim below carries its source. Where
a number is vendor-reported rather than measured by us, it says so.

## 1. Our Strata engine is 26 point-releases behind

Installed on the test host: **engine 0.1.39**. Current: **0.1.65**, shipped with
app **0.2.19** on 2026-11-22.

The release notes name our exact GPU class:

> **RTX 3090 & 2x RTX A4000 (`sm_86`):** `IQ3_S` / `UD-Q4_K_XL` / `IQ2_XS`
> prefill **+12%..+26%** (1.72x faster TTFT) and decode **+13%..+20%**
> (100% bit-exact).

Source: <https://github.com/Niko1221/Strata/releases/tag/v0.2.19>

Our measured 33.26 tok/s was taken on 0.1.39. A 13–20% decode gain is available
by updating alone, before any configuration change. **Update first, re-measure,
then tune** — tuning an old engine wastes the effort.

## 2. Inkling 2 Mini — the model we were missing

Thinking Machines Lab, released 2026-11-20, **Apache 2.0**.

| Property | Value |
|---|---|
| Parameters | 26.4B total, 3.4B active, **5.4B effective** (looped) |
| Architecture | 24 layers, middle 12 run **three times** → 48 visits/token |
| Experts | 8 of 128 per layer, sigmoid routing |
| Context | **1M tokens** |
| Modality | text + image in, text out (native ViT) |
| Checkpoint | ~20 GB, **NVFP4** layers 0–19 + **MXFP8** layers 20–23 |
| HF | `thinkingmachines/Inkling-2-Mini` |

Sources: <https://thinkingmachines.ai/model-card/inkling-2-mini/>,
<https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thinkingmachines/Inkling-2-Mini>

**Why it matters against our lineup.** It is a *smaller* MoE than the 35B-A3B
class while matching or beating the 27B dense on the published table:

| Benchmark | Inkling 2 Mini | Qwen3.8-27B |
|---|---|---|
| AA Index v5.0 | **20.1** | 20.1 |
| DeepSWE v1.1 | **48.2%** | 42.2% |
| SWEBench Verified | **79.2%** | 77.8% |
| SWEBench Pro v2 Hard | **77.9%** | 73.8% |
| Terminal Bench v4.0 | 5.6% | 5.6% |
| BrowseComp | **79.0%** | 61.6% |
| MCP Atlas | **82.7%** | 77.3% |

Source: <https://thinkingmachines.ai/news/inkling-2-mini/>

Caveat worth keeping: Terminal Bench v2.1 favours the 27B (73.0 vs 71.6), and
AA-Briefcase / GDPVal favour it too. It is not a universal win — it is a win on
agentic coding and search at a fraction of the active parameters.

### Strata packs — this is the low-hardware answer

`strata-inkling` 0.1.65 ships three packs
(<https://huggingface.co/strata-engine/Inkling-2-Mini-Strata>):

| Pack | Size | Fits entirely in VRAM on | Fidelity |
|---|---|---|---|
| **SQ-10** | 10.47 GB / 9.75 GiB | 12 GiB cards, 16 GiB Macs, 780M iGPU — **hybrid from 4 GiB** | — |
| **SQ-15** | 14.35 GB / 13.36 GiB | 16 GiB cards, 24 GiB Macs | 96.68% top-1, KL 0.00889 |
| **SQ-20** | 19.35 GB / 18.02 GiB | **24 GiB+ GPUs**, 32 GiB+ unified | **bit-exact lossless**, 100% top-1, KL 0.00000 |

**SQ-20 is lossless and fits a 24 GB 3090 with room for KV.** That is a native
SF4+MX8 build, not a post-hoc quant — it satisfies the Q4 ceiling as a *native*
precision rather than an exception.

Also included: vision pack (0.64 GB lossless BF16), built-in **W=4 Markov Beam
Tree DSpark** speculative decoding, direct `safetensors` and `.gguf` loading,
and **`--gpus 2` Dual-GPU Expert Parallelism**.

Vendor-reported throughput (RTX 5090): 250.3 tok/s solo, 198.9 on lossless SQ-20,
11,306 tok/s prefill. **Not our hardware** — the 3090 figure is not published for
Inkling, so we must measure it ourselves rather than borrow the 5090 number.

## 3. NVIDIA's quantization-aware-distilled checkpoints

`nvidia/Qwen3.8-Flash-Next-FP4-Local` is described as:

> NVIDIA's quantization-aware-distilled (QAD) FP4 checkpoint, close to bf16.
> NVIDIA trained the NVFP4 and MXFP8 weights against [the teacher].

Source: <https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold>

This is the "trained to be production at low precision" case. It is **not** a
Q4 quant of a bf16 model — the low precision is what was trained. Our quant
policy treats Q4 as a ceiling with exceptions above it; a QAD checkpoint is a
different category and should be first-class, ranked by its measured fidelity
rather than penalised for its bit width.

**Hardware constraint that must be stated:** NVFP4 is a Blackwell-native format.
DGX Spark (GB10) and RTX 50xx have hardware support; **Ampere (our 3090s, sm_86)
does not**. On sm_86 these run through dequantisation or Marlin-style kernels,
so the Spark throughput figures do not transfer. Strata's sm_86 gains are
reported separately and are smaller (+13–20% decode).

## 4. EXL3 is beating our GGUF path on the same card

From the inference-optimisation skill's measured results
(<https://github.com/sybil-solutions/inference-optimisation-skill>):

| Hardware | Model / quant / engine | Result |
|---|---|---|
| **RTX 3090** | Qwen3.8-27B **EXL3 3 bpw + MTP** | **96.2 tok/s** prose / **141.1** code, 1 user |
| **RTX 3090** | Qwen3.6-35B-A3B EXL3 3 bpw + MTP | **251.7 / 357.9 tok/s** |
| **RTX 3090** | Flash-Next EXL3, 256K ctx | MTP 45.5 → **65.6 tok/s** |
| Arc Pro B70 | Qwen3.8-27B EXL3 4bpw, vLLM | 91.2 tok/s vs **25.0** on llama.cpp Q4_K_M |

Our best measured Strata decode on a 3090 is **33.26 tok/s**. Their EXL3 3 bpw
figure on the same GPU class is **96.2**. Even allowing for different prompts
and measurement windows, that is a gap worth investigating rather than
explaining away — and 3 bpw sits inside our Q4 ceiling.

**Action:** EXL3 (ExLlamaV3) belongs in the engine matrix as a serving path, not
just GGUF/llama.cpp, Strata and vLLM.

## 5. The methodology we should adopt

Their skill is not a flag list. The workflow:

1. **Compute the theoretical ceiling first** — bytes/token ÷ bandwidth,
   FLOPs/token ÷ 50% peak, KV budget. If you don't know the maximum you can't
   tell whether 33 tok/s is good.
2. Change **one thing at a time**.
3. Measure against **interleaved controls**, ≥6 samples, **medians** — noise is
   ±5–10%, so most single-run "wins" are drift.
4. Log **failures** too.
5. Gate on **full-context retrieval** (exact-match needles + concurrent full-KV
   soak) before calling anything best.
6. Qualify a clean pinned build before publishing.

Two specifics that map directly onto our open problems:

- **"Speculative decoding is about per-step cost, not acceptance."** Prune the
  draft head's vocabulary; sweep draft length per concurrency and context; keep
  sampling on the GPU. Our MTP work has been chasing acceptance rate.
- Their B390 ceiling analysis showed 100 tok/s needed two useful drafts at ~14%
  acceptance at depth 2 → practical target 88–97. **Compute the ceiling before
  promising a number.**

## 6. What this changes in TurboFit

| Item | Change |
|---|---|
| Strata engine pin | 0.1.39 → **0.1.65**; re-measure before tuning |
| Catalog | add **Inkling 2 Mini** with SQ-10 / SQ-15 / SQ-20 |
| Quant policy | recognise **QAD / native-precision** checkpoints as first-class |
| Engine matrix | add **EXL3** as a serving path |
| Two-card path | Strata **`--gpus 2` Expert Parallelism** (not layer split) |
| Benchmarks | ceilings-first, interleaved controls, medians, full-context gate |
| Hardware honesty | NVFP4 gains are Blackwell; sm_86 figures are separate |

## Not yet verified by us

- Inkling 2 Mini throughput on **our** 3090s — no published sm_86 figure.
- EXL3 3 bpw on our exact host and prompt mix.
- Whether the +13–20% sm_86 decode gain reproduces on our pack and context.
- The X post referenced in the brief was not fetched; the same release notes
  were read from the primary GitHub source instead.
