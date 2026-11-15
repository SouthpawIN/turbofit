# Single-card 96 GB speed evidence: Strata IQ3_S at 414.6 tok/s

External leaderboard evidence, submitted by `Lewfkrad` (lEWFkRAD). Discovery
evidence only; Turbofit promotes runtime profiles after local gated runs pass.

Source: [localmaxxing.com model page](https://www.localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next)
· [full report](https://www.localmaxxing.com/en/models/Qwen/Qwen3.8-Flash-Next/reports/cmwh4ldbr0264mi016zsfwx3s)
· Machine-readable: [`research/localmaxxing-strata-rtx-pro-6000-20261115.json`](../research/localmaxxing-strata-rtx-pro-6000-20261115.json)

## What was measured

One NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB), Core Ultra 9
285K, 64 GB RAM, Ubuntu 24.04. Strata 0.1.56 serving
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` at IQ3_S, eight-slot, with a
DFlash2 drafter (6 draft tokens). Canonical `reasoning-v1` prompt, 256 output
tokens, one warm-up plus three timed iterations, median reported.

| Configuration | Decode | Prefill | TTFT | Rank of 806 |
|---|---:|---:|---:|---:|
| Strata IQ3_S + DFlash2 | 414.6 tok/s | 1524.0 | 197.5 ms | 1 (verified) |
| Strata IQ3_S plain | 413.4 tok/s | 1578.9 | 192.5 ms | 2 (verified) |

Both runs are Verified on the site. The next published Strata IQ3_S entry is
161.1 tok/s on a multi-GPU rig. Same-card other engines publish far lower
medians: vLLM NVFP4 160.5, SGLang NVFP4 156.6, vLLM FP8 about 125.

## KV-cache behavior

Two-step remote probe, reuse classified only from the timed request's own
counters. At every context length from 10K to 80K, 99.95% of prompt tokens
were served from cache; only 19 chat-template tokens were re-prefilled. TTFT
stayed under 474 ms at 80K context.

| Context | Cached / prompt | TTFT |
|---:|---:|---:|
| 10,000 | 10,041 / 10,060 | 110 ms |
| 80,000 | 80,041 / 80,060 | 474 ms |

## Relevance to Turbofit

- The 96 GB tier profile (`runtime-profiles/96gb.yaml`) currently assumes a
  4x24 topology. This is measured single-device 96 GB evidence for the same
  GSQ-RCO IQ3_S artifact family, at 262K-capable hardware.
- The `hardware-tier-tps.md` board has no single-96 GB row. This run fills
  that gap with third-party verified numbers.
- DFlash2 acceptance was 72% (195 of 271 draft tokens) on the canonical
  prompt, with no decode gain at 304 prompt tokens. Speculation gains show
  up in longer agentic contexts, not short leaderboard prompts.

## Reproduction

Exact commands and full methodology are in the report link above and in
[lEWFkRAD/qwen38-rtx-pro-6000](https://github.com/lEWFkRAD/qwen38-rtx-pro-6000).
