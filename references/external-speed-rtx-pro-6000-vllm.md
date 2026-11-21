# External speed evidence — RTX PRO 6000 (96 GB), vLLM, Inkling 2 Mini

Discovered 2026-11-21. Single RTX PRO 6000 Blackwell (SM120), vLLM
(`vllm/vllm-openai:inkling-2-mini`), served as `inkling2-mini`.

## Current config: DSpark spec 7, probabilistic (S7-prob)

| Metric | Value |
|---|---|
| C1 decode by frame (prose / code / list / agent) | 156.5 / 372.4 / 224.7 / 230.6 tok/s |
| C1 mean | 246.0 tok/s |
| C4 per-stream | 152.8 tok/s |
| C4 aggregate | 274.5 tok/s |
| Accepted-vs-drafted (last) | 2.76 accept, 160.9 tok/s accepted, 638 tok/s drafted |
| Boot time | 120 s |
| Tool calls | 5/5 |

Bench evidence (FP8 KV, no spec): C1 decode 159 tok/s, C4 aggregate 211 tok/s, C8 aggregate 414 tok/s, ~21.4K tok/s cold prefill at 22K prompt, ~11.4K tok/s at 101K prompt, tool calls 5/5. With FP8 KV (current): C1 168 tok/s, C4 aggregate 213 tok/s, C8 aggregate 429 tok/s, ~21.8K tok/s cold prefill at 22K prompt, tool calls 5/5.

Sweep winner was **S7-prob-CG128** (accept 3.14); S7-prob was the recommended default and is what the profile carries. Spec 0 (no spec) is the 167 tok/s C1 baseline.

## Notes

- Cold prefill is vLLM's strength on this rig: 9-22K tok/s vs Strata/Basalt (~1.6K tok/s prefill, ~10x gap). On cold 80K prompts this is the difference between a ~50 s first token (Strata) and ~3-4 s (vLLM).
- FP8 KV cache is load-bearing for 262K context on one 96 GB card.
- `--reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice` are required for agentic work; tool calls verified 5/5 both with and without spec.
- The S7-prob sweep is authoritative for the current default; the fp8kv bench is the prefill/tool-call receipt.

Evidence payload: `research/vllm-inkling2-mini-sweep-20261121.jsonl` (sweep) + `research/vllm-inkling2-mini-rtx-pro-6000-20261121.txt` (fp8kv bench).
