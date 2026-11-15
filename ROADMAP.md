# TurboFit — Roadmap (source of truth)

The source-of-truth map of TurboFit: the two ladders, what is built, and what is open.
The deep data lives in the Python (catalogs, profiles, policy); this file is the map.
Status is grounded in the committed tree and the green test suite (last full run:
**864 passed, 7 skipped**).

## The two ladders

1. **Hardware ladder** — memory pool (unified vs. dedicated) + backend (metal / cuda /
   rocm) → a **hardware profile** (how much model the machine actually holds, with UMA
   compensation for unified memory).
2. **Model ladder** — quality vs. speed → a **model**, anchored on **q95**
   (`qwen3.8-flash-next`), down through dense / MoE / small tiers by family.

Everything in the project hangs off these two ladders: the catalog, the profiles, the
allowlist, and the recommendation all map *hardware → profile → model*.

## Current state

- **Version:** 2.4.0 (released — two-ladder surface).
- **Runtime:** ~60 modules in `src/turbofit_runtime/` behind the `turbofit` CLI
  (`runtime_cli`), covering hardware, catalog, fit, allowlist, recommendation,
  benchmarks, campaigns, the agent tool surface, and the install/verify path.
- **Tests:** 864 passed, 7 skipped (last full run).
- **Recommendation bridge (`hermes_fit`):** built and verified end-to-end this cycle —
  **non-blocking** (estimator/predicted-based, *no* live model call, ~40 ms),
  deterministic, JSON-serializable. It uses the built-in Hermes local runtime when
  present (`probe_budget` + `catalog.predicted_decode_tok_s` +
  `catalog.recommended_entry`) and degrades to TurboFit's own allowlist + ladders on
  hosts without it. It never hangs on a model.

## Built (present in tree, covered by the green suite)

- Hardware inventory + **hardware profiles** (manual + auto probe; UMA compensation).
- **Model catalog** + quality ladder (q95 anchor; speed/quality tiers by family).
- **Allowlist / ban overlay** (`allowed_lineup`) — per-role, per-platform.
- **Fit ladders** (`unified/moe`, `dedicated/dense`) — the two memory regimes.
- **Recommendation policy + controller** + `hermes_fit` bridge + CLI `recommend`.
- **Benchmark pipeline** (`bench.py`, `benchmarks.json`, `turbofit_list.py`) — structure
  present; local numbers pending (see open gaps).
- Campaign / agent tool surface, runtime gateway, install/verify (`after-install.md`).
- Skills: `turbofit`, `turbo-model-store-cleanup`, `turbofit-model-fit`,
  `turbofit-repository-updates`.

## Open — the rest of "all of TurboFit"

Ordered by what unblocks the most:

1. **Local benchmarks → real TurboFit List numbers** (the #1 evidence gap). Most
   TurboFit List rows are still `pending benchmarks`; the measured decode/TTFT numbers
   need to be filled in from real runs on the Mac and Omarchy. This is what makes the
   List a genuine measured table instead of a predicted one.
2. **DeepSeek V4.1 (1017) native engine** — still pending (engine + fit data).
3. **macOS / Metal engine** for Flash Next and large unified boxes — no macOS engine yet
   (Apple currently runs CPU-only); q95 on Apple is the convergence target.
4. **Full local multimodal** — 2-GPU frontier; out of scope until the single-GPU path is
   solid.

## How to verify (fast, no model needed)

```bash
cd ~/turbofit

# Full suite (expect 864 passed, 7 skipped)
.venv/bin/python -m pytest tests -q

# Non-blocking recommendation (standalone path, deterministic; ~40ms)
PYTHONPATH=src .venv/bin/python -c "
from turbofit_runtime import hermes_fit
import types
fp = types.SimpleNamespace(total_usable_memory_mb=32*1024, system_ram_mb=32*1024,
                           memory_pool_kind='unified', backends={'metal'})
print(hermes_fit.build_recommendation(fingerprint=fp).to_dict())
"
```

The recommendation path is **proven non-blocking**: `scan()` and `build_recommendation()`
use `catalog.predicted_decode_tok_s(...)` (a physics estimator) and
`catalog.recommended_entry(...)` (the built-in's pick) — **not** a `/chat/completions`
call, no live probe, no wait for a model response.
