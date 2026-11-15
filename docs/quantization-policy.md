# Quantization policy — a Q4 ceiling, not a target

TurboFit never selects a quantization above the Q4 class by default. That is a
**ceiling**, not a preference: low quants are first-class. The GSQ-RCO repos ship
genuinely good Q2 and Q3 builds, and an IQ2/IQ3 rung is frequently the right
answer for a small memory band. Nothing in the product ranks, scores, or warns on
quant width.

Implemented once, in `src/turbofit_runtime/quant_policy.py`.

| Class | Examples | Default answer? |
|---|---|---|
| `allowed` | `Q1_0`, `Q2_0`, `Q3_K_S`, `IQ3_XXS`, `Q4_K_M`, `UD-Q4_K_XL`, `IQ4_XS`, `TQ2_0`, `PTQ1_0`, `native-24GB`, `2.788 bpw` | Yes — at or below the Q4 ceiling |
| `allowed-exception` | `Q8_0`, `F16`, `BF16`, `FP16`, `FP32` | Yes, with a printed reason |
| `excluded-by-default` | `Q5_K_M`, `UD-Q5_K_XL`, `Q6_K`, any Q5/Q6/Q7 equivalent | No — only with a recorded `quantization_justification` |

Classification reads the first quant token in the label, so mixed builds are
classified by their routing quant: `Maya-M (IQ2_S routed / Q6_K attention)` is
allowed. A catalog `id`/`name`/`artifact` that *names* an above-ceiling build is
also excluded, so an alias like `…-ud-q5-k-xl` pointing at a Q4 artifact cannot
smuggle a Q5 past the ceiling.

## Exceptions must state why

A Q8/full-precision entry is offered only when it fits the band (reason
`fits-fully-in-vram`) or when accuracy is the point (`accuracy-tier`, set by the
catalog's `accuracy_tier` flag). Every surface carries that reason into its
output.

An above-ceiling entry is admitted only when the catalog entry records:

```json
"quantization_justification": "why this Q5/Q6 build is the right default here"
```

The text is echoed in the output as `justified: …`.

## Where it applies

- **Tournament candidacy / placement** — `tier_fit.plan_candidates()`;
  `scripts/generate-hardware-tier-tournaments` writes per-tier
  `quant_exceptions` (Q8/full precision, with reason) and `quant_excluded`
  (held-out families, with reason) alongside `candidates`.
- **Validation** — `tier_tournament.validate_tournaments()` refuses a tier whose
  candidate is above the ceiling without a catalog justification or without an
  exception record.
- **Recommendation** — `recommend.recommend()` skips Q5/Q6 evidence profiles
  unless justified; `intel_leaderboard.build_leaderboard()` never recommends a
  measured Q5/Q6 configuration, keeping it visible as a non-default option with
  the skip reason; `tier_report.build_tier_report()` tags each candidate with
  `quant_class` and `quant_reason`; `allowed_lineup` never defaults a Q5/Q6 main.
- **Campaign row selection** — `catalog_campaign.build_campaign_matrix()` drops
  above-ceiling rows (with `enforce_quant_policy=True`, the default) and
  `quant_policy_rows()` reports them with reasons. `turbofit-catalog-campaign
  status` prints the excluded count and detail.
- **Fit ladders** — `fit_check.SHARED_TIERS` no longer escalates the quant with
  memory: 24–63 GB stays on `UD-Q4_K_XL`. More RAM buys context headroom, not a
  wider quant. The low rungs (`TQ2_0`, `IQ3_XXS`) are unchanged.

Catalog entries and artifacts for Q5/Q6 builds are **not** deleted; they remain
pinned and runnable when a user explicitly asks for one.
