# TurboFit UX redesign — the character sheet

Status: **phase 1 — static prototype + specification.** Nothing here is wired to the
plugin yet. `dashboard/ux/turbofit-sheet.html` renders from a fixture table that is
labelled as fixtures in the sheet footer. Phase 2 ports the same markup into
`dashboard/dist/index.js` against the live endpoints listed in §9.

Reference: a 1990s Japanese anime car "character sheet" for a Saab 900 Turbo in
`TURBO FIT` livery — cream paper, hard black panel rules, small-caps Latin labels
with Japanese subtitles, a bio block beside a hero vehicle, a row of three views,
and a livery/palette/notes strip signed `— .sOuthpaw`.

Files:

| Path | What it is |
|---|---|
| `dashboard/ux/turbofit-sheet.html` | Self-contained prototype. Inline CSS/JS, no build, no network assets. |
| `dashboard/ux/favicon.svg` | Tach-O mark, 64 px. Linked from the prototype. |
| `dashboard/ux/wordmark.svg` | Horizontal lockup for README/desktop use. |
| `docs/ux-redesign.md` | This document. |

Open the prototype with any static server (`python -m http.server`), not `file://` —
the browser harness blocks `file://`. There is no build step.

---

## 1. Design rules

1. **Four inks, no decoration.** Charcoal `#1B1B1B`, cream `#F2EDE4`, light cyan
   `#7FD4E8`, signal red `#E03A2F`. No gradients except the SVG hatch/dash patterns
   that stand for evidence classes. No photographic assets, no icon fonts, no emoji.
2. **Red is a warning, not a highlight.** Red is used only for needles, the speed
   redline, redline ticks, the release-gate tick, the floor stop, and the
   `NO MEASUREMENT` stamp. It is never used for "selected", "new", or emphasis.
3. **A dial never reads zero for missing data.** No measurement is a state, not a
   value: no needle, no arc, and a dashed stamp that says so.
4. **Provenance is drawn, not written in a tooltip.** Band style and needle style
   encode where the number came from (§8), and every dial repeats it in words.
5. **The layout is a sheet of panels.** Hard 2 px rules, full-bleed panel heads,
   numbered panels `01`–`07` with Japanese subtitles. Not cards on a grid.

Typography: Helvetica-class grotesque for display, `SFMono-Regular`/Menlo/Consolas
for all data, a Japanese system stack (`Hiragino Kaku Gothic ProN`, `Yu Gothic`,
`Noto Sans JP`) for subtitles only. Micro-labels are 8–10 px, `0.10–0.18em`
tracking, uppercase. Numbers are tabular.

### Panel anatomy

| # | Panel | Content |
|---|---|---|
| — | Header | Tach-O wordmark, motto, REF/SHEET/REV with dotted leaders, four status stamps |
| — | Config strip | Three configurations A/B/C as tab buttons; active is ink-inverted |
| 01 | Specifications `仕様` | Family, model id, quant + ceiling tag, context, aux, engine, tier, intel, speed, balanced, evidence, recipe hash |
| — | Hero `車体` | Line-art vehicle, plate number, engine/aux/rung strip |
| 02 | Instruments `計器` | Context, speed, intelligence dials |
| 03 | Shifter Tree `変速` | H-pattern gate, lane label, pressure slider |
| — | Engagement `選択理由` | Engaged gear, why, rejected gears, pressure table |
| 04 | Gear List `編集` | Reorder/rename/add/delete, JSON validate & apply |
| — | Provenance `出典` | Per-reading level and source for the engaged rung |
| 05 | Livery `塗装` | Tach-O patch, cyan flank stripe |
| 06 | Palette `配色` | Four swatches with roles |
| 07 | Notes `注記` | Quant ceiling, honesty rule, signature |
| — | Footer | Reading key (four sample strokes) and endpoint wiring |

---

## 2. The three dials

All dials share a 240° sweep: value fraction `t ∈ [0,1]` maps to a needle angle

```
angle_deg = -120 + 240 * t        // 0° = straight up, clockwise positive
```

Ticks are drawn at radius 86–92 (major 1.7 px, minor 1.0 px), labels at r = 66,
the value band at r = 78 (7 px), and the needle is a 3.6 px-base polygon reaching
r = 76 with an ink hub. The numeric readout sits **below the hub** (y = 166/180/194)
so a needle can never pass under it.

### CONTEXT — tokens, log₂

| Property | Value |
|---|---|
| Range | 64K … 1M, hard floor at 64K |
| Mapping | `t = (log2(tokens) − 16) / 4` → 64K = 0.00, 128K = 0.25, 256K = 0.50, 512K = 0.75, 1M = 1.00 |
| Major labels | 64K, 128K, 256K, 512K, 1M |
| Minor ticks | 32 log-uniform divisions (8 per octave) |
| Floor | Red tick outside the bezel at t = 0 + `FLOOR 64K` label. Nothing is drawn below it; the chain's `context_floor` is 65,536. |
| Band | Cyan from 0 to the engaged rung's window |
| Empty state | No needle, no band, `NO MEASUREMENT` stamp + `RESIDENCY RELEASED` (yield only) |

This dial shows a **configured** value (the engaged rung's window), not a
measurement. It is the one dial allowed to carry a reading without a benchmark,
because a configured window is a fact about the process, not a claim about quality.
The caption and the `CONFIGURED · NOT MEASURED` subline say so.

### SPEED — tokens/second

| Property | Value |
|---|---|
| Range | 0 … 60 linear |
| Ticks | Major every 10 (labels 0…60), minor every 1 |
| Redline | Red arc at r = 99 from 50 to 60; ticks and labels above 50 are red |
| Why 50 | `balanced = 200·i·s/(i+s)` with `s = min(tok/s ÷ 50, 1)` — the balanced score saturates at 50 tok/s, so past 50 the dial is measuring comfort, not score |
| Overflow | Values > 60 peg the needle at the end stop and print `60+` with `PEGGED · <value>` in red. Nothing is truncated silently. |
| Empty state | No needle, no band, `NO MEASUREMENT` |

### INTELLIGENCE — Hermes Index, 0–100

| Property | Value |
|---|---|
| Range | 0 … 100 linear |
| Ticks | Major every 20 (labels 0…100), minor every 2.5 |
| Band | From 0 to the measured score; **style encodes provenance** (see §8) |
| Release gate | Red tick at 60, labelled in the caption as *policy, not a measurement* |
| Empty state | No needle, no band, `NO MEASUREMENT` |

The band is the "marker band" from the brief. It is deliberately **not** a quality
zone: a screening result at 70 is still drawn hollow, and a promotion result at 45
is solid. The score never changes the band style; only evidence does.

### Gauge caption contract

Every dial prints four things under it, in this order:

1. the reading and unit (`46.9 / 100`),
2. an evidence chip (`SCREENING · n=1` / `MEASURED · THIS HOST` / `EXTERNAL · NOT
   RE-MEASURED` / `CONFIGURED` / `NO MEASUREMENT`),
3. the source sentence, including recipe hash or PR reference,
4. the scale line, including redline and gate meaning.

---

## 3. The shifter tree

**Semantics.** Gear 1 is the ceiling rung — the most memory, the widest window.
Increasing gear number means a lower rung. Reverse is `YIELD`: the process releases
residency entirely. The gate is drawn as a real H-pattern: three columns of two for
six gears, a horizontal neutral rail, and an outboard reverse gate behind a
`LIFT TO UNLOCK` bracket, because yield is never an inline downshift.

**Engagement rule (first fit, not best fit).** Given per-card headroom `H`:

```
for gear in gears_in_user_order:
    if H >= gear.needs_mb_per_card: engage(gear); stop
engage(YIELD)
```

The ceiling is not a suggestion: if gear 1 fits, lower gears are not considered even
if they would score better. That is the product's stated policy (pressure may lower
the effective rung, never raise the ceiling), and the engagement panel prints the
comparison that decided it.

**Direction is explicit.** The dashed arrow under the gate reads
`UPSHIFT — NATIVE WINDOW` → `DOWNSHIFT — SHED CONTEXT, THEN CLASS`. The chain's own
policy is context first: halve the window before swapping model, reach for a smaller
class only when the floor still does not fit, yield last.

**Pressure is simulated in the prototype.** The slider sets used memory per card as
a percentage of that configuration's card size; the sheet labels it `MEMORY PRESSURE`
with `模擬` (simulated). It is a demonstration control, not a product control — the
real value comes from `dials.memory_pressure` in `/status`. In phase 2 the slider is
deleted and the panel becomes read-only telemetry.

---

## 4. Shifter document — `turbofit.shifter/v1`

The editable object is a **presentation overlay** on `references/fallback-chain.json`
(`turbofit.fallback-chain/v1`). It owns order, names, and lane; it does **not** own
model identity, quant, context, memory requirement, or evidence. Those are projected
from the chain and the catalog and are read-only in the editor.

### Schema

```json
{
  "schema": "turbofit.shifter/v1",
  "name": "2×24 GB · dedicated — Qwen 3.8 27B GSQ-RCO",
  "lane": "dedicated",
  "context_floor": 65536,
  "updated": "2026-11-15T09:41:02.118Z",
  "gears": [
    {
      "id": "a0",
      "label": "GSQ-RCO / 262K",
      "model": "qwen3-8-27b-gsq-rco-iq3s",
      "quant": "UD-Q3_K_XL",
      "context": 262144,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 17400,
      "evidence": "measured"
    },
    {
      "id": "a1",
      "label": "LFM 24A2B / 131K",
      "model": "lfm-24a2b",
      "quant": "Q4_K_M",
      "context": 131072,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 15000,
      "evidence": "candidate"
    },
    {
      "id": "a2",
      "label": "GSQ-RCO / 64K",
      "model": "qwen3-8-27b-gsq-rco-iq3s",
      "quant": "UD-Q3_K_XL",
      "context": 65536,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 12200,
      "evidence": "measured"
    },
    {
      "id": "a3",
      "label": "BONSAI PTQ1 / 131K",
      "model": "ternary-bonsai-2-27b-ptq1",
      "quant": "PTQ1",
      "context": 131072,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 11200,
      "evidence": "measured"
    },
    {
      "id": "a4",
      "label": "MAPLE TQ2 / 128K",
      "model": "maple-preview-20b-a1b",
      "quant": "TQ2_0",
      "context": 131072,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 8000,
      "evidence": "portable-fit"
    },
    {
      "id": "a5",
      "label": "ARETE Q4 / 131K",
      "model": "arete-27b",
      "quant": "Q4_K_M",
      "context": 131072,
      "engine": "llama.cpp",
      "aux": "maple-preview-tq2",
      "needs_mb_per_card": 7600,
      "evidence": "candidate"
    }
  ],
  "reverse": { "id": "yield", "label": "YIELD", "kind": "yield" }
}
```

### Field ownership

| Field | Owner | Editable | Notes |
|---|---|---|---|
| `schema` | product | no | Must equal `turbofit.shifter/v1` |
| `name` | user | yes | Sheet title only |
| `lane` | user (constrained) | yes | `dedicated` or `unified`; must exist in the chain |
| `context_floor` | policy | no (below 65536) | May be raised, never lowered |
| `updated` | product | no | Written on save |
| `gears[].id` | chain | no | Stable id from `Level.id` (`d0`, `u3`, …) |
| `gears[].label` | **user** | yes | 1–28 chars, uppercase display |
| `gears[].model` | catalog | no | Must resolve in `model-catalog.json` |
| `gears[].quant` | catalog | no | Exactly as published |
| `gears[].context` | chain | no | One of 64K/128K/256K/1M |
| `gears[].engine` | catalog | no | `llama.cpp`, `prism`, `strata`, `vllm` |
| `gears[].aux` | policy | no | `maple-preview-tq2` only |
| `gears[].needs_mb_per_card` | measured/chain | no | From `Level.needs_mb_per_card` |
| `gears[].evidence` | chain | no | `measured` / `portable-fit` / `candidate` |
| `reverse` | product | no | Always present, always last |

**Measurements are not in this document.** `intel` and `tps` are attached at render
time from the evidence store keyed by `(configuration_id, gear.id)`. That is why the
JSON export strips them: a pasted number must never be able to become a reading.

### Validation (implemented in the prototype, `validateDoc`)

| Rule | Failure message |
|---|---|
| `schema == "turbofit.shifter/v1"` | `schema must be 'turbofit.shifter/v1'` |
| `gears` non-empty | `gears must be a non-empty array` |
| `context_floor >= 65536` | `context_floor may not go below 65536 tokens` |
| each gear has `id,label,model,context,needs_mb_per_card` | `gears[i].<field> is required` |
| `gears[i].context >= context_floor` | `gears[i].context <n> is below the floor` |
| memory requirement is non-increasing down the order | `gears[i] needs more memory than the gear above it — the ladder must not climb` |
| `aux == "maple-preview-tq2"` | `gears[i].aux must be maple-preview-tq2 (the only auxiliary)` |
| `lane ∈ {dedicated, unified}` | `lane must be 'dedicated' or 'unified'` |

The reorder buttons enforce the same monotonic rule interactively: an illegal move is
reverted and the editor prints `ORDER REJECTED — THE LADDER MUST NOT CLIMB`.

### Persistence

| Surface | Location | Lifetime |
|---|---|---|
| Prototype | `localStorage["turbofit.shifter.v1:<configuration_id>"]` | per browser profile |
| Product (phase 2) | `~/.config/turbofit/shifter.json` | per user |
| Export | the JSON block in panel 04 | copy/paste |

Read precedence for phase 2: user file → shipped default derived from
`references/fallback-chain.json` → empty state (no gears, yield only). A missing or
invalid file is never repaired silently: the sheet shows the last valid default and a
`REJECTED` message.

### Mapping to the existing chain document

| shifter | `fallback-chain.json` | transform |
|---|---|---|
| `gears[]` order | `lanes[lane].levels[]` | user order; default = chain order |
| `id` | `levels[].id` | verbatim |
| `label` | — | user; default `model / context` |
| `model` | `levels[].model` | verbatim |
| `quant` | `levels[].quant` (optional) | catalog lookup when absent |
| `context` | `levels[].context` | verbatim |
| `engine` | `levels[].engine` | verbatim |
| `needs_mb_per_card` | `levels[].needs_mb_per_card` | verbatim |
| `evidence` | `levels[].evidence` | verbatim |
| `reverse` | `kind == "yield"` level | last level, never reorderable |

A gear whose `model` is no longer in the chain is shown struck through and marked
`UNBOUND`; it can be deleted but never engaged. Renaming cannot invent a rung.

---

## 5. Customization model

Three layers, one editor:

1. **Evidence layer (read-only).** Model, quant, context, engine, aux, memory
   requirement, evidence class. Rendered as grey meta text under each name. The
   editor has no input for them, by design.
2. **Presentation layer (user-owned).** Order, display name, lane selection, whether
   a gear is present at all. These are the only things the JSON document carries.
3. **Policy layer (constrained).** `context_floor` may be raised but not lowered;
   yield must remain enabled and last; the memory ladder must stay monotonic. These
   constraints are the difference between a preference and a lie about physics.

Operations and their effects:

| Operation | Effect on engagement | Effect on storage |
|---|---|---|
| Rename | none | label rewritten, saved |
| Move up/down | first-fit result may change immediately | order saved |
| Add gear | inserted at the bottom with 0.85× the previous requirement | saved |
| Delete gear | removes the rung; yield remains | saved |
| Change lane | reloads that lane's chain, then re-applies saved order | saved per lane |
| Import JSON | validates, then replaces order/labels | saved on success only |
| Reset | restores the chain default | saved |

Live behaviour: the prototype recomputes engagement on every edit, so the sheet shows
the consequence of an edit immediately. In phase 2 the same document is handed to the
controller as an ordered rung preference list; the controller keeps authority to
refuse a rung that fails its own fit check. **The editor cannot promote a rung.**

---

## 6. What the sheet does not do

- It does not claim a reading for a recipe that has none. The `B` configuration is
  a full no-measurement tier: two dials empty, balanced withheld, stats table red
  `NO MEASUREMENT`, and the tree still fully usable.
- It does not blend provenance. An external 414.6 tok/s number is drawn as a ghost
  band and a pegged needle, labelled `EXTERNAL · NOT RE-MEASURED`; it is never mixed
  with a local number to make an average.
- It does not present quant as a ladder. Q4 is drawn as **AT CEILING**; Q3-class
  GSQ-RCO builds are drawn `BELOW CEILING · FIRST CLASS`; Q8/FP16 are described as
  exceptions in the notes. No row says "higher is better".
- It does not use a settings card as the primary surface. The unit of the page is a
  configuration, not a form.

---

## 7. Legibility

| Width | Behaviour |
|---|---|
| ≥ 1180 | Sheet at 1340 max, two-column bio/hero, three dials, shifter beside engagement |
| 1040–1180 | Wordmark 54 px, bio column 410 px, dials unchanged |
| 900–1040 | Bio stacks above hero; shifter stacks above engagement and editor; livery 2-up + notes full width |
| ≤ 900 | Config tabs become rows; body padding 8 px; dials remain three-up with reduced type (smallest printed number 8 px) |

The sheet has no horizontal scroll at 900 px (verified in the browser), and the three
dials keep their numbers/ticks — only captions lose a step in size. Print rules drop
the pressure slider and the JSON editor, because neither belongs on paper.

---

## 8. Provenance rules

Every number on the sheet carries a level. The level decides how it is drawn and
which words accompany it.

| Level | Meaning | Band | Needle | Chip | Allowed in composite? |
|---|---|---|---|---|---|
| `promotion` | Hash-bound evidence at promotion grade for this exact recipe | solid 7 px | solid | `PROMOTION` | yes |
| `measured-here` | Measured on this host, this recipe, hash-bound raw result | solid | solid | `MEASURED · THIS HOST` | yes |
| `screening` | Screening harness, n = 1 attempt, not promotion grade | dashed 3/4 | solid | `SCREENING · n=1` | yes, but labelled |
| `external-verified` | Hash-bound evidence captured elsewhere; not re-measured here | dashed 1/5, 2.5 px | solid | `EXTERNAL · NOT RE-MEASURED` | no |
| `derived` | Computed from other readings | n/a | n/a | `DERIVED` | n/a |
| `declared` | Configuration fact (window, engine, quant), not a measurement | cyan | solid | `CONFIGURED` | n/a |
| `none` | No evidence exists for this recipe | none | none | `NO MEASUREMENT` | no |

Rules:

1. **Unknown is not zero.** `none` removes the needle and the arc. Zero is a
   measurement and is drawn as one.
2. **A reading is bound to `(main, aux, quant, context, recipe_sha256)`.** Changing
   context invalidates speed and intelligence together. The prototype demonstrates
   this: gear 3 of configuration A is the same weights at 64K, and both dials go
   empty rather than inheriting the 262K numbers.
3. **Composites withhold.** `balanced` requires both intelligence and speed at
   allowed levels; otherwise it prints `WITHHELD`, not a partial score.
4. **External evidence never becomes local.** It is displayed with a ghost band and
   an explicit chip, and it is excluded from ranking on this host.
5. **Thresholds are policy, not data.** The 50 tok/s redline comes from the balanced
   formula's cap; the 60 release gate is a promotion policy. Both are labelled
   `POLICY, NOT A MEASUREMENT` so nobody reads them as measured capability.
6. **Screening is never called promoted.** The sheet has no state that upgrades a
   screening row; that transition is owned by the campaign, not the UI.
7. **Dates and hashes are printed.** Every reading names its date and either the
   recipe hash or the external evidence reference, so a stale number is visible.

---

## 9. Endpoint wiring

Current endpoints exist; the sheet is designed so each panel maps to one of them.
Nothing below requires a backend change except **one new read** (marked ⚠).

| Sheet element | Endpoint | Field(s) | Transform / rule |
|---|---|---|---|
| Status stamps (header) | `GET /status` | `gateway.reachable`, `tailnet.connected`, `provider.registered`, `serve` availability | 4 stamps; unknown renders dashed |
| Config strip (A/B/C) | `GET /hardware-tiers` + `GET /fit` | `tiers[].capacity_gb`, `topology`, `status`, `recommendations.*` | one tab per tier that has a recommendation; `no-measurements` tiers still appear |
| Bio: family / model / quant / context / aux / engine | `GET /fit` | `tiers[].recommendation.main`, `main_family`, `context`, `auxiliary.id`, `engine.name` | aux is always `maple-preview-tq2`; engine flavour shown as sub-label |
| Bio: tier + topology | `GET /hardware-tiers` | `tiers[].capacity_gb`, `tiers[].topology.topology` | `48 GB · 2×24` |
| Bio: intelligence | `GET /fit` | `tiers[].recommendation.intelligence_score` | level `screening`/`promotion` from ⚠ readings store |
| Bio: speed | `GET /fit` | `tiers[].recommendation.throughput_tps` | level as above; no value → `NO MEASUREMENT` |
| Bio: balanced | derived | `intelligence_score`, `throughput_tps` | `200·i·s/(i+s)`, `s = min(tps/50, 1)`; withheld if either missing |
| Bio: evidence + recipe | `GET /tournaments` / `GET /fit` | `winner.evidence`, `production_recipe_sha256` | shortened hash in the footer, full hash in provenance |
| Hero plate | `GET /fit` | `recommendation.main_family`, `context` | plate number `TF-<tier>` |
| Context dial | `GET /status` → `dials.chain` | `levels[].context`, `context_floor`, `active_level` | configured value of the engaged rung; cyan band |
| Speed dial | ⚠ readings + `GET /status` | `usage.tps` (live) or per-rung measured TPS | live TPS is always `measured-here` and timestamped; pegged above 60 |
| Intelligence dial | ⚠ readings | per-rung `intelligence_score`, `benchmark_level` | no fallback to catalog or vendor scores |
| Shifter gears | `GET /status` → `dials.chain.levels` | `id`, `model`, `engine`, `context`, `needs_mb_per_card`, `evidence`, `fits_now`, `is_active` | user order overlay (§4) |
| Engagement cause | `GET /status` → `dials.memory_pressure` | `headroom_mb`, `ratio`, `state`, `cards[]` | first-fit comparison printed |
| Gear editor meta | `GET /available-models` | `models[].family`, `quantization`, `engine`, `contexts`, `present`, `auxiliary_default` | read-only half of each gear; `present` drives the unbound/incomplete flag |
| Provenance panel | ⚠ readings | `{value, level, source, measured_at, host, recipe_sha256}` | one row per reading |
| `POST /configure` | — | `fallback_chain`, `profile` | still the save path for provider settings; the shifter document is a separate file |
| `POST /shift` | — | `target` = `up`, `down`, `<model>`, `intelligence`, `balanced`, `speed` | moves the literal rung; the gate animates to the new `active_level` |

⚠ **Missing read.** `/fit` carries one recommendation per tier, not evidence per rung.
The dials need, for the engaged rung:

```json
{
  "configuration_id": "qwen3-8-27b-gsq-rco-iq3s--auto--262k",
  "readings": {
    "intelligence": {
      "value": 46.875,
      "level": "screening",
      "suites": ["deep-swe", "agentic-pair"],
      "attempts": 1,
      "measured_at": "2026-08-23T05:38:24Z",
      "host": "omarchy 2×3090",
      "recipe_sha256": "sha256:2175ca09…"
    },
    "speed": {
      "value": 39.39,
      "level": "measured-here",
      "measured_at": "2026-08-23T05:38:24Z",
      "host": "omarchy 2×3090",
      "recipe_sha256": "sha256:2175ca09…"
    }
  }
}
```

Recommended shape: `GET /api/plugins/turbofit/readings?configuration_id=…` reading
`references/intelligence-scores.json` (which already has `benchmark_level`,
`measurements[]`, `production_recipe_sha256`). The prototype's `READING_REGISTRY`
is exactly this contract, hard-coded.

Until that exists, phase 2 can source intelligence from `/fit` and show speed as
`measured-here` only when `usage.tps` is live, with the dial empty otherwise. That is
narrower but still honest — and it is the fallback the sheet already supports,
because configuration B has no readings at all.

---

## 10. Replacing the current cards

`dashboard/dist/index.js` is 559 lines and one component. Its nine cards are:

| Current card | Fate |
|---|---|
| Setup (selection mode, model, aux, context, backend, endpoint, ports) | Moves behind a `SETUP` drawer, not a card. Manual selection becomes the config strip + gear editor. |
| Multimodal models | Moves into the drawer as a compact table. Not part of the sheet. |
| Active policy | Panel 01 evidence footer + provenance panel |
| Stable routes | Header `PROVIDER` stamp + provenance rows |
| Fit · what to run on this machine | **The sheet itself** (config strip, bio, dials) |
| Hardware tiers · speed versus intelligence | Config strip + dials; ranking text moves to engagement |
| Leaderboard | A collapsed table below the sheet (`BENCHMARK TABLE`), not a primary panel |
| Hardware-tier tournaments | Same table, evidence column |
| Auxiliary candidates | Deleted: there is one auxiliary, stated in panel 01 and the notes |

Migration in four steps, each shippable:

1. **Tokens.** Copy the prototype's `:root` values into `dashboard/dist/style.css` as
   `--tf-ink`, `--tf-cream`, `--tf-cyan`, `--tf-red`; keep the existing `tf-` prefix.
   `dashboard/index.html` is a separate dark stand-alone mock: either retire it or
   re-skin it, but do not keep two design languages in one product.
2. **Components.** Port the sheet as `Sheet`, `ConfigTabs`, `Bio`, `Hero`, `Dial`,
   `Shifter`, `Engagement`, `GearEditor`, `Livery`, `Palette`, `Notes`. Keep
   `SDK.fetchJSON` and the existing `/configure` save path. The gauge SVG builders
   translate almost line-for-line; `polar`/`arc`/`needlePath` are pure functions.
3. **State.** One `useReducer` with `{configId, pressure, shifterDoc}`. The shifter
   document load/save moves to the plugin (`~/.config/turbofit/shifter.json`) and is
   fetched with the rest in the existing `Promise.all`.
4. **Delete.** The nine `Card` blocks, `tf-grid`, `Field`, `Toggle`, the raw
   fallback-chain textarea, and the `.tf-*` dark rules that only those blocks used.
   Keep the textarea's *semantics* (direct JSON editing) as the gear editor's JSON
   block, now schema-validated.

Acceptance for the port: at 1280 and 900 px the sheet renders live `/status` and
`/fit`; a tier with `no-measurements` shows two empty dials; an unreachable gateway
shows dashed stamps; and no dial can display a number whose source is not printed
under it.

---

## 11. Fixtures used in the prototype

| Config | Source of the numbers | State demonstrated |
|---|---|---|
| A — 48 GB, Qwen 3.8 27B GSQ-RCO `UD-Q3_K_XL`, 262,144 | `references/intelligence-scores.json`: intelligence 46.875, balanced 58.778, tok/s 39.392, `benchmark_level: screening`, 2 suites, n = 1 | Populated dials, screening band, first-fit engagement |
| B — 16 GB, Arete 27B `Q4_K_M`, 131,072 | `references/hardware-tier-report.json`: tier has no measured winner | No-measurement tier: context configured, speed/intel empty, balanced withheld |
| C — 96 GB, Qwen 3.8 Flash Next `IQ3_S`, 262,144 | Strata 1×96 profile evidence (414.6 tok/s, PR #83); no intelligence run | External-only reading, pegged needle, empty intelligence dial |

Gear memory requirements and contexts come from `references/fallback-chain.json`
(`d0`–`d6`, `u0`–`u4`). Gear order in A is re-sequenced to start at the selected
rung so the sheet reads as one configuration; that ordering is fixture data, not a
policy change.

---

## 12. Open decisions

1. Does the shifter document get its own endpoint (`GET/PUT /shifter`) or ride along
   in `/status` as `dials.chain.presentation`? The former is cleaner; the latter
   avoids a second round-trip on load.
2. Should `context_floor` be user-raisable at all? It is today, but a raised floor
   makes some rungs unreachable, and the UI would have to grey them.
3. The balanced score's 50 tok/s cap is baked into the gauge; if the cap changes,
   the redline must move with it. That constant should be exported once, not copied.
4. Do we keep the simulated pressure slider in the shipped product as a "what if"
   affordance, or delete it? It is useful for explaining the ladder, but it is not
   telemetry and must never look like it.
