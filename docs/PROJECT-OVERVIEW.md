# TurboFit — project overview

Written from the committed tree, the running deployment, and live verification on
2026-11-03. Where the code and the docs disagree, the disagreement is listed rather than
smoothed over (§10).

## 1. What it is

TurboFit is a **local-inference provider for Hermes Agent** with a hard opinion: the machine's
memory decides which model runs, and pressure from *other* GPU work decides how long it keeps
running. It ships as a Hermes plugin (`plugin.yaml`: tools `turbofit_status`,
`turbofit_configure`; command `/turbofit`), a sidecar gateway on `:8091`, an adaptive
controller, and a lot of measured evidence.

It is not a model server. It is the **policy and evidence layer** around whatever serves the
model — llama.cpp built through its own pinned runtimes, the Strata engine, or an external
server it only reads.

## 2. Where it runs (deployment shape)

```
Hermes Agent  ──provider custom:turbofit──▶  TurboFit gateway  :8091
                                              │  auto / active:main / active:aux
                                              │  tailnet federation (TURBOFIT_PEERS)
                                              ▼
                                    route-state.json ──▶ backend (llama.cpp or Strata unit)
                                              ▲
        systemd --user: turbofit-controller ──┘  (probe → policy → reconcile → publish)
        systemd --user: turbofit-gateway.service
        systemd --user: turbofit-strata.service   (a rung, when declared as a unit)
```

- **VPS container (this box)**: no GPU. Runs the gateway as a federation front end only —
  `TURBOFIT_PEERS` lists the MacBook and omarchy gateways, reached through tailscaled's
  userspace HTTP proxy. Its keep-alive wrapper lives outside the repo
  (`/opt/data/bin/turbofit-gateway-up.py`), because it is deployment glue, not product.
- **omarchy (2× RTX 3090)**: the real host. Controller + gateway + Strata unit; profile
  `omarchy-2x3090` with rungs `strata-flashnext-32k` (unit-managed) → `iq3s-262k` → `iq3s-65k`.
- **MacBook**: gateway + native Metal/MLX runtimes, federated into the same surface.
- **Phone**: consumes the tailnet gateway; does not serve models.

State lives in `~/.local/state/turbofit/`: `controller.json` (adaptive state),
`controller-status.json` (last tick + `degraded`/`last_error`), `runtime-state.json` (routes),
`native/{main,aux}.json` (owned PIDs). Config lives in `~/.config/turbofit/`: `selection.json`,
`manual-profile.yaml`, `manual-runtime-resolutions.json`, `manual-rung-requirements.json`,
`controller.env`.

## 3. The authority chain

Seven authorities, in this order. Nothing lower may override something higher.

1. **Turbofile** (`runtime_profile.py`, `runtime-profiles/*.yaml`) — portable profile: hardware
   constraints, policy numbers, roles, and an ordered list of rungs. Each rung carries
   `id`, `context`, `aux_mode` (`shared-main` | `dedicated` | `api`), an evidence digest, and
   optionally `launcher: systemd-unit` + `unit`. No paths, no secrets, no device indices.
2. **Hardware fingerprint** (`hardware.py`) — physical topology and total usable memory, never
   transient free memory. `2x24` and `1x48` are different classes.
3. **Pressure snapshot** (`pressure.py` + `pressure_probe.py`) — ownership-aware: what is ours
   (reclaimable), what is external (untouchable), desktop baseline, safety reserve. `nvidia-smi`
   process attribution decides which is which; an external process ≥ 512 MiB flags the card
   `contended`.
4. **Pure policy** (`policy.py`, no I/O) — dwell, hysteresis, cooldown, flap quarantine,
   expansion margin, and one decision per tick: `none` | `activate` | `rollback` | `release`.
5. **Native backend** (`native_backend.py`) — the only thing allowed to start or stop a local
   process: PID-and-command-verified children it launched, or a declared systemd unit.
6. **Reconciler** (`reconciler.py`) — drain → activate → verify → publish routes; on failure,
   restore the previous rung and record the failure.
7. **Gateway route state** (`routes.py`, `scripts/turbofit-gateway.py`) — the backing targets
   behind `auto`, `active:main`, `active:aux`; reloaded by the gateway without a restart.

Loop: `controller --once` = probe hardware → synchronize selection + persisted state → probe
pressure → `policy.reconcile` → reconcile/transition → publish routes → save state. Every 2 s.

Pathways that are **not** authority: `scripts/serve`, `scaling-watcher.py`, `vram-watcher.py`,
`turbofit-mesh-gateway.py`, `turbofit-resolver.py`. They are compatibility tools.

## 4. Pressure behaviour, precisely

- **Contraction**: a rung that no longer fits (VRAM deficit, host-RAM deficit, or a contended
  card) starts a `deficit_since` timer; after `contraction_dwell_s` the policy re-fits against
  *usable* memory and picks the deepest rung that fits, preferring the same model family.
- **Contended card**: on top of re-fitting, the current rung's cards get an 8 GiB-per-card
  reserve (`CONTENTION_RESERVE_MB`) so the co-resident app and the model stop fighting.
- **Yield (new)**: if the card is contended and **no cheaper rung fits**, policy returns
  `release`: residency is unloaded, VRAM returns to the app, and nothing reloads until pressure
  has been clear for `expansion_dwell_s` (a single quiet sample does not count). If something
  external revives the rung, the controller re-releases it. This is
  `external_gpu_priority: absolute` finally being enforced instead of declared.
- **Expansion**: one rung at a time (or directly, per `expansion_mode`), gated by
  `expansion_dwell_s`, a per-card `expansion_margin_gb_per_card`, and the **ceiling** — pressure
  can only lower the effective rung, never raise the recommendation.
- **Flap control**: `cooldown_s` after every transition, `failure_times` inside `flap_window_s`,
  then `quarantine_until` to stop a bad rung from thrashing.
- Failure of an activation always rolls back to `last_stable_index`.

## 5. Code map (62 modules, 15.3k lines in `src/turbofit_runtime/`)

| Subsystem | Modules | Role |
|---|---|---|
| Authority | `runtime_profile`, `profile_io`, `selection`, `manual_profiles`, `migration`, `controller`, `policy`, `reconciler`, `runtime_service` | ladders, adaptive state, transitions |
| Pressure | `pressure`, `pressure_probe`, `gpu` | ownership-aware measurement |
| Engine | `native_backend`, `native_lifecycle`, `managed_engine`, `hybrid_runtime`, `executor`, `runtime_backends`, `backend`, `engine_check` | process lifecycle, launch, acceptance |
| Fit & catalog | `recipe`(s), `model_catalog`, `fit_check`, `fit_ladders`, `allowed_lineup`, `recommend`, `predict*`, `tier_report`, `leaderboard` | what fits, what is allowed, what is fastest |
| Evidence | `benchmark_stage`, `benchmark_schema`, `campaign`, `catalog_campaign`, `intelligence`, `intelligence_campaign`, `agentic_pair_benchmark`, `evidence`, `list_promotion`, `turbofit_list`, `tier_tournament`, `model_intel`, `salt_qualification` | measurement, promotion gates |
| Product surface | `runtime_cli`, `gateway_runtime`, `routes`, `tailnet`, `hermes_fit`, `wiki`, `multimodal`, `discovery_queue`, `downloads`, `freetoken`, `apple_mlx`, `request_normalisation`, `offload` | CLI, routing, bridges, installs |

Outside `src/`: `plugin_tools.py` (1.4k lines — Hermes-facing configure/status/install),
`product_ops.py` (`/turbofit` actions), `dashboard/` (FastAPI router + HTML), `desktop/plugin.js`,
`scripts/` (72 entry points), `references/` (data), `tests/` (112 files).

## 6. Model authority (the two ladders)

`references/fit-lists.json` is generated from `fit_check.py` and published by `fit_ladders.py`;
`references/model-catalog.json` holds 62 configurations across the families.

**Ladder 1 — dedicated VRAM (dense is fine)**

| Band | Model | Context |
|---|---|---|
| 8–11 GB | Ternary Bonsai 2 27B (PTQ1_0 / PQ2_0) | 131k |
| 12–15 GB | Underdog 27B (PQ2_0) | 131k |
| 16–23 GB | Qwen 3.8 27B GSQ-RCO IQ3_S | 131k |
| 24–63 GB | Qwen 3.8 27B GSQ-RCO IQ3_S | 262k |
| 64 GB+ | Qwen 3.8 Flash Next (Strata) | 262k |

**Ladder 2 — unified / CPU (MoE only, never dense)**

| Band | Model | Context |
|---|---|---|
| 8–15 GB | Maple Preview 20B-A1B (TQ2_0) | 131k |
| 16–23 GB | Cyber-Tiel Coder 35B-A3B (UD-IQ3_XXS / Q4 / Q5 / Q6 by band) | 262k |
| 64 GB+ | Qwen 3.8 Flash Next (Strata) | 262k |

Cross-cutting rules: MTP on wherever the model supports it; **IQ3_S is lossless**, so 32 GB and
48 GB both stay on GSQ-RCO IQ3_S rather than paying for a heavier quant; DFlash2 drafters belong
to Qwen only, Bonsai keeps its own Prism DSpark sidecar; dense models are never offered on
unified memory.

**Qwen 3.8 27B "Unleashed" is retired** — GSQ-RCO replaced it and it appears in no catalog, fit
list, profile, or doc. It survives only in `promo/`, `assets/*.svg`, and
`references/results/*.json`, which are historical artwork and measurement records: rewriting
those would falsify evidence of what actually ran on 2026-08-22/10-19.

## 7. Evidence and promotion

- A **candidate** is a catalog row. It becomes a **rung** only with a Turbofile entry and a
  saturated rung requirement (per-card MiB + host-RAM spill), bound by an evidence digest.
- `scripts/turbofit-catalog-campaign` proves native fit and TPS; it does not produce intelligence
  scores. `scripts/turbofit-intelligence-campaign` runs pinned DeepSWE plus the agentic
  main/aux pair and writes resumable state.
- `scripts/turbofit-promote-list-winner` promotes only an exact-tier candidate with current
  physical evidence, positive intelligence/TPS/balanced values and matching recipe hashes.
- Portable-fit lanes (fit computed from physics, never measured on this box) are labelled
  `portable-fit; benchmark-required` and are never presented as measured.

## 8. Surfaces

- **Plugin tools**: `turbofit_status`, `turbofit_configure`.
- **`/turbofit`**: `scan|status|tiers|update|shift up|down|<model>|serve|smoke|intelligence|
  balanced|speed|setup|aux <model>`. `scan` (no args) auto-applies the top measured lane, and
  falls back to the best portable-fit lane when nothing is measured yet.
- **CLI**: `scripts/turbofit-runtime list|set|status`, `scripts/turbofit-controller --once`,
  `scripts/release-check [--real]`, `scripts/turbofit-hardware-tiers`, `scripts/turbofit-list`.
- **Dashboard**: FastAPI router (`/status`, `/profiles`, `/combinations`, `/backends`,
  `/tournaments`, `/hardware-tiers`, `/leaderboard`, `/auxiliary-tiers`, `/recommendations`,
  `/multimodal`, `/configure`, `/update`) plus a static `index.html`.
- **Desktop**: `desktop/plugin.js` settings surface.
- **Skill**: `skills/turbofit/SKILL.md` + the plugin-root `SKILL.md` (same doctrine, agent-facing).

## 9. What is proven

- Test suite: **872 passed, 7 skipped, 17 failed** — the 17 are pre-existing and unrelated
  (artifact-download fixtures, three dashboard tests needing `fastapi`, catalog-matrix
  regeneration, one product-ops retire test).
- Live, this week, on omarchy: cold start expanded 65k → 262k → Strata one rung at a time;
  **yield** then unloaded the 43 GB engine and returned the card (22 GB/card → 1.7 GB) with
  dwell-gated restore instead of the 2 s flap the first cut produced; a systemd unit served as a
  rung; missing artifacts pruned instead of wedging; `shared-main` resolved to one process.
- Live, this week, on this box: TurboSouth installs from the public URL (`turbosouth v2.3.1`),
  and the missing-distribution error is now one actionable sentence.

## 10. Debt, drift, and open edges

1. **Doc/data drift on the unified top rung.** `SKILL.md` and the 2.4 changelog name DeepSeek
   V4.1 (1017) above Flash Next; `fit-lists.json` ends the unified ladder at Flash Next 64 GB+.
   Either the rung was never published to the data or the doc is aspirational.
2. **Cold start always picks the cheapest rung.** `ProfileCatalog.select()` hardcodes
   `initial_rung_index = len(rungs) - 1` and ignores the persisted index, so every restart begins
   at the smallest rung and heals upward through dwells. Correct by design for safety, poor for
   time-to-quality.
3. **Per-profile config can still wedge the controller.** One unresolvable resolution used to
   abort every tick *before* pressure was measured (that is the 3d22h wedge). Startup pruning
   now drops bad rungs, but the resolver itself is still per-rung all-or-nothing.
4. **Artifact drift between catalog, recipes and downloads.** The shipped GSQ-RCO rungs want
   `mmproj-Qwen3.8-27B-BF16.gguf`, which the download manifests do not stage the same way; the
   rung is now saved by the text-only fallback rather than by the data being right.
5. **`distribution.yaml`'s `dependencies:` / `post_install:` are inert.** Hermes' manifest parser
   has no such fields, so a profile-only install of TurboSouth arrives without the plugin.
   Handled in docs + `AGENTS.md`, not enforced by Hermes.
6. **Replaced engine layer.** `native_backend` (800 lines), `native_lifecycle`, `managed_engine`,
   `hardware`, `hybrid_runtime` and the launch half of `recipes` duplicate
   `hermes_cli/local_runtime/`. See `docs/hermes-local-runtime-adapter.md`.
7. **17 failing tests** — see §9; they predate this work but they mean `release-check` is not
   green, so "release readiness" claims must stay qualified.
8. **The VPS deployment is not a release artifact.** It is the plugin directory with a
   fast-forwarded clone; only `~/.local/state` and `~/.config` are durable.
9. **Federation is stale-while-revalidate, not live truth.** The gateway caches the remote
   catalog per peer configuration (`TURBOFIT_REMOTE_CACHE_TTL`, 30 s default) and serves the last
   known-good set rather than dropping peers on a hiccup. The cost is the opposite failure: a
   peer that just went away can still be listed for up to the TTL and fail on first use.

## 11. Relationship to Hermes' built-in runtime

Hermes ships `hermes_cli/local_runtime/` (18 modules, ~4.8k lines): hardware budget probe,
PM-pinned engines, GGUF headers, estimator + context policy, preset planning, a router-mode
`LlamaServerSupervisor`, idle sweep, external-server detection, endpoint resolution, bootstrap,
recovery. `hermes_fit.py` **already** consumes it (`probe_budget`,
`catalog.predicted_decode_tok_s`, `catalog.recommended_entry`) and degrades to TurboFit's own
ladders when it is absent.

What Hermes lacks is exactly TurboFit's product: external-app pressure, the ladder with
dwell/hysteresis/ceiling, evidence gates, routed IDs and federation, tiers/benchmarks, and the
model authority. Migration plan (P0 bridge → P1 delegate supervision → P2 presets from the
ladder → P3 delete the replaced modules) is in `docs/hermes-local-runtime-adapter.md`.

## 12. Invariants

- Never signal a process TurboFit did not launch (PID-and-command verified) — or a unit the
  Turbofile explicitly declares.
- External GPU processes are read-only signals: measured, never touched.
- The hardware recommendation is the healing ceiling; pressure only lowers the effective rung.
- Engines are PM-pinned; TurboFit never downloads a binary at boot, and never claims a TPS it
  did not measure on the box in question.
- `--jinja` on every llama.cpp launch; MTP/DFlash companions stay with their own families.
- Secrets never enter a Turbofile; state never leaves the profile's Hermes home.
