# Adapting TurboFit onto Hermes' built-in local runtime

Status: analysis + plan. Nothing in this document deletes code yet.

Hermes ships `hermes_cli/local_runtime/` — 18 modules, ~4,845 lines — that already owns
most of the *engine* work TurboFit has been duplicating. What it does **not** have is the
part TurboFit exists for: pressure-aware residency policy, evidence-gated ladders, and
the model authority.

The goal is not a rewrite. It is: **Hermes owns the engine, TurboFit owns the decisions and
the model list.**

## 1. What Hermes already ships

| Concern | Hermes seam | Notes |
|---|---|---|
| Hardware budget | `hardware.probe_budget(planning: bool)` → `HardwareBudget` | `planning=False` = live free VRAM; `planning=True` = capacity minus margin. Already covers discrete vs unified, Windows commit limits, lazy reads. |
| GPU/device inventory | `hardware.py` (`nvidia-smi` memory.total/free/used/util, PCI ids), `devices.py`, `detect.py` | Same data TurboFit probes, plus vendor/PMA selection. |
| Engine binaries | `binaries.select_backend`, `installed_engine`, `ensure_engine`, `pinned_tag`, `Engine` | PM-pinned store; **never downloads at boot**, verifies archives, publishes atomically. Replaces `scripts/install-native-runtimes` and TurboFit's binary resolution. |
| GGUF inspection | `gguf.read_gguf_header` | Replaces TurboFit's header parsing. |
| Fit math / quant choice | `estimator.HardwareBudget`, `ctx_bytes`, `physics_check`, `profile_from_gguf`; `catalog.select_variant`, `predicted_decode_tok_s`, `recommended_entry` | Physics-predicted, per-device, with spill accounting. |
| Launch flags | `context_policy.launch_args(profile, decision, flash_attention, mtp_capable, mtp_draft_depth, uma, mtp_prefill)`; `context_policy.ladder/initial_window/FLOOR` | Emits `-c`, q8 KV where FA exists, `-ot` placement on spilled configs, MTP draft flags. TurboFit currently generates this by hand in `recipes.py`. |
| Preset planning | `presets.plan_presets` / `generate_presets(models_dir, budget, preset_path, mtp_capable, live=, residents=, engine_tag=)` / `write_presets` | Plans **every staged GGUF**, refuses what does not fit, writes one INI. DFlash + MTP companion planning included. |
| Process supervision | `supervisor.LlamaServerSupervisor(binary, models_dir, models_max, port, preset_path)` | Router mode: `start/stop`, `models()`, `load_model`, `unload_model`, `ensure_model_ready`, `touch_generate`, `sweep_idle`, `is_idle`. Verified-root tree termination, watchdog, orphan reaping, stable port + stable API key, state file. |
| Idle unload | `supervisor.sweep_idle` + `_idle_since` | Replaces `native_lifecycle.IdleLifecycle`. |
| External server adoption | `detect.detect_server(extra_ports)` (read-only probe) | The "discovered backend" case TurboFit handles ad hoc. |
| Endpoint resolution | `endpoint.resolve_llamacpp_endpoint(config)` managed-first, detect-second | Single endpoint. TurboFit's routed multi-ID surface is a superset. |
| Boot/shutdown | `bootstrap.ensure_local_runtime(config)`, `shutdown_local_runtime`, boot lock, `refit_idle_presets` | Replaces controller boot logic for the process side. |
| Recovery | `recovery.py`, `load_progress.py`, `growth.py` (window overrides) | Restart/backoff + context growth memory. |

## 2. What Hermes does **not** have (TurboFit's actual product)

Verified by reading the package, not assumed:

- **No external-app pressure model.** Nothing attributes VRAM to processes that are not ours,
  nothing distinguishes desktop baseline from a game, and there is no "someone else is using
  the card" state. `sweep_idle` unloads on *idleness*, never on *contention*.
- **No contraction/expansion policy.** No dwell, hysteresis, cooldown, flap quarantine, or
  recommendation ceiling. There is no notion of a rung ladder at all: presets are a flat set of
  admitted models.
- **No evidence gates.** No measured-on-box scores, no Fit List, no promotion rules, no
  campaign harness. `catalog.json` is a packaged default of **5 models** (qwen3.8-27b,
  qwen3.8-flash-next, nemotron-3.5-bolt-74b-a9b, qwen3.6-35b-a3b, deepseek-v4-flash) with one
  quant each.
- **No routed role surface.** No `auto` / `active:main` / `active:aux`, no route-state file,
  no tailnet federation of peer gateways.
- **No tiers dashboard, benchmarks, or TurboSouth.**
- **No unit-managed rungs.** Nothing can treat a systemd unit as a rung (the Strata case).

## 3. Replacement map

Line counts are TurboFit `src/turbofit_runtime/` (15,258 total).

| TurboFit module | Lines | Verdict | Target |
|---|---|---|---|
| `native_backend.py` | 800 | **replace** | `supervisor.LlamaServerSupervisor`. Keep only the unit-rung launcher and the `release_current` wrapper over `unload_model`/`sweep_idle`. |
| `native_lifecycle.py` | 428 | **replace** | `sweep_idle` + `ensure_model_ready` + `growth.py`. |
| `managed_engine.py` | 402 | **replace** | `binaries.ensure_engine` / `installed_engine`. |
| `hardware.py` | 478 | **replace** | `probe_budget(planning=…)`; keep a thin adapter so Turbofiles keep their declared constraints. |
| `hybrid_runtime.py` | 329 | **replace** | `presets` + `launch_args`. |
| `recipes.py` (launch half) | ~400 of 697 | **replace** | `plan_presets` / `launch_args` / `read_gguf_header`. Keep the catalog/alias layer as the model authority. |
| `routes.py` (endpoint half) | ~150 of ? | **adapt** | Resolve through `resolve_llamacpp_endpoint`, then publish TurboFit's stable IDs on top. |
| `engine_check.py` | 446 | **keep** | Evidence/acceptance, not engine plumbing. |
| `pressure.py`, `pressure_probe.py` | ~380 | **keep** | The differentiator. Add a `HardwareBudget` bridge so the probe and Hermes agree on device indices. |
| `policy.py`, `controller.py` | 987 | **keep** | Dwell/hysteresis/ceiling/flap/yield. This is the product. |
| `runtime_profile.py`, `selection.py`, `rung-requirements` | ~700 | **keep** | Portable ladders + evidence binding. |
| `hermes_fit.py` | 565 | **adapt** | Stop shipping its own catalog view; consume the backend's chosen model + TurboFit's ladder. |
| `benchmark_*`, `campaign*`, `intelligence_campaign` | ~1,900 | **keep** | Evidence engine. |
| `salt_qualification.py` | 600 | **keep** | Unrelated qualification gate. |

Net: roughly **2,400–2,800 lines** of engine/supervision code collapses into calls, and
about **12,500 lines** of policy, evidence, distribution, and product stay.

## 4. Model authority: TurboFit's ladder wins

Hermes' model choice comes from the packaged `catalog.json` + `select_variant` +
`recommended_entry`. That is the list you consider weaker, and it is not something we can
edit in the platform-managed image.

The seam that makes your models authoritative without touching Hermes core:

1. **`models_dir` is the input.** `plan_presets` plans over whatever GGUFs are staged there,
   refuses what will not fit, and writes the INI the supervisor loads. So the staging set *is*
   the model authority.
2. **TurboFit owns staging.** `references/model-recipes.json` (variants, quants, projectors,
   drafts) decides what lands in `models_dir` — Bonsai 2 27B PTQ1_0/PQ2_0, Underdog 27B PQ2_0,
   Qwen 3.8 27B GSQ-RCO IQ3_S/IQ2_XS, Flash Next, Maple Preview, Cyber-Tiel 35B-A3B,
   DeepSeek V4.1 — instead of the five packaged entries.
3. **TurboFit owns ordering and refusals.** Hermes' preset planner is fit-only; it has no
   ladder order, no tier doctrine ("IQ3_S is lossless; do not push heavier past it") and no
   evidence gate. Those stay in the Turbofile and are applied as: which GGUFs are staged,
   `residents=1`, and an explicit ordered selection over `models()`.
4. **Fallback stays cheap.** If a TurboFit model is absent, `plan_presets`/`detect` degrade to
   whatever Hermes already knows — never to a hosted API.

## 5. Phased plan

**P0 — bridge, ship nothing breaking.** Add `src/turbofit_runtime/hermes_engine.py`: thin
wrappers over `probe_budget`, `plan_presets`, `generate_presets`, `LlamaServerSupervisor`,
`detect_server`, `resolve_llamacpp_endpoint`, `binaries.installed_engine`. No behaviour change.
Tests assert the wrappers agree with TurboFit's existing fingerprint on this box.

**P1 — delegate supervision.** `NativeRuntimeBackend` calls `LlamaServerSupervisor` for
llama.cpp rungs; TurboFit keeps `release_current` (now `unload_model`) and the unit-rung path.
Delete PID bookkeeping, watchdog, and orphan reaping. The pressure probe's `managed_pids` starts
coming from the supervisor's state file instead of TurboFit's own registry.

**P2 — presets from the ladder.** Generate Hermes presets from TurboFit's staged GGUFs at
controller boot (`generate_presets(models_dir, probe_budget(planning=True), preset_path, ...)`),
and have `selection` choose among `supervisor.models()` in Turbofile order. This is where your
model list actually replaces the built-in five.

**P3 — delete the replaced modules** and fold the surviving seams into `hermes_engine.py`.
Release-check must still pass with the same evidence artifacts.

**P4 — evidence and product** continue unchanged: campaigns, Fit List, tiers, dashboard,
TurboSouth, tailnet federation, unit rungs.

## 6. Invariants that must survive the migration

- External GPU processes stay **read-only** pressure signals; never signalled.
- Lifecycle control stays PID-and-command verified — now for children the *supervisor* owns,
  plus declared units for unit rungs.
- The hardware recommendation remains the healing ceiling; pressure only lowers the effective
  rung.
- Hermes' engine bytes stay PM-pinned; TurboFit never downloads an engine.
- `--jinja` and the MTP/DFlash companion rules are unchanged.

## 7. Open questions

1. Should TurboFit keep its own gateway (`:8091` with `auto`/`active:main`/`active:aux`) or
   publish the routed IDs on top of Hermes' single local endpoint? Keeping it is what preserves
   Tailscale federation for the MacBook and phone.
2. Do we want `models_dir` to be Hermes' directory (so the built-in picker sees your models) or
   a TurboFit-owned directory passed in explicitly (cleaner separation, invisible to Hermes'
   picker)?
3. Hermes' catalog refresh is remote and packaged; confirm we treat it as a **fallback default**
   only, never as an override of the Turbofile ladder.
