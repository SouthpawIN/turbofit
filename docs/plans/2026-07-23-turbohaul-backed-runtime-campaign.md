# Turbohaul-Backed Turbofit Runtime Campaign Implementation Plan

> **For Hermes:** Execute this plan task-by-task using strict TDD and real GPU verification.

**Goal:** Turn Turbofit into an evidence-backed system that systematically benchmarks every requested Main:Aux/context row, derives one optimal Turbohaul-compatible runtime string, registers each passing runtime immediately, recommends the best fitting pair, and swaps it for the user.

**Architecture:** A typed matrix schema is the single source of truth. A resumable campaign runner owns GPU clearing, component launch, Turbohaul admission/placement, inference, telemetry, evidence publication, and runtime registration. Turbofit ranks only passing profiles; Turbohaul owns fit, residency, queueing, and KV lifecycle. Existing ad-hoc sweep scripts become adapters behind the campaign runner instead of separate authorities.

**Tech Stack:** Python 3.11 stdlib, pytest, YAML/JSON manifests, llama.cpp/Prism/TurboQuant launchers, Turbohaul Manager v0.7.0, Docker, NVIDIA SMI, GitHub Actions.

---

### T01: Establish a clean source branch and ignore generated artifacts

**Files:**
- Modify: `.gitignore`
- Create: `docs/plans/2026-07-23-turbohaul-backed-runtime-campaign.md`

**Deliverable:** Source, tests, compact benchmark summaries, and manifests remain versioned; model binaries, compiled runtimes, caches, logs, and raw transient outputs do not enter Git.

**Verification:** `git status --short` shows no compiled libraries or `__pycache__` candidates.

### T02: Define the canonical 75-row matrix and runtime schema

**depends:** T01

**Files:**
- Create: `src/turbofit_runtime/schema.py`
- Create: `references/main-aux-matrix.json`
- Create: `tests/test_matrix_schema.py`

**Deliverable:** Exactly 75 unique rows; Bonsai-at-1M rows absent; contexts normalized to integers; every row has deterministic ID, main, aux, context, status, and compatibility method priority.

**Verification:** RED then GREEN: `pytest -q tests/test_matrix_schema.py`.

### T03: Add GPU-clear and per-card admission primitives

**depends:** T02

**Files:**
- Create: `src/turbofit_runtime/gpu.py`
- Create: `tests/test_gpu_clear.py`
- Refactor: `scripts/matrix_utils.py`

**Deliverable:** Every configuration starts only after three consecutive clear samples and ends only after managed processes are gone and both cards return below measured baseline ceilings. Unrelated desktop GPU processes are reported but never killed.

**Verification:** RED then GREEN unit tests plus a real `nvidia-smi` clear-gate run.

### T04: Add canonical benchmark result and evidence publication

**depends:** T02, T03

**Files:**
- Create: `src/turbofit_runtime/evidence.py`
- Create: `tests/test_evidence.py`
- Modify: `~/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md` through the publisher only

**Deliverable:** Only a complete pass may check a row. Every pass writes bidirectional evidence, exact metrics, GPU-clear event, runtime string, and source hashes. Failed/unsupported rows receive evidence without success promotion.

**Verification:** RED then GREEN tests using temporary wiki fixtures; audit all internal links.

### T05: Build Turbohaul v0.7 manifest and runtime-string compiler

**depends:** T02, T04

**Files:**
- Create: `src/turbofit_runtime/turbohaul.py`
- Create: `tests/test_turbohaul_manifest.py`
- Create: `references/turbohaul/`

**Deliverable:** Convert each passing component into content-addressed Turbohaul manifest data with exact context, per-card measured footprint, measured KV bytes/token when available, `auto_place` eligibility, projector identity, and only allowlisted flags. Compile a deterministic pair activation string.

**Verification:** Validate generated manifests against Turbohaul v0.7.0's real `Manifest` model from the pinned source checkout; reject DSpark/DFlash/NexTN combinations that require an unsupported engine flag rather than silently translating them.

### T06: Build one resumable campaign runner

**depends:** T03, T04, T05

**Files:**
- Create: `src/turbofit_runtime/campaign.py`
- Create: `scripts/turbofit-campaign`
- Create: `tests/test_campaign_resume.py`
- Create: `tests/test_campaign_state_machine.py`

**Deliverable:** `turbofit-campaign run` walks pending rows bottom-up, clears GPUs between rows, selects DSpark then MTP then NexTN/family-compatible alternatives, executes controlled inference, records results, publishes passes, and resumes after interruption without repeating passed rows.

**Verification:** RED then GREEN state-machine tests; dry-run prints all remaining rows in deterministic order; one real row runs end to end.

### T07: Replace catalog-only recommendation with pair recommendation

**depends:** T04, T05

**Files:**
- Refactor: `scripts/turbofit-runtime-recommend`
- Modify: `scripts/turbofit-runtime`
- Modify: `scripts/serve`
- Create: `tests/test_recommendation.py`

**Deliverable:** Turbofit recommends only evidence-backed pair profiles. Fit uses Turbohaul-style per-card admission and measured context cost. Ranking supports balanced, speed, quality, context, memory efficiency, vision, and live-state modes.

**Verification:** RED then GREEN ranking tests; CLI result carries exact evidence and activation string.

### T08: Import and normalize existing successful runs

**depends:** T04, T05

**Files:**
- Modify: `references/successful-runtime-profiles.json`
- Create: `references/campaign-state.json`
- Normalize: `references/results/*.json`

**Deliverable:** Existing checked rows are migrated without losing evidence. Every imported success passes the same schema and link audit as new runs.

**Verification:** Success count equals runtime-profile count; every evidence file exists; every profile validates.

### T09: Execute all remaining matrix rows

**depends:** T06, T08

**Deliverable:** Exercise all pending rows from smallest/cheapest families upward. After each row, clear both GPUs, publish evidence, and register passing runtime immediately. Diagnose and retry launch failures with one materially different recipe; mark truly unsupported combinations with explicit evidence.

**Verification:** Matrix has no unclassified `[ ]` rows; each row is either `[x]` with evidence/profile or `[-]` with reproducible incompatibility evidence.

### T10: Full audit, documentation, and GitHub publication

**depends:** T09

**Files:**
- Update: `README.md`
- Update: `skills/turbofit/SKILL.md`
- Update: Turbofit wiki pages
- Add/update: GitHub Actions tests

**Deliverable:** Document the one-command flow: recommend → hook up → swap → optimize. Commit source and compact evidence, push the feature branch, open a GitHub PR, and verify CI.

**Verification:** Full tests pass; shell syntax passes; all links resolve; GitHub PR and CI URLs are recorded.
