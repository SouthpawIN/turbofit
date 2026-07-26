# Adaptive Turbohaul Runtime Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task with strict RED → GREEN → REFACTOR and two-stage review.

**Goal:** Make Turbofit recommend the best evidence-backed runtime for the user's hardware, yield its own VRAM politely to external GPU load, and self-heal upward through portable Turbohaul-backed runtime rungs.

**Architecture:** Turbohaul v0.7 remains the only local serving/residency substrate. Turbofit adds a pure policy layer above it: portable Turbofile profiles, hardware/evidence ranking, external-load classification, and a hysteretic reconciliation state machine. Hermes keeps one `custom:turbofit` provider with `auto`, `active:main`, and `active:aux` role IDs.

**Tech Stack:** Python 3.11 stdlib, optional PyYAML loader, pytest, Turbohaul Manager v0.7 OpenAI/Ollama APIs, Hermes custom provider/plugin hooks/cron, JSON benchmark evidence, Markdown wiki.

**Canonical design:** `~/.hermes/wiki/topics/turbofit/turbohaul-v0.7-integration.md`

**Git safety:** Source checkpoints are required, but the current environment approval gate blocked staging. Never bypass it. Continue producing tested files; checkpoint only when the gate permits the explicitly authorized operation.

---

### Task 1: Define the portable Turbofile schema

**Objective:** Represent hardware constraints, role routing, policy timing, and an ordered adaptive rung ladder without machine-specific paths.

**Files:**
- Create: `src/turbofit_runtime/runtime_profile.py`
- Create: `tests/test_runtime_profile.py`
- Create: `runtime-profiles/schema-example.yaml`

**Steps:**
1. Write failing tests for valid parsing, ordered unique rungs, required API terminal rung, valid aux modes, positive contexts/margins, and rejection of absolute local paths or credentials.
2. Run `PYTHONPATH=src pytest tests/test_runtime_profile.py -q` and verify RED.
3. Implement immutable dataclasses and `Turbofile.from_mapping()` / `to_mapping()` using stdlib types.
4. Run the focused test and verify GREEN.
5. Run `PYTHONPATH=src pytest -q -o 'addopts='`.

**Acceptance:** A runtime profile is portable, deterministic, and has no host-specific identity.

### Task 2: Add YAML/JSON loading and canonical serialization

**Objective:** Load a shareable `Turbofile.yaml`, emit deterministic normalized data, and reject unknown fields.

**Files:**
- Modify: `src/turbofit_runtime/runtime_profile.py`
- Create: `src/turbofit_runtime/profile_io.py`
- Create: `tests/test_profile_io.py`

**Steps:**
1. RED tests for YAML and JSON loading, exact round-trip, duplicate keys, unknown keys, schema-version mismatch, and absent optional YAML dependency.
2. Implement a narrow loader with clear dependency failure; JSON remains dependency-free.
3. Add canonical JSON hashing for profile/evidence identity.
4. Verify focused and full suites.

### Task 3: Build a stable hardware fingerprint

**Objective:** Distinguish total VRAM from topology and produce a portable recommendation key.

**Files:**
- Create: `src/turbofit_runtime/hardware.py`
- Create: `tests/test_hardware.py`

**Steps:**
1. RED tests for 1×24 versus 2×24, per-card ordering, compute capability, RAM, OS, backend capability, missing probes, and NVIDIA CSV parsing.
2. Implement pure probe parsers and an injectable live probe shell.
3. Verify no GPU environment degrades honestly rather than guessing.
4. Verify focused and full suites.

### Task 4: Separate hardware recommendation from live fit

**Objective:** Ensure resident Turbofit memory never makes `auto` conclude that its recommended profile no longer fits.

**Files:**
- Create: `src/turbofit_runtime/recommend.py`
- Create: `tests/test_recommend.py`
- Modify: `scripts/turbofit-runtime-recommend`

**Steps:**
1. RED tests: physical-capacity recommendation, topology constraints, evidence-only candidates, current live rung, and resident-memory invariance.
2. Implement lexicographic quality → 128K → 30 tok/s → 262K → 100 tok/s → 1M ranking.
3. Preserve named policy variants without an opaque weighted score.
4. Verify current CLI behavior and full suite.

### Task 5: Model ownership-aware VRAM pressure

**Objective:** Calculate external demand without treating Turbofit's own resident models as external load.

**Files:**
- Create: `src/turbofit_runtime/pressure.py`
- Create: `tests/test_pressure.py`

**Steps:**
1. RED tests for managed PIDs, Turbohaul sidecars, unrelated GPU processes, desktop baseline, safety reserve, in-flight reservations, and unavailable process data.
2. Implement immutable pressure snapshots and per-card available budgets.
3. Prove unrelated processes are never returned as action targets.
4. Verify focused and full suites.

### Task 6: Implement the pure adaptive rung state machine

**Objective:** Select contraction/expansion transitions with dwell, hysteresis, cooldown, rollback, and flap quarantine.

**Files:**
- Create: `src/turbofit_runtime/policy.py`
- Create: `tests/test_policy.py`

**Steps:**
1. RED tests for single-sample immunity, severe-deficit skipping, one-rung expansion, expansion margin, target ceiling, cooldown, failed activation rollback, and quarantine.
2. Implement a pure `reconcile(state, snapshot, profile, now)` returning an action plan without side effects.
3. Verify deterministic replay from recorded snapshots.
4. Verify focused and full suites.

### Task 7: Add a thin Turbohaul v0.7 client

**Objective:** Reuse Turbohaul status, manifests, residents, admission, and clean unload instead of direct process authority.

**Files:**
- Create: `src/turbofit_runtime/turbohaul_client.py`
- Create: `tests/test_turbohaul_client.py`
- Modify: `src/turbofit_runtime/turbohaul.py`

**Steps:**
1. RED contract tests using a local fake HTTP server for status, manifest ETags, inference, `keep_alive: 0`, queue state, and errors.
2. Implement stdlib HTTP client with bounded timeouts and typed errors.
3. Require post-action state verification.
4. Verify focused and full suites.

### Task 8: Reconcile roles politely

**Objective:** Drain and unload dedicated auxiliary, switch `active:aux` to shared-main, then move context/main/API rungs without changing Hermes provider configuration.

**Files:**
- Create: `src/turbofit_runtime/reconciler.py`
- Create: `tests/test_reconciler.py`

**Steps:**
1. RED tests for no new aux admission, active-stream drain, clean unload, KV-preserving seam, shared-main route, context/model transition, API terminal rung, rollback, and owned-process-only escalation.
2. Implement effect executor behind injected Turbohaul/gateway interfaces.
3. Verify each transition before state publication.
4. Verify focused and full suites.

### Task 9: Integrate the stable provider model IDs

**Objective:** Keep `auto`, `active:main`, and `active:aux` stable while the effective rung changes underneath.

**Files:**
- Modify: `scripts/turbofit-gateway.py`
- Modify: `tests/test_unified_provider.py`
- Create: `tests/test_gateway_runtime_policy.py`

**Steps:**
1. RED tests for dynamic main/aux/shared-main/API routing and warm requests.
2. Wire gateway route resolution to reconciler state, not fixed process ports.
3. Preserve loading stall and API fallback.
4. Verify fresh Hermes main and delegated aux calls.

### Task 10: Migrate current evidence to portable profiles

**Objective:** Remove machine-specific paths/indices from published identities while retaining local resolution and every evidence backlink.

**Files:**
- Create: `scripts/migrate-runtime-profiles`
- Create: `tests/test_profile_migration.py`
- Modify: `references/successful-runtime-profiles.json`
- Populate: `runtime-profiles/`

**Steps:**
1. RED fixture tests for current profile migration.
2. Convert component identity to content hashes and capability constraints.
3. Keep local paths only in ignored local resolution state.
4. Verify every migrated rung resolves to evidence and a Turbohaul manifest.

### Task 11: Define all seven hardware-class profiles

**Objective:** Publish honest 8/16/24/48/96/200/300 GB recommendation envelopes with topology subkeys.

**Files:**
- Create: `runtime-profiles/8gb.yaml`
- Create: `runtime-profiles/16gb.yaml`
- Create: `runtime-profiles/24gb.yaml`
- Create: `runtime-profiles/48gb.yaml`
- Create: `runtime-profiles/96gb.yaml`
- Create: `runtime-profiles/200gb.yaml`
- Create: `runtime-profiles/300gb.yaml`
- Create: `tests/test_hardware_profiles.py`

**Steps:**
1. RED tests for all classes, topology branches, terminal API rung, and evidence status.
2. Populate only measured model winners; represent unproven rungs as candidates, never recommendations.
3. Verify schema, links, and recommendation behavior.

### Task 12: Establish the benchmark suite and promotion gates

**Objective:** Turn compatibility, performance, quality, and adaptive behavior into reproducible evidence.

**Files:**
- Create: `benchmarks/suite.yaml`
- Create: `src/turbofit_runtime/benchmark_schema.py`
- Create: `tests/test_benchmark_schema.py`
- Modify: `scripts/matrix-benchmark.py`

**Steps:**
1. RED tests for artifact, runtime, performance, quality, and pressure/self-heal stages.
2. Require hashes, host fingerprint, observed context, throughput, TTFT, per-card VRAM, power, quality, and raw result identity.
3. Prevent candidate results from entering recommendation until every required gate passes.
4. Verify focused and full suites.

### Task 13: Add daily model and API intelligence pipelines

**Objective:** Discover Hugging Face candidates and global/API model news without auto-promoting them.

**Files:**
- Create: `research/discover_huggingface.py`
- Create: `research/discover_model_news.py`
- Create: `research/discover_api_models.py`
- Create: `research/candidates.json`
- Create: `tests/test_research_candidates.py`
- Create: `docs/cron/model-intelligence.md`

**Steps:**
1. RED tests for diff-only candidate updates, provenance, no credentials, and no production writes.
2. Implement deterministic collectors and candidate normalization.
3. Document self-contained Hermes cron prompts/toolsets; create live cron jobs only after delivery/schedule approval.
4. Verify fixtures without network, then smoke current sources read-only.

### Task 14: Publish generated wiki views

**Objective:** Generate recommendation tables, candidate queues, and evidence links from canonical data without creating another authority.

**Files:**
- Create: `src/turbofit_runtime/wiki.py`
- Create: `tests/test_wiki.py`
- Modify: `~/.hermes/wiki/topics/turbofit/README.md` through the publisher
- Modify: `~/.hermes/wiki/topics/turbofit/main-aux-inference-checklist.md` through the publisher

**Steps:**
1. RED fixture tests for deterministic output and bidirectional links.
2. Generate views from profiles/evidence/candidates.
3. Reject broken links or unchecked evidence.
4. Verify the real wiki after a dry-run diff.

### Task 15: Run real adaptive acceptance tests

**Objective:** Prove the system yields to external load and self-heals without harming unrelated work.

**Files:**
- Create: `tests/integration/test_adaptive_runtime.py`
- Create: `scripts/adaptive-acceptance`
- Create: `references/results/adaptive-runtime-acceptance.json`

**Steps:**
1. Simulate every transition with fake adapters.
2. Run controlled external GPU allocation on the current host.
3. Verify aux unload → shared-main → context/model contraction → API as required.
4. Release external load and verify one-rung self-healing to the recommendation.
5. Verify no unrelated PID receives a signal.
6. Record exact evidence and rollback behavior.

### Task 16: Consolidate docs and release gates

**Objective:** Remove obsolete launch/scaling authorities and document one install/recommend/run path.

**Files:**
- Modify: `README.md`
- Modify: `SKILL.md`
- Modify: `skills/turbofit/SKILL.md`
- Modify: Turbofit wiki pages
- Modify: CI configuration

**Steps:**
1. Remove stale Darwin/Carnice and dual-provider instructions.
2. Document Turbohaul-only local serving and the Turbofile contract.
3. Run syntax, unit, integration, link, and real provider/delegation checks.
4. Complete spec and code-quality review.
5. Create the approved Git checkpoint and push the feature branch when the environment gate permits.
