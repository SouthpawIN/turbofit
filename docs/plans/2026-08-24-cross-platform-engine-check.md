# Cross-Platform Inference Engine Check Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make TurboFit a self-configuring and self-optimizing OpenAI-compatible server that checks MLX, llama.cpp, SGLang, vLLM, FreeToken, and MrTrenchTrucker's Turbohaul Manager on each user machine, selects only physically verified recipes, contracts safely, and heals upward.

**Architecture:** Add one engine capability registry and machine-local TurboFit Check probe shared by Dashboard, Desktop, CLI, setup, and the controller. Engines expose a common lifecycle/health/benchmark contract behind stable `auto`, `active:main`, and `active:aux` routes. Platform support is honest: TurboFit checks every engine, but marks unsupported native combinations instead of pretending every engine runs on every OS; WSL is treated as Linux. Selection uses local smoke/throughput/memory evidence, while fallback/healing remains controller-owned and engine-agnostic.

**Tech Stack:** Python 3.11 stdlib, existing Turbofit hardware/profile/controller modules, OpenAI-compatible HTTP, pytest, platform-native services (systemd, launchd, Windows Task Scheduler/service).

---

### Task 1: Canonical engine capability registry

**Objective:** Define one source of truth for OS, architecture, accelerator, executable/module, endpoint, source repository, and install constraints for all six runtime options.

**Files:**
- Create: `src/turbofit_runtime/engine_check.py`
- Create: `tests/test_engine_check.py`

**Steps:**
1. Write failing tests for all six runtime IDs, Windows/Linux/macOS compatibility, WSL-as-Linux, Apple MLX eligibility, FreeToken's Linux x86_64 NVIDIA/CUDA requirement, and Turbohaul Manager's pinned MrTrenchTrucker GitHub identity.
2. Run `PYTHONPATH=src:. pytest tests/test_engine_check.py -q -o 'addopts='` and verify failure because the module does not exist.
3. Implement immutable engine specs and deterministic compatibility reports.
4. Re-run focused tests and verify pass.

### Task 2: Installed/running engine probes

**Objective:** Detect executable or Python-module installation and bounded OpenAI health without mutating the machine.

**Files:**
- Modify: `src/turbofit_runtime/engine_check.py`
- Modify: `tests/test_engine_check.py`

**Steps:**
1. Add failing tests for executable discovery, module discovery, `/health`, and `/v1/models` fallback.
2. Implement dependency-injected probes with bounded timeouts and redacted errors.
3. Verify unsupported engines remain reported, not silently omitted.

### Task 3: Machine-local benchmark contract

**Objective:** TurboFit Check must run a bounded real completion for each compatible installed engine and record TTFT, decode TPS, peak memory, context, model recipe hash, and failure class.

**Files:**
- Create: `src/turbofit_runtime/engine_benchmark.py`
- Create: `tests/test_engine_benchmark.py`
- Create: `references/engine-check-schema.json`

**Steps:**
1. Test schema validation and refusal of zero-call/zero-token evidence.
2. Implement OpenAI-compatible smoke/benchmark adapter and immutable result records.
3. Bind results to hardware fingerprint, engine/version, model artifact, and launch recipe.

### Task 4: TurboFit Check CLI and product API

**Objective:** Expose the same engine report through CLI, Dashboard, Desktop, and plugin tools.

**Files:**
- Create: `scripts/turbofit-check`
- Modify: `dashboard/plugin_api.py`
- Modify: `plugin_tools.py`
- Modify: `desktop/plugin.js`
- Add tests under `tests/test_plugin.py`, `tests/test_dashboard_plugin.py`, and `tests/test_desktop_plugin.py`.

**Steps:**
1. Add failing parity tests.
2. Implement `scan`, `probe-engines`, `benchmark-compatible`, and `apply-best` operations.
3. Return explicit `supported`, `installed`, `running`, `benchmarked`, `eligible`, and `reason` fields for every engine.

### Task 5: Unified engine lifecycle adapters

**Objective:** Start, stop, verify, and route every engine through one controller interface.

**Files:**
- Create: `src/turbofit_runtime/engine_runtime.py`
- Modify: `src/turbofit_runtime/reconciler.py`
- Modify: `src/turbofit_runtime/routes.py`
- Modify: `scripts/turbofit-gateway.py`
- Add engine-specific tests and integration fixtures.

**Steps:**
1. Define adapter protocol: `prepare`, `start`, `health`, `models`, `complete`, `stats`, `stop_owned`.
2. Implement llama.cpp, MLX, SGLang, vLLM, FreeToken, and Turbohaul Manager adapters.
3. Preserve strict PID/process ownership and never signal external processes.
4. Publish stable routes only after completion verification.

### Task 6: Engine-aware fallback and heal-up policy

**Objective:** Let the controller move across engines as well as context/model rungs without changing stable provider IDs.

**Files:**
- Modify: `src/turbofit_runtime/policy.py`
- Modify: `src/turbofit_runtime/controller.py`
- Modify: `src/turbofit_runtime/runtime_profile.py`
- Modify: `src/turbofit_runtime/reconciler.py`
- Add tests in `tests/test_policy.py`, `tests/test_controller.py`, and integration tests.

**Steps:**
1. Add engine identity and locally measured score to rung candidates.
2. Contract: dedicated aux → shared main → lower context → lower quant/model → alternate verified engine → API.
3. Heal upward only after margin, dwell, cooldown, and verification.
4. Quarantine repeatedly failing engine/recipe pairs without disabling the engine globally.

### Task 7: Cross-platform installation and services

**Objective:** Install/check compatible engines and keep TurboFit online on Windows, Linux, and macOS.

**Files:**
- Extend native installers under `scripts/`.
- Add launchd templates for macOS, systemd templates for Linux, and Windows service/task templates.
- Add installer tests for all three operating systems.

**Steps:**
1. llama.cpp: native Windows/Linux/macOS.
2. MLX: Apple Silicon/macOS only.
3. SGLang: Linux native; Windows through WSL; unsupported native macOS combinations reported honestly.
4. vLLM: Linux native and supported vLLM-Metal path on Apple Silicon; Windows through WSL where applicable.
5. FreeToken: Linux x86_64 + NVIDIA CUDA 13/r580+ only until upstream expands support; Windows WSL may qualify as Linux.
6. Turbohaul Manager: install pinned `MrTrenchTrucker/turbohaul-manager` release on Linux NVIDIA hosts and qualifying Windows WSL; preserve `/var/lib/turbohaul` state and verify `/status` plus `/api/tags`.
7. Never install an incompatible runtime merely to satisfy a checklist.

### Task 8: Release gates and documentation

**Objective:** Block release unless all engine/platform contracts and fallback/heal simulations pass.

**Files:**
- Modify: `scripts/turbofit-completion-check`
- Modify: `scripts/release-check`
- Modify: `docs/turbofit-check.md`
- Modify: `README.md`

**Steps:**
1. Require all six runtime options in the capability registry and product API.
2. Require Windows/Linux/macOS deterministic compatibility tests.
3. Require local real checks only for engines compatible with the current host.
4. Require fallback and heal-up simulation across engine failures.
5. Report unsupported, unavailable, failed, and passed separately—never collapse them into one false red/green state.
