#!/usr/bin/env python3
"""
turbofit-scaling-watcher v2 — gradual contraction + multi-profile management.

The "conscious stream" — monitors per-GPU VRAM and automatically contracts
and expands local models without user intervention.

CONTRACTION LADDER (applied per-GPU, graduated, not hard-kill):
  0. HEALTHY: Everything running at full ctx, no expert offload
  1. SHRINK_CTX: Reduce main ctx 262K → 131K → 65K (Hermes hard floor)
  2. EXPERT_OFFLOAD: If MoE model, add --cpu-moe to shed VRAM
  3. SWAP_MODEL: Swap main to smaller model (Darwin Q4 → Prism Eagle 13.7GB)
  4. STOP_AUX: Stop aux daemons on the pressured GPU
  5. STOP_MAIN: Stop main, switch all local profiles to API fallback
  6. CRITICAL: Everything local dead, API-only mode

EXPANSION (reverse, with hysteresis):
  - When VRAM recovers, undo contractions step by step
  - Each recovery threshold is 4GB above the contraction threshold

MULTI-PROFILE:
  - Detects ALL Hermes profiles pointing at localhost
  - When switching to API, reconfigures ALL of them
  - When switching back to local, restores ALL of them

Preferences: ~/.config/turbofit/preferences.yaml
"""

import subprocess
import json
import time
import os
import sys
import logging
import argparse
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scale] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scale")

# ─── Paths ────────────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
PREFS = os.environ.get("TURBOFIT_PREFS", f"{HOME}/.config/turbofit/preferences.yaml")
HERMES_HOME = os.environ.get("HERMES_HOME", f"{HOME}/.hermes")

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MAIN_FALLBACK = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_PROVIDER = "nous"

# ─── Contraction thresholds (GB free per GPU) ─────────────────────────────────
SHRINK_CTX_THRESHOLD = 6.0
EXPERT_OFFLOAD_THRESHOLD = 4.0
SWAP_MODEL_THRESHOLD = 3.0
STOP_AUX_THRESHOLD = 2.0
STOP_MAIN_THRESHOLD = 1.0

# Recovery thresholds (hysteresis)
SHRINK_CTX_RESTORE = 10.0
EXPERT_OFFLOAD_RESTORE = 8.0
SWAP_MODEL_RESTORE = 7.0
STOP_AUX_RESTORE = 6.0
STOP_MAIN_RESTORE = 5.0

STARTUP_GRACE = 60
STABILITY_CHECKS = 2

# Context tiers
CTX_FULL = 262144
CTX_MID = 131072
CTX_FLOOR = 65536

# ─── Model swap ladder (main models, largest to smallest) ─────────────────────
MODEL_SWAP_LADDER = [
    ("darwin-28b-reason", 16.6, False, 262144),
    ("darwin-28b-coder", 16.5, False, 262144),
    ("prism-eagle-27b", 13.7, False, 262144),
    ("darwin-apex-36b", 16.0, True, 262144),
    ("carnice-v2-27b", 16.0, False, 262144),
]


def load_prefs():
    prefs = {"api_fallback": {}, "hermes": {}}
    try:
        with open(PREFS) as f:
            import yaml
            data = yaml.safe_load(f) or {}
            prefs["api_fallback"] = data.get("api_fallback", {})
            prefs["hermes"] = data.get("hermes", {})
    except Exception:
        pass
    return prefs


def load_catalog():
    try:
        import yaml
        with open(CATALOG) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_per_gpu_vram():
    """Return list of (gpu_id, free_gb, total_gb, used_gb)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    gpus.append((int(parts[0]), int(parts[1])/1024, int(parts[2])/1024, int(parts[3])/1024))
        return gpus
    except Exception as e:
        log.error(f"nvidia-smi failed: {e}")
        return []


def get_total_free_vram():
    return sum(g[1] for g in get_per_gpu_vram())


def get_running_daemons():
    """Return list of dicts with daemon info."""
    daemons = []
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--no-legend"],
            capture_output=True, text=True, timeout=5
        )
        catalog = load_catalog()
        for line in result.stdout.split("\n"):
            if "turbofit-" not in line or ".service" not in line:
                continue
            parts = line.split()
            for p in parts:
                if p.startswith("turbofit-") and p.endswith(".service"):
                    alias = p.replace("turbofit-", "").replace(".service", "")
                    model = catalog.get("models", {}).get(alias, {})
                    active = "active" in line
                    daemons.append({
                        "alias": alias,
                        "role": model.get("role", ""),
                        "gpu": model.get("gpu", 0),
                        "port": model.get("port", 0),
                        "ctx": model.get("ctx", 262144),
                        "service": p,
                        "active": active,
                        "binary": model.get("binary", ""),
                        "path": model.get("path", ""),
                        "presets": model.get("presets", []),
                        "is_moe": "apex" in alias or "a3b" in alias or "moe" in alias,
                        "size_gb": model.get("size_gb", 0),
                        "mmproj": model.get("mmproj", ""),
                    })
                    break
        # Also check for non-turbofit-prefixed services ({alias}.service)
        # Iterate through ALL catalog model aliases, not just a hardcoded subset,
        # so any model in the catalog (carnice, darwin, prism-eagle, etc.) is
        # detected even when its service isn't prefixed with "turbofit-".
        catalog_aliases = list(catalog.get("models", {}).keys())
        for line in result.stdout.split("\n"):
            for alias in catalog_aliases:
                svc_name = f"{alias}.service"
                if svc_name in line and ".service" in line:
                    model = catalog.get("models", {}).get(alias, {})
                    active = "active" in line
                    already = any(d["alias"] == alias for d in daemons)
                    if not already:
                        daemons.append({
                            "alias": alias,
                            "role": model.get("role", ""),
                            "gpu": model.get("gpu", 0),
                            "port": model.get("port", 0),
                            "ctx": model.get("ctx", 262144),
                            "service": svc_name,
                            "active": active,
                            "binary": model.get("binary", ""),
                            "path": model.get("path", ""),
                            "presets": model.get("presets", []),
                            "is_moe": "apex" in alias or "a3b" in alias or "moe" in alias,
                            "size_gb": model.get("size_gb", 0),
                            "mmproj": model.get("mmproj", ""),
                        })
    except Exception as e:
        log.error(f"Failed to list daemons: {e}")
    return daemons


def daemon_action(action, alias):
    # Try service name variants. For is-active and stop, we must verify the
    # service was actually running before claiming success — otherwise stopping
    # a dead turbofit-{alias}.service returns 0 and we never try {alias}.service.
    #
    # Service name variants (in priority order):
    #   1. {alias}.service             (e.g. darwin.service, carnice.service)
    #   2. turbofit-{alias}.service    (legacy turbofit-managed daemons)
    # Also try common short-form aliases (darwin-28b-reason → darwin).
    candidates = [f"{alias}.service", f"turbofit-{alias}.service"]
    # Add short-form for known long aliases
    short_map = {
        "darwin-28b-reason": "turbofit-darwin-28b-reason.service",
        "darwin-28b-coder": "darwin-coder.service",
    }
    if alias in short_map:
        candidates.insert(0, short_map[alias])

    for svc in candidates:
        try:
            if action == "is-active":
                result = subprocess.run(["systemctl", "--user", "is-active", svc],
                    capture_output=True, text=True, timeout=5)
                if result.stdout.strip() == "active":
                    return True
                # Try next variant
                continue

            elif action == "stop":
                # Only stop if actually running
                check = subprocess.run(["systemctl", "--user", "is-active", svc],
                    capture_output=True, text=True, timeout=5)
                if check.stdout.strip() != "active":
                    continue  # Not running on this service name, try next
                # Use --kill-mode=control-group to force kill child processes
                result = subprocess.run(["systemctl", "--user", "stop", svc],
                    capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    log.info(f"    (stopped {svc})")
                    return True

            elif action == "start":
                # Non-blocking: systemctl start returns once the service is
                # launched, not once the model is loaded. But we MUST verify
                # the service actually has a process — legacy dead services
                # (turbofit-{alias}.service) return 0 on start but do nothing.
                result = subprocess.run(["systemctl", "--user", "start", svc],
                    capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    continue  # Service doesn't exist or failed, try next
                # Verify it actually has a MainPID (not a dead legacy service)
                pid_result = subprocess.run(
                    ["systemctl", "--user", "show", svc, "--property=MainPID", "--value"],
                    capture_output=True, text=True, timeout=5)
                main_pid = pid_result.stdout.strip()
                if main_pid and main_pid != "0":
                    log.info(f"    (starting {svc}, pid={main_pid})")
                    return True
                # No PID — dead legacy service, try next variant
                continue  # Try next variant

            elif action == "restart":
                # Non-blocking: stop, brief pause, start. Don't wait for load.
                subprocess.run(["systemctl", "--user", "stop", svc],
                    capture_output=True, text=True, timeout=15)
                time.sleep(2)
                result = subprocess.run(["systemctl", "--user", "start", svc],
                    capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    log.info(f"    (restarting {svc})")
                    return True
                continue
        except Exception as e:
            log.error(f"daemon_action {action} {svc}: {e}")
    return False


# ─── Multi-profile management ─────────────────────────────────────────────────

def find_local_profiles():
    """Find ALL Hermes profiles pointing at localhost."""
    profiles = []
    profiles_dir = Path(HERMES_HOME) / "profiles"
    if not profiles_dir.exists():
        return profiles
    for profile_dir in profiles_dir.iterdir():
        if not profile_dir.is_dir():
            continue
        cfg_path = profile_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("model", {})
            base_url = model.get("base_url", "")
            if "127.0.0.1" in base_url or "localhost" in base_url:
                profiles.append({
                    "name": profile_dir.name,
                    "config_path": str(cfg_path),
                    "model": model.get("default", ""),
                    "base_url": base_url,
                })
        except Exception:
            pass
    return profiles


def set_hermes_model(config_path, model_id, base_url, provider=None):
    try:
        import yaml
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        if "model" not in cfg:
            cfg["model"] = {}
        cfg["model"]["default"] = model_id
        cfg["model"]["base_url"] = base_url
        if provider:
            cfg["model"]["provider"] = provider
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        log.error(f"set_hermes_model failed for {config_path}: {e}")
        return False


def switch_all_local_profiles_to_api(fallback_model, fallback_url, fallback_provider):
    profiles = find_local_profiles()
    switched = []
    for p in profiles:
        log.info(f"  Switching {p['name']} → API: {fallback_model}")
        if set_hermes_model(p["config_path"], fallback_model, fallback_url, fallback_provider):
            switched.append(p)
    return switched


def restore_all_profiles_to_local(local_model, local_url, local_provider):
    profiles_dir = Path(HERMES_HOME) / "profiles"
    restored = []
    if not profiles_dir.exists():
        return restored
    for profile_dir in profiles_dir.iterdir():
        if not profile_dir.is_dir():
            continue
        cfg_path = profile_dir / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            import yaml
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("model", {})
            base_url = model.get("base_url", "")
            if "inference-api" in base_url or "127.0.0.1" not in base_url:
                log.info(f"  Restoring {profile_dir.name} → local: {local_model}")
                set_hermes_model(str(cfg_path), local_model, local_url, local_provider)
                restored.append(profile_dir.name)
        except Exception:
            pass
    return restored


# ─── Catalog mutation ─────────────────────────────────────────────────────────

def update_catalog_model(alias, updates):
    try:
        import yaml
        with open(CATALOG) as f:
            cfg = yaml.safe_load(f) or {}
        if alias in cfg.get("models", {}):
            cfg["models"][alias].update(updates)
            with open(CATALOG, "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            return True
    except Exception as e:
        log.error(f"update_catalog_model failed: {e}")
    return False


def get_catalog_model(alias):
    try:
        import yaml
        with open(CATALOG) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("models", {}).get(alias, {})
    except:
        return {}


# ─── Contraction actions ──────────────────────────────────────────────────────

def shrink_context(alias, current_ctx):
    if current_ctx > CTX_MID:
        new_ctx = CTX_MID
    elif current_ctx > CTX_FLOOR:
        new_ctx = CTX_FLOOR
    else:
        return None
    log.info(f"  ▼ SHRINK CTX: {alias} {current_ctx} → {new_ctx}")
    update_catalog_model(alias, {"ctx": new_ctx})
    daemon_action("restart", alias)
    return new_ctx


def restore_context(alias, current_ctx):
    if current_ctx < CTX_FLOOR:
        new_ctx = CTX_FLOOR
    elif current_ctx < CTX_MID:
        new_ctx = CTX_MID
    elif current_ctx < CTX_FULL:
        new_ctx = CTX_FULL
    else:
        return None
    log.info(f"  ▲ RESTORE CTX: {alias} {current_ctx} → {new_ctx}")
    update_catalog_model(alias, {"ctx": new_ctx})
    daemon_action("restart", alias)
    return new_ctx


def add_expert_offload(alias):
    model = get_catalog_model(alias)
    presets = model.get("presets", [])
    if "cpu-moe" in presets:
        return False
    presets.append("cpu-moe")
    log.info(f"  ▼ EXPERT OFFLOAD: {alias} — experts → CPU")
    update_catalog_model(alias, {"presets": presets})
    daemon_action("restart", alias)
    return True


def remove_expert_offload(alias):
    model = get_catalog_model(alias)
    presets = model.get("presets", [])
    if "cpu-moe" not in presets:
        return False
    presets = [p for p in presets if p != "cpu-moe"]
    log.info(f"  ▲ RESTORE EXPERTS: {alias} — experts → GPU")
    update_catalog_model(alias, {"presets": presets})
    daemon_action("restart", alias)
    return True


def swap_to_smaller_model(current_alias, daemons):
    current_idx = None
    for i, (alias, size, is_moe, ctx) in enumerate(MODEL_SWAP_LADDER):
        if alias == current_alias:
            current_idx = i
            break
    if current_idx is None:
        current_model = get_catalog_model(current_alias)
        current_size = current_model.get("size_gb", 99)
        for i, (alias, size, is_moe, ctx) in enumerate(MODEL_SWAP_LADDER):
            if size < current_size:
                current_idx = i - 1
                break
    if current_idx is None or current_idx >= len(MODEL_SWAP_LADDER) - 1:
        return None

    next_alias, next_size, next_moe, next_ctx = MODEL_SWAP_LADDER[current_idx + 1]
    next_model = get_catalog_model(next_alias)
    if not next_model or not os.path.exists(next_model.get("path", "")):
        log.warning(f"  Swap target {next_alias} not available")
        return None

    log.info(f"  ▼ SWAP: {current_alias} ({MODEL_SWAP_LADDER[current_idx][1]}GB) → "
             f"{next_alias} ({next_size}GB)")
    daemon_action("stop", current_alias)
    time.sleep(3)
    daemon_action("start", next_alias)
    return next_alias


# ─── Main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="turbofit scaling watcher v2")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval seconds")
    parser.add_argument("--dry-run", action="store_true", help="Log but don't execute")
    args = parser.parse_args()

    prefs = load_prefs()
    fallback_main = prefs.get("api_fallback", {}).get("main", DEFAULT_MAIN_FALLBACK)
    fallback_url = prefs.get("api_fallback", {}).get("base_url", DEFAULT_BASE_URL)
    fallback_provider = prefs.get("api_fallback", {}).get("provider", DEFAULT_PROVIDER)

    local = prefs.get("api_fallback", {}).get("local", {})
    local_model = local.get("main", "darwin-28b-reason")
    local_url = local.get("base_url", "http://127.0.0.1:8091/main/v1")
    local_provider = local.get("provider", "localhost")

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  turbofit scaling watcher v2 — gradual contraction      ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info(f"  Poll: {args.interval}s | Dry-run: {args.dry_run}")
    log.info(f"  Contraction: ctx<{SHRINK_CTX_THRESHOLD}GB → expert<{EXPERT_OFFLOAD_THRESHOLD}GB → "
             f"swap<{SWAP_MODEL_THRESHOLD}GB → aux<{STOP_AUX_THRESHOLD}GB → main<{STOP_MAIN_THRESHOLD}GB")
    log.info(f"  Recovery: ctx>{SHRINK_CTX_RESTORE}GB → expert>{EXPERT_OFFLOAD_RESTORE}GB → "
             f"swap>{SWAP_MODEL_RESTORE}GB → aux>{STOP_AUX_RESTORE}GB → main>{STOP_MAIN_RESTORE}GB")
    log.info(f"  API fallback: {fallback_main} @ {fallback_url}")
    log.info(f"  Local main:   {local_model} @ {local_url}")
    log.info(f"  Multi-profile: manages ALL profiles pointing at localhost")

    # State
    contraction_level = 0
    swapped_to_alias = None
    api_switched_profiles = []
    consecutive_failures = 0
    last_action_time = 0
    stability_counter = 0
    last_target_level = 0

    # Startup: auto-start main daemon if VRAM is healthy
    gpus = get_per_gpu_vram()
    if gpus:
        total_free = sum(g[1] for g in gpus)
        log.info(f"  Startup: {total_free:.1f}GB total free VRAM")
        for gpu_id, free, total, used in gpus:
            log.info(f"    GPU {gpu_id}: {free:.1f}GB free / {total:.1f}GB total ({used:.1f}GB used)")
        daemons = get_running_daemons()
        main_running = any(d["active"] and d["role"] == "main" for d in daemons)
        if not main_running and total_free > STOP_MAIN_RESTORE + 5:
            log.info(f"  Startup: No main daemon running — starting {local_model}")
            daemon_action("start", local_model)
            time.sleep(10)
        last_action_time = time.time()

    log.info("─" * 60)

    while True:
        try:
            gpus = get_per_gpu_vram()
            if not gpus:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    log.warning("nvidia-smi failed 10 times, exiting")
                    sys.exit(1)
                time.sleep(args.interval)
                continue
            consecutive_failures = 0

            daemons = get_running_daemons()
            active_daemons = [d for d in daemons if d["active"]]

            # Use TOTAL free VRAM for contraction decisions.
            # BUT: subtract the VRAM used by turbofit's own daemons — their usage
            # is expected, not external pressure. We only contract when
            # EXTERNAL processes (ACE Step, ComfyUI, turbohaul, etc) eat VRAM.
            total_free = sum(g[1] for g in gpus)

            # Calculate how much VRAM turbofit daemons are ACTUALLY using
            # by querying nvidia-smi for per-process VRAM and matching PIDs
            # to turbofit's llama-server processes.
            daemon_pids = set()
            for d in active_daemons:
                # Find PIDs of llama-server processes matching this daemon
                try:
                    result = subprocess.run(
                        ["pgrep", "-f", d.get("path", "")],
                        capture_output=True, text=True, timeout=3
                    )
                    for pid_line in result.stdout.strip().split("\n"):
                        if pid_line.strip():
                            daemon_pids.add(int(pid_line.strip()))
                except Exception:
                    pass

            # Get per-process VRAM from nvidia-smi
            daemon_vram_mb = 0
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) >= 2:
                            pid = int(parts[0])
                            mem_mb = int(parts[1])
                            if pid in daemon_pids:
                                daemon_vram_mb += mem_mb
            except Exception:
                pass

            daemon_vram = daemon_vram_mb / 1024  # Convert to GB

            # How much VRAM is being used by non-turbofit processes?
            total_vram = sum(g[2] for g in gpus)  # total capacity
            total_used = total_vram - total_free
            non_turbofit_used = total_used - daemon_vram
            if non_turbofit_used < 0:
                non_turbofit_used = 0

            # Find the GPU with minimum free VRAM (for logging)
            min_free_gb = 999
            min_free_gpu = 0
            for gpu_id, free, total, used in gpus:
                if free < min_free_gb:
                    min_free_gb = free
                    min_free_gpu = gpu_id

            log.debug(f"VRAM: total_free={total_free:.1f}GB, "
                      f"non_turbofit_used={non_turbofit_used:.1f}GB, "
                      f"min_gpu{min_free_gpu}={min_free_gb:.1f}GB, "
                      f"contraction={contraction_level}")

            # ─── CONTRACTION (scale down) ───────────────────────────────────
            # Contract based on ABSOLUTE free VRAM per GPU.
            # When any GPU gets tight, turbofit sheds models to make room.
            # This handles BOTH external pressure (ComfyUI, games) AND
            # turbofit-internal pressure (ACE-Step, another daemon loading).
            #
            # The original external-only logic missed the case where a
            # turbofit daemon itself causes VRAM pressure (e.g. ACE-Step
            # loading alongside Darwin + Carnice). We now use per-GPU
            # free VRAM as the primary contraction signal.
            target_level = 0

            # Contract based on EXTERNAL (non-turbofit) VRAM usage.
            # Turbofit's own models consuming VRAM is expected.
            if non_turbofit_used > 14:
                target_level = 5
            elif non_turbofit_used > 10:
                target_level = 4
            elif non_turbofit_used > 6:
                target_level = 3
            elif non_turbofit_used > 4:
                target_level = 2
            elif non_turbofit_used > 3:
                target_level = 1

            if target_level > contraction_level:
                # Stability check
                if target_level > last_target_level:
                    stability_counter = 1
                elif target_level == last_target_level:
                    stability_counter += 1
                else:
                    stability_counter = 0
                last_target_level = target_level

                # Startup grace
                grace_elapsed = time.time() - last_action_time
                if grace_elapsed < STARTUP_GRACE or stability_counter < STABILITY_CHECKS:
                    time.sleep(args.interval)
                    continue

                # One level per poll
                capped_target = min(target_level, contraction_level + 1)

                log.warning(f"⚠ CONTRACT: level {contraction_level} → {capped_target} "
                           f"(min_free: {min_free_gb:.1f}GB on GPU {min_free_gpu}, "
                           f"external: {non_turbofit_used:.1f}GB, total_free: {total_free:.1f}GB)")

                if not args.dry_run:
                    active_main = next((d for d in active_daemons if d["role"] == "main"), None)

                    if capped_target >= 1 and contraction_level < 1 and active_main:
                        new_ctx = shrink_context(active_main["alias"], active_main["ctx"])
                        contraction_level = 1
                        last_action_time = time.time()
                        stability_counter = 0

                    if capped_target >= 2 and contraction_level < 2 and active_main:
                        if active_main.get("is_moe"):
                            add_expert_offload(active_main["alias"])
                        contraction_level = 2
                        last_action_time = time.time()
                        stability_counter = 0

                    if capped_target >= 3 and contraction_level < 3 and active_main:
                        new_alias = swap_to_smaller_model(active_main["alias"], active_daemons)
                        if new_alias:
                            swapped_to_alias = new_alias
                        contraction_level = 3
                        last_action_time = time.time()
                        stability_counter = 0

                    if capped_target >= 4 and contraction_level < 4:
                        for d in active_daemons:
                            if d["role"] == "aux":
                                log.info(f"  ▼ STOP AUX: {d['alias']}")
                                daemon_action("stop", d["alias"])
                        contraction_level = 4
                        last_action_time = time.time()
                        stability_counter = 0

                    if capped_target >= 5 and contraction_level < 5:
                        for d in active_daemons:
                            if d["role"] == "main":
                                log.info(f"  ▼ STOP MAIN: {d['alias']} → API mode")
                                daemon_action("stop", d["alias"])
                        api_switched_profiles = switch_all_local_profiles_to_api(
                            fallback_main, fallback_url, fallback_provider
                        )
                        log.info(f"  Switched {len(api_switched_profiles)} profiles to API")
                        contraction_level = 5
                        last_action_time = time.time()
                        stability_counter = 0

            # ─── EXPANSION ──────────────────────────────────────────────
            elif target_level < contraction_level:
                # Expand when per-GPU free VRAM recovers (with hysteresis).
                # Uses absolute free VRAM thresholds (+4GB above contraction).
                RESTORE_CTX = 10.0       # Restore full ctx when > 10GB free
                RESTORE_EXPERTS = 8.0    # Restore experts when > 8GB free
                RESTORE_SWAP = 7.0       # Swap back to big model when > 7GB free
                RESTORE_AUX = 6.0        # Restart aux when > 6GB free
                RESTORE_MAIN = 5.0       # Restart main when > 5GB free

                restore_level = contraction_level
                if contraction_level >= 5 and non_turbofit_used < 10:
                    restore_level = 4
                elif contraction_level >= 4 and non_turbofit_used < 8:
                    restore_level = 3
                elif contraction_level >= 3 and non_turbofit_used < 5:
                    restore_level = 2
                elif contraction_level >= 2 and non_turbofit_used < 3:
                    restore_level = 1
                elif contraction_level >= 1 and non_turbofit_used < 2:
                    restore_level = 0

                if restore_level < contraction_level:
                    log.info(f"✓ EXPAND: level {contraction_level} → {restore_level} "
                            f"(min_free: {min_free_gb:.1f}GB on GPU {min_free_gpu}, "
                            f"total free: {total_free:.1f}GB)")

                    if not args.dry_run:
                        if contraction_level >= 5 and restore_level < 5:
                            start_alias = swapped_to_alias or local_model
                            log.info(f"  ▲ START MAIN: {start_alias}")
                            daemon_action("start", start_alias)
                            last_action_time = time.time()
                            time.sleep(10)
                            restore_all_profiles_to_local(local_model, local_url, local_provider)
                            api_switched_profiles = []

                        if contraction_level >= 4 and restore_level < 4:
                            all_daemons = get_running_daemons()
                            per_gpu = get_per_gpu_vram()
                            for d in all_daemons:
                                if d["role"] == "aux" and not d["active"]:
                                    # Check the free VRAM on the SPECIFIC GPU
                                    # this aux daemon is assigned to — not the
                                    # sum across all GPUs (which could be high
                                    # while the target GPU is still full).
                                    target_gpu_id = d.get("gpu", 0)
                                    gpu_free = next(
                                        (free for (gid, free, _t, _u) in per_gpu
                                         if gid == target_gpu_id),
                                        0.0,
                                    )
                                    needed = d["size_gb"] + 2  # 2GB headroom
                                    if gpu_free >= needed:
                                        log.info(f"  ▲ START AUX: {d['alias']} "
                                                 f"(GPU {target_gpu_id} free {gpu_free:.1f}GB >= {needed:.1f}GB)")
                                        daemon_action("start", d["alias"])
                                        last_action_time = time.time()
                                        time.sleep(5)
                                    else:
                                        log.debug(f"  SKIP AUX {d['alias']}: GPU {target_gpu_id} "
                                                  f"free {gpu_free:.1f}GB < {needed:.1f}GB needed")

                        if contraction_level >= 3 and restore_level < 3 and swapped_to_alias:
                            log.info(f"  ▲ SWAP BACK: {swapped_to_alias} → {local_model}")
                            daemon_action("stop", swapped_to_alias)
                            time.sleep(3)
                            daemon_action("start", local_model)
                            last_action_time = time.time()
                            swapped_to_alias = None

                        if contraction_level >= 2 and restore_level < 2:
                            for d in get_running_daemons():
                                if d["active"] and d.get("is_moe"):
                                    remove_expert_offload(d["alias"])

                        if contraction_level >= 1 and restore_level < 1:
                            for d in get_running_daemons():
                                if d["active"] and d["role"] == "main":
                                    restore_context(d["alias"], d["ctx"])

                        contraction_level = restore_level

            # ─── Auto-recovery: daemon died but VRAM healthy ───────────
            if contraction_level < 5:
                active_daemons = get_running_daemons()
                main_running = any(d["active"] and d["role"] == "main" for d in active_daemons)
                # Only auto-restart if there's enough room (external usage is low)
                if not main_running and non_turbofit_used < 20:
                    log.warning(f"Main daemon dead but external usage only {non_turbofit_used:.1f}GB — auto-restart {local_model}")
                    daemon_action("start", local_model)
                    last_action_time = time.time()
                    time.sleep(10)

            # ─── Endpoint health check ──────────────────────────────
            # Actually probe the main endpoint. If systemctl says active
            # but the endpoint isn't serving, restart it.
            #
            # BUG HISTORY: this check used to hit `/models` or `/v1models`
            # (both 404 on llama-server), which caused the watcher to think
            # the daemon was down during its 20s load window and enter a
            # restart loop — kill the daemon 20s after it became ready,
            # poll again, see 404 (loading), kill again, ad infinitum.
            #
            # Fix:
            #  1. Probe the actual llama-server endpoint `/v1/models` (or
            #     fall back to `/health`).
            #  2. Strip the trailing /v1 from local_url before probing so
            #     the constructed URL is correct.
            #  3. Honor a startup grace window using the systemd unit's
            #     ActiveEnterTimestamp, not `last_action_time`. The bug in
            #     the previous attempt was that `last_action_time` only
            #     updates when THIS WATCHER triggers a restart — if the
            #     daemon dies for any OTHER reason (OOM, manual kill, system
            #     pressure, another watcher), `last_action_time` is stale
            #     and grace check passes immediately, causing an immediate
            #     restart.
            #
            # Now: if the daemon has been alive for less than 60s (i.e. it
            # just restarted and is still loading its 16GB GGUF into VRAM),
            # skip the probe entirely. 60s covers the realistic load time
            # (~20s) + systemd overhead + the watcher's own sleep(10).
            if contraction_level < 5:
                endpoint_ok = False
                # Check systemd ActiveEnterTimestamp — when the unit last
                # came up, regardless of why.
                import subprocess as _sp
                try:
                    _r = _sp.run(
                        ["systemctl", "--user", "show", f"turbofit-{local_model}.service",
                         "--property=ActiveEnterTimestamp", "--value"],
                        capture_output=True, text=True, timeout=5
                    )
                    _active_since = _r.stdout.strip()
                    # Format: "Tue 2026-06-30 11:53:53 CDT" — parse to epoch
                    from datetime import datetime as _dt
                    try:
                        _ts = _dt.strptime(_active_since, "%a %Y-%m-%d %H:%M:%S %Z").timestamp()
                    except Exception:
                        _ts = 0
                except Exception:
                    _ts = 0

                daemon_age = time.time() - _ts if _ts else 999

                if daemon_age < 60.0:
                    # Daemon just came up within the last 60s — it's still
                    # loading its GGUF. Don't probe. This breaks the
                    # restart loop at the root, regardless of WHY the
                    # daemon restarted.
                    endpoint_ok = True
                    if int(time.time()) % 30 < 2:  # log every ~30s, not every poll
                        log.info(f"  ⏳ Main daemon alive {daemon_age:.0f}s — grace window, skipping probe")
                else:
                    import urllib.request
                    # local_url is e.g. "http://127.0.0.1:11500/v1" — strip
                    # the /v1 suffix so we can probe canonical llama-server
                    # paths. Try /v1/models first (most reliable), then
                    # /health as a fallback.
                    base = local_url.rstrip("/")
                    if base.endswith("/v1"):
                        base = base[:-3]
                    probe_paths = ["/v1/models", "/health"]
                    for probe in probe_paths:
                        try:
                            req = urllib.request.Request(base + probe)
                            with urllib.request.urlopen(req, timeout=3) as resp:
                                if resp.status == 200:
                                    endpoint_ok = True
                                    break
                        except Exception:
                            continue

                if not endpoint_ok and non_turbofit_used < 15:
                    log.warning(f"Main endpoint {local_url} is DOWN but VRAM is free — restarting {local_model}")
                    daemon_action("restart", local_model)
                    last_action_time = time.time()
                    time.sleep(10)

            # ─── Status every ~2 min ────────────────────────────────────
            if int(time.time()) % (args.interval * 8) < args.interval:
                level_names = ["HEALTHY", "CTX_SHRUNK", "EXPERT_OFFLOAD",
                               "MODEL_SWAPPED", "AUX_STOPPED", "API_ONLY"]
                active_count = len([d for d in get_running_daemons() if d["active"]])
                log.info(f"Status: {level_names[contraction_level]} | "
                        f"free={total_free:.1f}GB | "
                        f"external={non_turbofit_used:.1f}GB | "
                        f"min_gpu{min_free_gpu}={min_free_gb:.1f}GB | "
                        f"active_daemons={active_count}")

        except KeyboardInterrupt:
            log.info("Shutting down — restoring all models")
            for d in get_running_daemons():
                if not d["active"] and d["role"] in ("main", "aux"):
                    daemon_action("start", d["alias"])
            restore_all_profiles_to_local(local_model, local_url, local_provider)
            sys.exit(0)
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
