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
        # Also check for non-turbofit-prefixed services (carnice.service, darwin.service)
        for line in result.stdout.split("\n"):
            for short_name in ("carnice", "darwin"):
                if f"{short_name}.service" in line and ".service" in line:
                    alias = short_name
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
                            "service": f"{short_name}.service",
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
    # Support both turbofit-{alias} and bare {alias} service names
    for svc in (f"turbofit-{alias}.service", f"{alias}.service"):
        try:
            if action == "stop":
                result = subprocess.run(["systemctl", "--user", "stop", svc],
                    capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return True
            elif action == "start":
                result = subprocess.run(["systemctl", "--user", "start", svc],
                    capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return True
            elif action == "is-active":
                result = subprocess.run(["systemctl", "--user", "is-active", svc],
                    capture_output=True, text=True, timeout=5)
                return result.stdout.strip() == "active"
            elif action == "restart":
                subprocess.run(["systemctl", "--user", "stop", svc],
                    capture_output=True, text=True, timeout=30)
                time.sleep(2)
                subprocess.run(["systemctl", "--user", "start", svc],
                    capture_output=True, text=True, timeout=30)
                return True
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
    local_url = local.get("base_url", "http://127.0.0.1:11500/v1")
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
            # Contract based on EXTERNAL VRAM usage (non-turbofit processes).
            # When external processes use too much VRAM, we contract turbofit
            # to make room. This prevents the "I keep having to stop models
            # manually" problem — turbofit backs off automatically.
            target_level = 0
            # Contract based on how much VRAM external (non-turbofit) processes are using.
            # When external usage exceeds these thresholds, turbofit contracts to make room.
            # Thresholds are cumulative: at each level, turbofit sheds more VRAM.
            EXTERNAL_PRESSURE_LOW = 8.0     # Shrink ctx when external usage > 8GB
            EXTERNAL_PRESSURE_MED = 12.0   # Expert offload when > 12GB external
            EXTERNAL_PRESSURE_HIGH = 16.0  # Swap to smaller model when > 16GB external
            EXTERNAL_PRESSURE_CRITICAL = 20.0  # Stop aux when > 20GB external
            EXTERNAL_PRESSURE_MAX = 24.0   # Stop main, go API when > 24GB external

            if non_turbofit_used > EXTERNAL_PRESSURE_MAX:
                target_level = 5
            elif non_turbofit_used > EXTERNAL_PRESSURE_CRITICAL:
                target_level = 4
            elif non_turbofit_used > EXTERNAL_PRESSURE_HIGH:
                target_level = 3
            elif non_turbofit_used > EXTERNAL_PRESSURE_MED:
                target_level = 2
            elif non_turbofit_used > EXTERNAL_PRESSURE_LOW:
                target_level = 1

            if target_level > contraction_level:
                log.warning(f"⚠ CONTRACT: level {contraction_level} → {target_level} "
                           f"(external: {non_turbofit_used:.1f}GB, free: {total_free:.1f}GB)")

                if not args.dry_run:
                    active_main = next((d for d in active_daemons if d["role"] == "main"), None)

                    if target_level >= 1 and contraction_level < 1 and active_main:
                        new_ctx = shrink_context(active_main["alias"], active_main["ctx"])
                        contraction_level = 1

                    if target_level >= 2 and contraction_level < 2 and active_main:
                        if active_main.get("is_moe"):
                            add_expert_offload(active_main["alias"])
                        contraction_level = 2

                    if target_level >= 3 and contraction_level < 3 and active_main:
                        new_alias = swap_to_smaller_model(active_main["alias"], active_daemons)
                        if new_alias:
                            swapped_to_alias = new_alias
                        contraction_level = 3

                    if target_level >= 4 and contraction_level < 4:
                        for d in active_daemons:
                            if d["role"] == "aux":
                                log.info(f"  ▼ STOP AUX: {d['alias']}")
                                daemon_action("stop", d["alias"])
                        contraction_level = 4

                    if target_level >= 5 and contraction_level < 5:
                        for d in active_daemons:
                            if d["role"] == "main":
                                log.info(f"  ▼ STOP MAIN: {d['alias']} → API mode")
                                daemon_action("stop", d["alias"])
                        api_switched_profiles = switch_all_local_profiles_to_api(
                            fallback_main, fallback_url, fallback_provider
                        )
                        log.info(f"  Switched {len(api_switched_profiles)} profiles to API")
                        contraction_level = 5

            # ─── EXPANSION ──────────────────────────────────────────────
            elif target_level < contraction_level:
                # Expand when external pressure drops (with hysteresis)
                EXPRESSION_LOW = 6.0      # Expand ctx when external < 6GB
                EXPRESSION_MED = 10.0    # Restore experts when external < 10GB
                EXPRESSION_HIGH = 14.0   # Swap back when external < 14GB
                EXPRESSION_CRITICAL = 18.0  # Restart aux when external < 18GB
                EXPRESSION_MAX = 22.0    # Restart main when external < 22GB

                restore_level = contraction_level
                if contraction_level >= 5 and non_turbofit_used < EXPRESSION_MAX:
                    restore_level = 4
                elif contraction_level >= 4 and non_turbofit_used < EXPRESSION_CRITICAL:
                    restore_level = 3
                elif contraction_level >= 3 and non_turbofit_used < EXPRESSION_HIGH:
                    restore_level = 2
                elif contraction_level >= 2 and non_turbofit_used < EXPRESSION_MED:
                    restore_level = 1
                elif contraction_level >= 1 and non_turbofit_used < EXPRESSION_LOW:
                    restore_level = 0

                if restore_level < contraction_level:
                    log.info(f"✓ EXPAND: level {contraction_level} → {restore_level} "
                            f"(external: {non_turbofit_used:.1f}GB, total free: {total_free:.1f}GB)")

                    if not args.dry_run:
                        if contraction_level >= 5 and restore_level < 5:
                            start_alias = swapped_to_alias or local_model
                            log.info(f"  ▲ START MAIN: {start_alias}")
                            daemon_action("start", start_alias)
                            time.sleep(10)
                            restore_all_profiles_to_local(local_model, local_url, local_provider)
                            api_switched_profiles = []

                        if contraction_level >= 4 and restore_level < 4:
                            all_daemons = get_running_daemons()
                            for d in all_daemons:
                                if d["role"] == "aux" and not d["active"]:
                                    if d["size_gb"] < total_free:
                                        log.info(f"  ▲ START AUX: {d['alias']}")
                                        daemon_action("start", d["alias"])
                                        time.sleep(5)

                        if contraction_level >= 3 and restore_level < 3 and swapped_to_alias:
                            log.info(f"  ▲ SWAP BACK: {swapped_to_alias} → {local_model}")
                            daemon_action("stop", swapped_to_alias)
                            time.sleep(3)
                            daemon_action("start", local_model)
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
