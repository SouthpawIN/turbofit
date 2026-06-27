#!/usr/bin/env python3
"""
turbofit-scaling-watcher — full scaling ladder with automatic API fallback.

Turbofit's "conscious stream" — monitors GPU VRAM and automatically scales down
through four tiers, switching between local and API models without user intervention.

Ladder:
  Tier 0 (≥24GB free): Everything running — main + aux local, full context
  Tier 1 (16-24GB free): Stop aux, main stays on full context
  Tier 2 (8-16GB free): Stop main, switch Hermes config to API fallback
  Tier 3 (<8GB free): API-only mode — ensure everything local is stopped

Recovery is progressive in reverse — API→local→aux.
Hysteresis prevents flapping: restore thresholds are 4GB higher than stop thresholds.

Preferences:
  Set in ~/.config/turbofit/preferences.yaml:
    api_fallback:
      main: "z-ai/glm-5.2"          # API model ID for main
      base_url: "https://inference-api.nousresearch.com/v1"
      provider: "nous"
    hermes:
      profile: "senter"              # which Hermes profile to reconfigure
"""

import subprocess
import json
import time
import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scale] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scale")

# Paths
HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
PREFS = os.environ.get("TURBOFIT_PREFS", f"{HOME}/.config/turbofit/preferences.yaml")
HERMES_HOME = os.environ.get("HERMES_HOME", f"{HOME}/.hermes")

# Default fallback preferences
DEFAULT_MAIN_FALLBACK = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_PROVIDER = "nous"
DEFAULT_PROFILE = "senter"

# Scaling thresholds (GB free across all GPUs)
TIER1_STOP = 16.0   # Below this: stop aux
TIER2_STOP = 8.0    # Below this: stop main, switch to API
TIER3_STOP = 4.0    # Below this: critical, ensure everything stopped

# Recovery thresholds (higher than stop = hysteresis)
TIER2_RESTORE = 20.0  # Above this: switch back to local, restart main
TIER1_RESTORE = 24.0  # Above this: restart aux


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


def get_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        free_mib = 0
        total_mib = 0
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [int(x.strip()) for x in line.split(",") if x.strip()]
                if len(parts) >= 2:
                    free_mib += parts[0]
                    total_mib += parts[1]
        return free_mib / 1024
    except Exception as e:
        return None


def get_running_daemons(role_filter=None):
    """Return list of (alias, role) tuples for running turbofit daemons."""
    daemons = []
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--no-legend"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "turbofit-" not in line or ".service" not in line:
                continue
            parts = line.split()
            for p in parts:
                if p.startswith("turbofit-") and p.endswith(".service"):
                    alias = p.replace("turbofit-", "").replace(".service", "")
                    role = get_catalog_role(alias)
                    if role_filter and role != role_filter:
                        break
                    daemons.append((alias, role))
                    break
    except Exception as e:
        log.error(f"Failed to list systemd services: {e}")
    return daemons


def get_catalog_role(alias):
    try:
        result = subprocess.run(
            ["python3", "-c", f"""
import yaml
with open("{CATALOG}") as f: cfg = yaml.safe_load(f) or {{}}
print(cfg.get("models", {{}}).get("{alias}", {{}}).get("role", ""))
"""],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except:
        return ""


def daemon_action(action, alias):
    """Start, stop, or check a turbofit daemon."""
    svc = f"turbofit-{alias}.service"
    if action == "stop":
        result = subprocess.run(
            ["systemctl", "--user", "stop", svc],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    elif action == "start":
        result = subprocess.run(
            ["systemctl", "--user", "start", svc],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    elif action == "is-active":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", svc],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == "active"
    return False


def get_hermes_config_path(profile):
    """Get the config.yaml path for a Hermes profile."""
    if profile == "default" or not profile:
        return f"{HERMES_HOME}/config.yaml"
    return f"{HERMES_HOME}/profiles/{profile}/config.yaml"


def get_current_hermes_model(config_path):
    """Read current model.default from Hermes config."""
    try:
        result = subprocess.run(
            ["python3", "-c", f"""
import yaml
with open("{config_path}") as f: cfg = yaml.safe_load(f) or {{}}
m = cfg.get("model", {{}}).get("default", "")
b = cfg.get("model", {{}}).get("base_url", "")
print(f"{{m}}|{{b}}")
"""],
            capture_output=True, text=True, timeout=5
        )
        model, base_url = result.stdout.strip().split("|", 1)
        return model, base_url
    except:
        return "", ""


def is_hermes_on_api(config_path):
    """Check if Hermes is currently configured to use a remote API (not localhost)."""
    _, base_url = get_current_hermes_model(config_path)
    return "127.0.0.1" not in base_url and "localhost" not in base_url


def set_hermes_model(config_path, model_id, base_url, provider=None):
    """Update Hermes config to use a specific model and base URL."""
    provider_str = f'"{provider}"' if provider else "null"
    script = f"""
import yaml, os
path = \"{config_path}\"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path) as f: cfg = yaml.safe_load(f) or {{}}
if 'model' not in cfg: cfg['model'] = {{}}
cfg['model']['default'] = \"{model_id}\"
cfg['model']['base_url'] = \"{base_url}\"
cfg['model']['provider'] = \"{provider}\" if \"{provider}\" != \"null\" else cfg['model'].get('provider', 'nous')
with open(path, 'w') as f: yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print('OK')
"""
    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False


def get_local_main_config():
    """Get the local main model config for Hermes from preferences."""
    prefs = load_prefs()
    local = prefs.get("api_fallback", {}).get("local", {})
    return {
        "model_id": local.get("main", "darwin-28b-reason"),
        "base_url": local.get("base_url", "http://127.0.0.1:11500/v1"),
        "provider": local.get("provider", "localhost"),
    }


def main():
    parser = argparse.ArgumentParser(description="turbofit scaling watcher")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    parser.add_argument("--tier1", type=float, default=TIER1_STOP, help="Stop aux below this GB free")
    parser.add_argument("--tier2", type=float, default=TIER2_STOP, help="Stop main + API fallback below this GB")
    parser.add_argument("--tier3", type=float, default=TIER3_STOP, help="Critical VRAM threshold")
    parser.add_argument("--restore2", type=float, default=TIER2_RESTORE, help="Restore local main above this GB")
    parser.add_argument("--restore1", type=float, default=TIER1_RESTORE, help="Restore aux above this GB")
    args = parser.parse_args()

    prefs = load_prefs()
    profile = prefs.get("hermes", {}).get("profile", DEFAULT_PROFILE)
    config_path = get_hermes_config_path(profile)

    fallback_main = prefs.get("api_fallback", {}).get("main", DEFAULT_MAIN_FALLBACK)
    fallback_url = prefs.get("api_fallback", {}).get("base_url", DEFAULT_BASE_URL)
    fallback_provider = prefs.get("api_fallback", {}).get("provider", DEFAULT_PROVIDER)

    local = get_local_main_config()
    local_model = local["model_id"]
    local_url = local["base_url"]

    log.info(f"turbofit scaling watcher started")
    log.info(f"  Poll: {args.interval}s")
    log.info(f"  Tier 1 (stop aux):    <{args.tier1}GB")
    log.info(f"  Tier 2 (API fallback): <{args.tier2}GB")
    log.info(f"  Tier 3 (critical):    <{args.tier3}GB")
    log.info(f"  Restore local main:   >{args.restore2}GB")
    log.info(f"  Restore aux:          >{args.restore1}GB")
    log.info(f"  Fallback API: {fallback_main} @ {fallback_url}")
    log.info(f"  Local main:   {local_model} @ {local_url}")
    log.info(f"  Hermes config: {config_path}")

    # State tracking
    running_tier = 0  # 0=normal, 1=aux_stopped, 2=api_fallback, 3=critical
    stopped_main = set()
    consecutive_failures = 0

    # --- Startup: check if we should already be on local ---
    # If VRAM is healthy and Hermes is on API, switch to local immediately
    if not is_hermes_on_api(config_path):
        log.info(f"Hermes already on local — no startup transition needed")
    else:
        free_gb = get_vram()
        if free_gb and free_gb > args.restore2:
            # VRAM healthy but Hermes is on API — likely a system restart,
            # switch back to local
            log.info(f"Startup: VRAM healthy ({free_gb:.1f}GB > {args.restore2}GB) but Hermes on API — switching to local")
            set_hermes_model(config_path, local_model, local_url, "localhost")
            # Restart main daemons if they're stopped
            for alias, role in get_running_daemons():
                if role == "main" and not daemon_action("is-active", alias):
                    daemon_action("start", alias)
                    log.info(f"  Startup: started main daemon {alias}")
        else:
            log.info(f"Startup: staying on API (free={free_gb}GB)")

    def stop_all_aux():
        stopped = set()
        for alias, role in get_running_daemons(role_filter="aux"):
            if daemon_action("stop", alias):
                log.info(f"  Stopped aux: {alias}")
                stopped.add(alias)
            else:
                log.warning(f"  Failed to stop aux: {alias}")
        return stopped

    def stop_all_main():
        stopped = set()
        for alias, role in get_running_daemons(role_filter="main"):
            if daemon_action("stop", alias):
                log.info(f"  Stopped main: {alias}")
                stopped.add(alias)
            else:
                log.warning(f"  Failed to stop main: {alias}")
        return stopped

    while True:
        try:
            free_gb = get_vram()
            if free_gb is None:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    log.warning("nvidia-smi failed 5 times, exiting")
                    sys.exit(1)
                time.sleep(args.interval)
                continue
            consecutive_failures = 0

            current_tier = 0
            if free_gb < args.tier3:
                current_tier = 3
            elif free_gb < args.tier2:
                current_tier = 2
            elif free_gb < args.tier1:
                current_tier = 1

            # SCALE DOWN — only when moving to a higher (worse) tier
            if current_tier > running_tier:
                log.warning(f"Scaling DOWN: tier {running_tier} → {current_tier} ({free_gb:.1f}GB free)")

                if current_tier >= 1 and running_tier < 1:
                    # Stop all aux daemons
                    stopped_main.update(stop_all_aux())

                if current_tier >= 2 and running_tier < 2:
                    # Stop all main daemons
                    stopped_main.update(stop_all_main())
                    # Switch Hermes to API fallback
                    if not is_hermes_on_api(config_path):
                        log.info(f"Switching Hermes to API: {fallback_main}")
                        set_hermes_model(config_path, fallback_main, fallback_url, fallback_provider)
                    else:
                        log.info(f"Hermes already on API")

                if current_tier >= 3 and running_tier < 3:
                    # Critical — make double-sure everything local is dead
                    for alias, _ in get_running_daemons():
                        daemon_action("stop", alias)
                        stopped_main.add(alias)
                    log.warning("CRITICAL VRAM — all local models stopped")

                running_tier = current_tier

            # SCALE UP — only when VRAM recovers enough
            elif current_tier < running_tier:
                log.info(f"Scaling UP: tier {running_tier} → {current_tier} ({free_gb:.1f}GB free)")

                if running_tier >= 2 and current_tier < 2:
                    # Switch Hermes back to local
                    if free_gb > args.restore2:
                        if is_hermes_on_api(config_path):
                            log.info(f"Switching Hermes back to local: {local_model}")
                            set_hermes_model(config_path, local_model, local_url, "localhost")
                        # Restart main daemons
                        for alias in list(stopped_main):
                            if daemon_action("start", alias):
                                log.info(f"  Restarted main: {alias}")
                                stopped_main.discard(alias)

                if running_tier >= 1 and current_tier < 1:
                    # Restart aux only if we have enough headroom
                    if free_gb > args.restore1:
                        for alias in list(stopped_main):
                            role = get_catalog_role(alias)
                            if role == "aux" and daemon_action("start", alias):
                                log.info(f"  Restarted aux: {alias}")
                                stopped_main.discard(alias)

                running_tier = current_tier

            # Status log every 10 cycles (~5 min)
            if int(time.time()) % (args.interval * 10) < args.interval:
                on_api = is_hermes_on_api(config_path)
                log.info(f"Status: tier={running_tier}, free={free_gb:.1f}GB, "
                         f"on_api={on_api}, stopped={len(stopped_main)}")

        except KeyboardInterrupt:
            log.info("Shutting down — restoring everything")
            for alias in list(stopped_main):
                daemon_action("start", alias)
            if is_hermes_on_api(config_path):
                log.info(f"Reverting Hermes to local: {local_model}")
                set_hermes_model(config_path, local_model, local_url, "localhost")
            sys.exit(0)
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
