#!/usr/bin/env python3
"""
turbofit-vram-watcher - monitors GPU VRAM and auto-stops aux models when pressure increases.

When you start loading a new model on the GPU (comfyui, another llama-server, whatever),
the watcher detects VRAM pressure and gracefully stops the aux daemon (Carnice) to free
up space. When VRAM frees up again, it restarts the aux daemon.

This solves the "I keep having to stop models manually" problem.

Usage:
  turbofit-vram-watcher                    # foreground, 30s poll interval
  turbofit-vram-watcher --interval 15     # 15s poll

Environment:
  TURBOFIT_FREE_THRESHOLD  - GB below which aux is stopped (default: 8)
  TURBOFIT_RESTORE_THRESHOLD - GB above which aux is restarted (default: 16)
  TURBOFIT_CATALOG         - path to models.yaml (default: ~/.config/turbofit/models.yaml)
"""

import subprocess
import json
import time
import os
import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watcher] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("watcher")

CATALOG = os.environ.get("TURBOFIT_CATALOG", os.path.expanduser("~/.config/turbofit/models.yaml"))
FREE_THRESHOLD = float(os.environ.get("TURBOFIT_FREE_THRESHOLD", "8"))
RESTORE_THRESHOLD = float(os.environ.get("TURBOFIT_RESTORE_THRESHOLD", "16"))


def get_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        free_mib = sum(int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip())
        return free_mib / 1024
    except Exception as e:
        log.error(f"VRAM probe failed: {e}")
        return None


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


def get_aux_daemons():
    aux_aliases = []
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--no-legend"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "turbofit-" in line and ".service" in line:
                parts = line.split()
                for p in parts:
                    if p.startswith("turbofit-") and p.endswith(".service"):
                        alias = p.replace("turbofit-", "").replace(".service", "")
                        role = get_catalog_role(alias)
                        if role == "aux":
                            aux_aliases.append(alias)
                        break
    except Exception as e:
        log.error(f"Failed to list systemd services: {e}")
    return aux_aliases


def stop_daemon(alias):
    log.info(f"Stopping aux daemon: {alias}")
    result = subprocess.run(
        ["systemctl", "--user", "stop", f"turbofit-{alias}.service"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log.info(f"Stopped {alias} - VRAM freed")
    else:
        log.error(f"Failed to stop {alias}: {result.stderr}")


def start_daemon(alias):
    log.info(f"Starting aux daemon: {alias}")
    result = subprocess.run(
        ["systemctl", "--user", "start", f"turbofit-{alias}.service"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log.info(f"Started {alias} - proxy will wake backend on first request")
    else:
        log.error(f"Failed to start {alias}: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="turbofit VRAM watcher")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("--threshold", type=float, default=FREE_THRESHOLD, help=f"Stop aux below this GB (default: {FREE_THRESHOLD})")
    parser.add_argument("--restore", type=float, default=RESTORE_THRESHOLD, help=f"Restart aux above this GB (default: {RESTORE_THRESHOLD})")
    args = parser.parse_args()

    log.info(f"turbofit VRAM watcher started (poll={args.interval}s, stop<{args.threshold}GB, restore>{args.restore}GB)")

    aux_stopped_by_us = set()
    consecutive_failures = 0

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

            running_aux = get_aux_daemons()

            if free_gb < args.threshold:
                for alias in running_aux:
                    if alias not in aux_stopped_by_us:
                        log.warning(f"VRAM pressure: {free_gb:.1f}GB free < {args.threshold}GB threshold")
                        stop_daemon(alias)
                        aux_stopped_by_us.add(alias)

            elif free_gb > args.restore and aux_stopped_by_us:
                log.info(f"VRAM recovered: {free_gb:.1f}GB free > {args.restore}GB threshold")
                for alias in list(aux_stopped_by_us):
                    start_daemon(alias)
                    aux_stopped_by_us.discard(alias)

        except KeyboardInterrupt:
            log.info("Shutting down watcher")
            for alias in aux_stopped_by_us:
                log.info(f"Restoring {alias} before exit")
                start_daemon(alias)
            sys.exit(0)
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
