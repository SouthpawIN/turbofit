#!/usr/bin/env python3
"""
turbofit-resolver — dynamic endpoint resolver for nginx.

Reads the current state from:
  - nvidia-smi (VRAM per GPU)
  - turbofit catalog (~/.config/turbofit/models.yaml)
  - scaling watcher state (via daemon health checks)
  - Hermes config (which model the watcher has set)

Outputs the active main and aux endpoints as JSON.
Used by nginx to dynamically proxy to whatever model is currently running.

Usage:
  python3 turbofit-resolver.py           # JSON output
  python3 turbofit-resolver.py --main    # just the main base_url
  python3 turbofit-resolver.py --aux     # just the aux base_url
  python3 turbofit-resolver.py --status  # human-readable status
"""

import subprocess
import json
import os
import sys
import argparse
from pathlib import Path

HOME = os.path.expanduser("~")
CATALOG = os.environ.get("TURBOFIT_CATALOG", f"{HOME}/.config/turbofit/models.yaml")
PREFS = os.environ.get("TURBOFIT_PREFS", f"{HOME}/.config/turbofit/preferences.yaml")
HERMES_HOME = os.environ.get("HERMES_HOME", f"{HOME}/.hermes")

FALLBACK_MODEL = "z-ai/glm-5.2"
FALLBACK_URL = "https://inference-api.nousresearch.com/v1"
FALLBACK_PROVIDER = "nous"


def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_prefs():
    return load_yaml(PREFS)


def load_catalog():
    return load_yaml(CATALOG)


def get_per_gpu_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "id": int(parts[0]),
                        "free_gb": int(parts[1]) / 1024,
                        "total_gb": int(parts[2]) / 1024,
                    })
        return gpus
    except:
        return []


def check_port_responding(port):
    try:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "2",
             f"http://127.0.0.1:{port}/v1/models"],
            capture_output=True, text=True, timeout=5
        )
        return "data" in result.stdout
    except:
        return False


def get_active_main():
    catalog = load_catalog()
    prefs = load_prefs()
    models = catalog.get("models", {})

    local = prefs.get("api_fallback", {}).get("local", {})
    preferred_main = local.get("main", "darwin-28b-reason")
    local_url = local.get("base_url", "http://127.0.0.1:11500/v1")

    swap_ladder = [
        preferred_main,
        "darwin-28b-coder",
        "prism-eagle-27b",
        "darwin-apex-36b",
        "carnice-v2-27b",
    ]

    for alias in swap_ladder:
        if alias not in models:
            continue
        model = models[alias]
        port = model.get("port", 0)
        if port and check_port_responding(port):
            return {
                "type": "local",
                "alias": alias,
                "model_id": model.get("aliases", [alias])[0] if model.get("aliases") else alias,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "port": port,
                "gpu": model.get("gpu", 0),
                "size_gb": model.get("size_gb", 0),
                "ctx": model.get("ctx", 262144),
            }

    # Check if Hermes is on API fallback
    senter_cfg_path = f"{HERMES_HOME}/profiles/senter/config.yaml"
    senter_cfg = load_yaml(senter_cfg_path)
    senter_model = senter_cfg.get("model", {})
    senter_url = senter_model.get("base_url", "")

    if "inference-api" in senter_url or "127.0.0.1" not in senter_url:
        return {
            "type": "api",
            "alias": "api-fallback",
            "model_id": senter_model.get("default", FALLBACK_MODEL),
            "base_url": senter_url or FALLBACK_URL,
            "port": 0,
            "gpu": -1,
            "size_gb": 0,
            "ctx": 0,
        }

    return {
        "type": "local",
        "alias": preferred_main,
        "model_id": preferred_main,
        "base_url": local_url,
        "port": 11500,
        "gpu": 0,
        "size_gb": 0,
        "ctx": 0,
        "status": "not_responding",
    }


def get_active_aux():
    catalog = load_catalog()
    models = catalog.get("models", {})

    for alias, model in models.items():
        if model.get("role") != "aux":
            continue
        port = model.get("port", 0)
        if port and check_port_responding(port):
            return {
                "type": "local",
                "alias": alias,
                "model_id": model.get("aliases", [alias])[0] if model.get("aliases") else alias,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "port": port,
                "gpu": model.get("gpu", 1),
                "size_gb": model.get("size_gb", 0),
            }

    return {
        "type": "none",
        "alias": "none",
        "model_id": "",
        "base_url": "",
        "port": 0,
        "gpu": -1,
        "size_gb": 0,
    }


def get_status():
    main = get_active_main()
    aux = get_active_aux()
    gpus = get_per_gpu_vram()

    lines = [
        "turbofit endpoint status",
        "=" * 50,
    ]

    for gpu in gpus:
        lines.append(f"GPU {gpu['id']}: {gpu['free_gb']:.1f}GB free / {gpu['total_gb']:.1f}GB total")

    lines.append("")
    lines.append(f"Main: {main['alias']} ({main['type']})")
    lines.append(f"  URL: {main['base_url']}")
    if main.get("port"):
        lines.append(f"  Port: {main['port']}")
    if main.get("size_gb"):
        lines.append(f"  Size: {main['size_gb']}GB")
    if main.get("ctx"):
        lines.append(f"  Context: {main['ctx']//1024}K")
    if main.get("status"):
        lines.append(f"  WARNING: {main['status']}")

    lines.append("")
    lines.append(f"Aux: {aux['alias']} ({aux['type']})")
    if aux.get("base_url"):
        lines.append(f"  URL: {aux['base_url']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="turbofit endpoint resolver")
    parser.add_argument("--main", action="store_true", help="Output only the main base_url")
    parser.add_argument("--aux", action="store_true", help="Output only the aux base_url")
    parser.add_argument("--status", action="store_true", help="Human-readable status")
    parser.add_argument("--json", action="store_true", help="Full JSON output (default)")
    args = parser.parse_args()

    if args.status:
        print(get_status())
    elif args.main:
        main_info = get_active_main()
        print(main_info["base_url"])
    elif args.aux:
        aux_info = get_active_aux()
        print(aux_info.get("base_url", ""))
    else:
        result = {
            "main": get_active_main(),
            "aux": get_active_aux(),
            "gpus": get_per_gpu_vram(),
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
