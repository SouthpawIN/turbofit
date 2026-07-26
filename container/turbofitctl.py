#!/usr/bin/env python3
"""User-facing local controller for the Turbofit container stack."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hardware_probe import probe


HERE = Path(__file__).resolve().parent
MANIFEST = Path(os.getenv("TURBOFIT_MANIFEST", HERE / "manifest.json"))
STATE = Path(os.getenv("TURBOFIT_STATE", Path.home() / ".config/turbofit/container-state.json"))
MODEL_ROOT = Path(os.getenv("TURBOFIT_MODEL_ROOT", Path.home() / ".cache/turbofit/models"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def gpu_state() -> list[dict]:
    return probe()["gpus"]


def recommendation(manifest: dict) -> dict:
    hardware = probe()
    gpus = hardware["gpus"]
    free = hardware["free_vram_gb"]
    if not gpus:
        ram = hardware["system_ram_gb"]
        preferred = "ternary-bonsai-27b-dspark" if ram >= 16 else "bonsai-27b-1bit" if ram >= 6 else None
        model = next((m for m in manifest["models"] if m["alias"] == preferred), None)
        return {"hardware": hardware, "mode": "local-cpu" if model else "api", "profile": model.get("profile") if model else None, "model": model, "fallback": manifest["fallback"]}
    chosen = None
    for profile in sorted(manifest["profiles"], key=lambda p: p.get("minimum_free_vram_gb", p.get("minimum_memory_gb", 0))):
        if len(gpus) >= profile.get("minimum_gpu_count", 1) and free >= profile.get("minimum_free_vram_gb", profile.get("minimum_memory_gb", 0)):
            chosen = profile
    if chosen is None:
        chosen = manifest["profiles"][0]
    model = next(m for m in manifest["models"] if m["alias"] == chosen["preferred"])
    return {"hardware": hardware, "mode": "local", "profile": chosen, "model": model, "fallback": manifest["fallback"]}


def cmd_use(alias: str, manifest: dict) -> int:
    model = next((m for m in manifest["models"] if m["alias"] == alias), None)
    if not model:
        print(json.dumps({"error": f"unknown model: {alias}"})); return 2
    if not model.get("port") or not model.get("model_subdir"):
        print(json.dumps({"error": "model has no local container settings", "model": model})); return 2
    image = model["image"]
    check = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if check.returncode:
        pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        if pull.returncode:
            print(json.dumps({"error": "container image unavailable", "image": image, "detail": pull.stderr.strip()})); return 3
    if not gpu_state():
        print(json.dumps({"error": "no visible NVIDIA GPU; use an API provider"})); return 4
    probe = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.6.3-base-ubuntu24.04", "nvidia-smi", "-L"], capture_output=True, text=True)
    if probe.returncode:
        bootstrap = subprocess.run([str(HERE.parent / "scripts/ensure-gpu-runtime.sh")], capture_output=True, text=True)
        if bootstrap.returncode:
            print(json.dumps({"error": "Docker GPU runtime unavailable", "detail": (bootstrap.stderr or bootstrap.stdout).strip()})); return 4
    name = "turbofit-model-" + alias.replace(".", "-")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    model_dir = MODEL_ROOT / model["model_subdir"]
    if not model_dir.exists():
        print(json.dumps({"error": "model files are not installed", "model_dir": str(model_dir), "hf_repo": model.get("hf_repo")})); return 5
    run = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--network", "host", "-e", f"PORT={model['port']}", "-v", f"{model_dir}:/models:ro", image]
    result = subprocess.run(run, capture_output=True, text=True)
    if result.returncode:
        print(json.dumps({"error": "container start failed", "detail": result.stderr.strip()})); return result.returncode
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"main": {"alias": alias, "model": alias, "base_url": f"http://127.0.0.1:{model['port']}"}, "aux": None, "mode": "local", "last_transition": "use"}, indent=2) + "\n")
    print(json.dumps({"started": True, "container": name, "state": str(STATE), "model": model}, indent=2)); return 0


def main(argv: list[str]) -> int:
    manifest = read_json(MANIFEST)
    command = argv[1] if len(argv) > 1 else "recommend"
    if command == "recommend": print(json.dumps(recommendation(manifest), indent=2)); return 0
    if command == "models": print(json.dumps(manifest["models"], indent=2)); return 0
    if command == "use" and len(argv) == 3: return cmd_use(argv[2], manifest)
    if command == "install-provider":
        return subprocess.call([str(HERE / "install-hermes-provider.sh")])
    print("usage: turbofitctl [recommend|models|use ALIAS|install-provider]", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))