"""Generate one exact, evidence-backed manual Turbofile for the current machine."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from .hardware import HardwareFingerprint
from .recipes import ResolvedRecipe
from .runtime_profile import Turbofile

MANUAL_PROFILE = "manual-profile.yaml"
MANUAL_RESOLUTIONS = "manual-runtime-resolutions.json"
MANUAL_REQUIREMENTS = "manual-rung-requirements.json"


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def manual_profile_paths(config_dir: str | Path) -> tuple[Path, Path, Path]:
    root = Path(config_dir)
    return root / MANUAL_PROFILE, root / MANUAL_RESOLUTIONS, root / MANUAL_REQUIREMENTS


def build_manual_profile_payload(
    *,
    profile_id: str,
    profile_entry: Mapping[str, Any],
    recipe: ResolvedRecipe,
    hardware: HardwareFingerprint,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build portable profile, native resolution, and requirement sidecars."""
    metrics = profile_entry.get("metrics") if isinstance(profile_entry.get("metrics"), Mapping) else {}
    raw_gpu = metrics.get("gpu_peak_mb") if isinstance(metrics, Mapping) else {}
    gpu_peak = {
        int(index): int(value)
        for index, value in (raw_gpu.items() if isinstance(raw_gpu, Mapping) else ())
        if int(value) > 0
    }
    if not gpu_peak:
        raise ValueError(f"manual combination lacks measured per-GPU residency: {profile_id}")
    if not hardware.devices:
        required = [sum(gpu_peak.values())]
        min_devices = 1
        per_device_min_mb = required[0]
        topology = "shared-memory"
    else:
        max_index = max(gpu_peak)
        if max_index >= len(hardware.devices):
            raise ValueError(f"manual combination requires unavailable GPU {max_index}")
        required = [gpu_peak.get(index, 0) for index in range(len(hardware.devices))]
        min_devices = max_index + 1
        per_device_min_mb = max(gpu_peak.values())
        counts = Counter(round(device.memory_total_mb / 1024) for device in hardware.devices)
        topology = "+".join(f"{count}x{memory_gb}" for memory_gb, count in sorted(counts.items()))
    context = int(profile_entry.get("context") or 0)
    if context <= 0:
        raise ValueError("manual combination context must be positive")
    evidence = str(profile_entry.get("production_recipe_sha256") or "")
    if not evidence.startswith("sha256:"):
        evidence = _sha({"profile": profile_id, "recipe": recipe.row_id, "context": context})

    components = {component.role: component for component in recipe.components}
    main = components.get("main")
    if main is None:
        raise ValueError("manual combination has no main component")
    dedicated = recipe.aux_mode == "dedicated"
    aux = components.get("aux")
    if dedicated and aux is None:
        raise ValueError("dedicated manual combination has no auxiliary component")

    def manifest(component: Any) -> str:
        return _sha({
            "family": component.family,
            "alias": component.alias,
            "gpu": component.gpu,
            "port": component.port,
            "command": list(component.command),
        })

    local_rung: dict[str, Any] = {
        "id": "manual-exact",
        "context": context,
        "aux_mode": recipe.aux_mode,
        "evidence": evidence,
        "main_manifest": manifest(main),
    }
    if dedicated:
        local_rung["aux_manifest"] = manifest(aux)
    api_evidence = _sha({"profile": profile_id, "fallback": "api:auto"})
    profile = {
        "schema": "turbofit.runtime/v1",
        "id": profile_id,
        "revision": 1,
        "hardware": {
            "class_vram_gb": max(1, math.ceil(sum(required) / 1024)),
            "min_devices": min_devices,
            "total_vram_gb": max(1, max(sum(required), min_devices * per_device_min_mb) / 1024),
            "per_device_min_gb": max(1, per_device_min_mb / 1024),
            "accelerator": "llama.cpp-local",
            "topology": topology,
        },
        "policy": {
            "recommendation": "manual-evidence-backed",
            "external_gpu_priority": "absolute",
            "contraction_dwell_s": 5,
            "expansion_dwell_s": 30,
            "expansion_margin_gb_per_card": 0.5,
            "cooldown_s": 30,
            "flap_failure_limit": 3,
            "flap_window_s": 300,
        },
        "roles": {"main": "active:main", "auxiliary": "active:aux", "fallback": "active:main"},
        "rungs": [
            local_rung,
            {
                "id": "api",
                "context": context,
                "aux_mode": "api",
                "evidence": api_evidence,
                "main_api_policy": "api:auto",
                "aux_api_policy": "api:auto",
            },
        ],
    }
    Turbofile.from_mapping(profile)

    roles: dict[str, Any] = {}
    for role, component in components.items():
        if role not in {"main", "aux"}:
            continue
        index = int(component.gpu) if str(component.gpu).isdigit() else 0
        roles[role] = {
            "model_tag": component.alias,
            "expected_vram_mb": max(1, gpu_peak.get(index, 1)),
            "family": component.family,
            "gpu": component.gpu,
            "port": component.port,
        }
    resolutions = {
        "schema": "turbofit.runtime-resolutions/v1",
        "profiles": {profile_id: {"manual-exact": roles}},
    }
    requirements = {
        "schema": "turbofit.rung-requirements/v1",
        "profiles": {
            profile_id: [
                {"rung_id": "manual-exact", "evidence": evidence, "required_mb_per_card": required},
                {"rung_id": "api", "evidence": api_evidence, "required_mb_per_card": []},
            ]
        },
    }
    return profile, resolutions, requirements


def write_manual_profile(
    config_dir: str | Path,
    *,
    profile_id: str,
    profile_entry: Mapping[str, Any],
    recipe: ResolvedRecipe,
    hardware: HardwareFingerprint,
) -> tuple[Path, Path, Path]:
    profile, resolutions, requirements = build_manual_profile_payload(
        profile_id=profile_id,
        profile_entry=profile_entry,
        recipe=recipe,
        hardware=hardware,
    )
    profile_path, resolutions_path, requirements_path = manual_profile_paths(config_dir)
    _atomic_text(profile_path, yaml.safe_dump(profile, sort_keys=False))
    _atomic_text(resolutions_path, json.dumps(resolutions, indent=2, sort_keys=True) + "\n")
    _atomic_text(requirements_path, json.dumps(requirements, indent=2, sort_keys=True) + "\n")
    return profile_path, resolutions_path, requirements_path
