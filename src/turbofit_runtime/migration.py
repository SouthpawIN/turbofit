"""Deterministic migration from local runtime records to portable Turbofiles."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_profile import Turbofile


@dataclass(frozen=True)
class MigrationResult:
    migrated: int
    profile_paths: tuple[Path, ...]
    manifest_paths: tuple[Path, ...]


def migrate_registry(
    registry_path: str | Path,
    output_dir: str | Path,
    evidence_index_path: str | Path,
    *,
    hardware_class_vram_gb: int = 24,
    accelerator: str = "nvidia-cuda",
    portable_prefix: str = "runtime-profiles",
) -> MigrationResult:
    registry_target = Path(registry_path)
    output = Path(output_dir)
    evidence_target = Path(evidence_index_path)
    registry = json.loads(registry_target.read_text(encoding="utf-8"))
    profiles = registry.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("legacy registry must contain a profiles mapping")
    if hardware_class_vram_gb <= 0:
        raise ValueError("hardware_class_vram_gb must be positive")

    profile_paths: list[Path] = []
    manifest_paths: set[Path] = set()
    evidence_index: dict[str, dict[str, Any]] = {}
    for profile_id in sorted(profiles):
        legacy = profiles[profile_id]
        if not isinstance(legacy, dict):
            raise ValueError(f"legacy profile must be a mapping: {profile_id}")
        portable, descriptors, evidence_identity = _migrate_one(
            profile_id,
            legacy,
            hardware_class_vram_gb=hardware_class_vram_gb,
            accelerator=accelerator,
        )
        validated = Turbofile.from_mapping(portable)
        profile_path = output / "migrated" / f"{profile_id}.json"
        _atomic_write(profile_path, _json_text(validated.to_mapping()))
        profile_paths.append(profile_path)

        for digest, descriptor in descriptors.items():
            manifest_path = output / "manifests" / f"{digest.removeprefix('sha256:')}.json"
            _atomic_write(manifest_path, _json_text(descriptor))
            manifest_paths.add(manifest_path)

        evidence_source = str(legacy["evidence"])
        entry = evidence_index.setdefault(
            evidence_identity,
            {"source": evidence_source, "profiles": []},
        )
        if entry["source"] != evidence_source:
            raise ValueError(f"evidence digest collision: {evidence_identity}")
        entry["profiles"].append(profile_id)
        legacy["portable_profile"] = (
            f"{portable_prefix.rstrip('/')}/migrated/{profile_id}.json"
        )
        legacy["evidence_identity"] = evidence_identity

    for entry in evidence_index.values():
        entry["profiles"] = sorted(set(entry["profiles"]))
    _atomic_write(evidence_target, _json_text(evidence_index))
    _atomic_write(registry_target, _json_text(registry))
    return MigrationResult(
        migrated=len(profile_paths),
        profile_paths=tuple(profile_paths),
        manifest_paths=tuple(sorted(manifest_paths)),
    )


def _migrate_one(
    profile_id: str,
    legacy: Mapping[str, Any],
    *,
    hardware_class_vram_gb: int,
    accelerator: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    context = _positive_int(legacy.get("context"), f"{profile_id}.context")
    expected = legacy.get("expected") or {}
    if not isinstance(expected, Mapping):
        raise ValueError(f"{profile_id}.expected must be a mapping")
    main_alias = _nonempty(expected.get("main_alias"), f"{profile_id}.main_alias")
    aux_alias = _nonempty(expected.get("aux_alias"), f"{profile_id}.aux_alias")
    aux_mode = str(expected.get("aux_mode") or "")
    if aux_mode not in {"dedicated", "shared-main"}:
        raise ValueError(f"unsupported legacy aux_mode for {profile_id}: {aux_mode}")
    evidence_source = Path(_nonempty(legacy.get("evidence"), f"{profile_id}.evidence"))
    try:
        evidence_bytes = evidence_source.read_bytes()
    except OSError as exc:
        raise ValueError(f"evidence source is unavailable for {profile_id}: {evidence_source}") from exc
    evidence_identity = "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
    metrics = legacy.get("metrics") or {}
    method = str(metrics.get("method") or "legacy-verified") if isinstance(metrics, Mapping) else "legacy-verified"

    descriptors: dict[str, dict[str, Any]] = {}
    main_descriptor = _manifest_descriptor(
        role="main",
        alias=main_alias,
        context=context,
        aux_mode=aux_mode,
        method=method,
    )
    main_identity = _descriptor_identity(main_descriptor)
    descriptors[main_identity] = main_descriptor
    local_rung: dict[str, Any] = {
        "id": f"local-{context}",
        "context": context,
        "aux_mode": aux_mode,
        "evidence": evidence_identity,
        "main_manifest": main_identity,
    }
    min_devices = 1
    if aux_mode == "dedicated":
        aux_descriptor = _manifest_descriptor(
            role="auxiliary",
            alias=aux_alias,
            context=context,
            aux_mode=aux_mode,
            method=method,
        )
        aux_identity = _descriptor_identity(aux_descriptor)
        descriptors[aux_identity] = aux_descriptor
        local_rung["aux_manifest"] = aux_identity
        min_devices = 2

    portable = {
        "schema": "turbofit.runtime/v1",
        "id": profile_id,
        "revision": 1,
        "hardware": {
            "class_vram_gb": hardware_class_vram_gb,
            "min_devices": min_devices,
            "total_vram_gb": min_devices * hardware_class_vram_gb,
            "per_device_min_gb": hardware_class_vram_gb,
            "accelerator": accelerator,
            "topology": f"{min_devices}x{hardware_class_vram_gb}",
        },
        "policy": {
            "recommendation": "quality-first",
            "external_gpu_priority": "absolute",
            "contraction_dwell_s": 5,
            "expansion_dwell_s": 120,
            "expansion_margin_gb_per_card": 2,
            "cooldown_s": 30,
            "flap_failure_limit": 3,
            "flap_window_s": 300,
        },
        "roles": {
            "main": "active:main",
            "auxiliary": "active:aux",
            "fallback": "api:auto",
        },
        "rungs": [
            local_rung,
            {
                "id": "api",
                "context": context,
                "aux_mode": "api",
                "evidence": evidence_identity,
                "main_api_policy": "api:auto",
                "aux_api_policy": "api:auto",
            },
        ],
    }
    return portable, descriptors, evidence_identity


def _manifest_descriptor(
    *, role: str, alias: str, context: int, aux_mode: str, method: str
) -> dict[str, Any]:
    return {
        "schema": "turbofit.manifest-ref/v1",
        "role": role,
        "model_alias": alias,
        "context": context,
        "aux_mode": aux_mode,
        "method": method,
    }


def _descriptor_identity(descriptor: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _atomic_write(path: Path, content: str) -> None:
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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()
