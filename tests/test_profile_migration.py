from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.migration import migrate_registry
from turbofit_runtime.profile_io import load_json_profile


def legacy_registry(tmp_path: Path, *, mode: str = "dedicated") -> Path:
    evidence = tmp_path / "evidence.md"
    evidence.write_text("verified benchmark evidence\n")
    components = [
        {
            "role": "main",
            "kind": "process",
            "gpu": "1",
            "port": 11605,
            "command": ["/home/test/llama-server", "-m", "/models/main.gguf"],
        }
    ]
    if mode == "dedicated":
        components.append(
            {
                "role": "aux",
                "kind": "process",
                "gpu": "0",
                "port": 11607,
                "command": ["/home/test/llama-server", "-m", "/models/aux.gguf"],
            }
        )
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {
            "tested-pair-128k": {
                "description": "tested pair",
                "context": 131072,
                "evidence": str(evidence),
                "expected": {
                    "main_alias": "main-model",
                    "aux_alias": "aux-model" if mode == "dedicated" else "auto:main-model",
                    "aux_mode": mode,
                },
                "metrics": {
                    "main_tps": 40,
                    "aux_tps": 80,
                    "gpu_peak_mb": {"0": 12000, "1": 21000},
                },
                "components": components,
            }
        },
    }))
    return path


def test_migration_creates_valid_portable_profile_and_content_addressed_manifests(
    tmp_path: Path,
) -> None:
    registry = legacy_registry(tmp_path)
    output = tmp_path / "runtime-profiles"
    evidence_index = output / "evidence-index.json"

    result = migrate_registry(registry, output, evidence_index)

    assert result.migrated == 1
    profile_path = output / "migrated" / "tested-pair-128k.json"
    profile = load_json_profile(profile_path)
    main_manifest = profile.rungs[0].main_manifest
    aux_manifest = profile.rungs[0].aux_manifest
    assert main_manifest is not None and main_manifest.startswith("sha256:")
    assert aux_manifest is not None and aux_manifest.startswith("sha256:")
    assert profile.rungs[-1].aux_mode.value == "api"
    for digest in (main_manifest, aux_manifest):
        assert (output / "manifests" / f"{digest.removeprefix('sha256:')}.json").is_file()


def test_portable_outputs_remove_paths_ports_and_gpu_indices(tmp_path: Path) -> None:
    registry = legacy_registry(tmp_path)
    output = tmp_path / "runtime-profiles"

    migrate_registry(registry, output, output / "evidence-index.json")

    portable = (output / "migrated" / "tested-pair-128k.json").read_text()
    manifests = "".join(path.read_text() for path in (output / "manifests").glob("*.json"))
    combined = portable + manifests
    assert "/home/" not in combined
    assert "/models/" not in combined
    assert '"gpu"' not in combined
    assert '"port"' not in combined
    assert '"command"' not in combined
    assert '"mounts"' not in combined


def test_legacy_local_resolution_and_evidence_backlink_are_retained(tmp_path: Path) -> None:
    registry = legacy_registry(tmp_path)
    original = json.loads(registry.read_text())
    output = tmp_path / "runtime-profiles"
    evidence_index = output / "evidence-index.json"

    migrate_registry(registry, output, evidence_index)

    updated = json.loads(registry.read_text())
    old_profile = original["profiles"]["tested-pair-128k"]
    new_profile = updated["profiles"]["tested-pair-128k"]
    assert new_profile["components"] == old_profile["components"]
    assert new_profile["evidence"] == old_profile["evidence"]
    assert new_profile["portable_profile"] == "runtime-profiles/migrated/tested-pair-128k.json"
    assert new_profile["evidence_identity"].startswith("sha256:")
    index = json.loads(evidence_index.read_text())
    assert index[new_profile["evidence_identity"]]["source"] == old_profile["evidence"]


def test_shared_main_migration_has_no_aux_manifest(tmp_path: Path) -> None:
    registry = legacy_registry(tmp_path, mode="shared-main")
    output = tmp_path / "runtime-profiles"

    migrate_registry(registry, output, output / "evidence-index.json")

    profile = load_json_profile(output / "migrated" / "tested-pair-128k.json")
    assert profile.rungs[0].aux_manifest is None
    assert profile.rungs[0].aux_mode.value == "shared-main"


def test_migration_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    registry = legacy_registry(tmp_path)
    output = tmp_path / "runtime-profiles"
    evidence_index = output / "evidence-index.json"

    migrate_registry(registry, output, evidence_index)
    first_profile = (output / "migrated" / "tested-pair-128k.json").read_bytes()
    first_registry = registry.read_bytes()
    first_index = evidence_index.read_bytes()
    migrate_registry(registry, output, evidence_index)

    assert (output / "migrated" / "tested-pair-128k.json").read_bytes() == first_profile
    assert registry.read_bytes() == first_registry
    assert evidence_index.read_bytes() == first_index
