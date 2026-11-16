"""Automatic selection must never pick a configuration it cannot launch.

The live failure: a fresh ``selection.json`` in ``mode: auto`` chose a profile by
tier fit alone, then failed every tick with ``model artifact is missing``. These
tests pin the three properties that make that impossible:

* a present artifact wins over a higher-ranked absent one;
* a sharded model missing siblings is absent, not partially usable;
* nothing present is ``waiting-download`` with a byte count, not a crash.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_runtime_profile import valid_mapping
from test_selection import hardware
from turbofit_runtime.artifacts import inspect_artifact
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.routes import load_runtime_resolutions
from turbofit_runtime.runtime_profile import Turbofile
from turbofit_runtime.runtime_service import RuntimeService
from turbofit_runtime.selection import (
    ArtifactPresence,
    ProfileCatalog,
    SelectionMode,
    WaitingForDownload,
)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def local_profile(class_gb: int, topology: str, rungs: int = 2) -> Turbofile:
    """A local-only profile: no API rung to mask an absent artifact."""
    data = valid_mapping()
    data["id"] = f"hardware-{class_gb}gb"
    devices = sum(int(part.split("x", 1)[0]) for part in topology.split("+"))
    data["hardware"].update(
        class_vram_gb=class_gb,
        min_devices=devices,
        total_vram_gb=class_gb,
        per_device_min_gb=min(int(part.split("x", 1)[1]) for part in topology.split("+")),
        topology=topology,
    )
    first = data["rungs"][0]
    data["rungs"] = [
        {**first, "id": f"rung-{class_gb}-{index}", "context": 65536 * (index + 1)}
        for index in range(rungs)
    ]
    return Turbofile.from_mapping(data)


def present(model: str = "Bonsai 27B") -> ArtifactPresence:
    return ArtifactPresence(present=True, model=model)


def absent(model: str, missing_bytes: int) -> ArtifactPresence:
    return ArtifactPresence(present=False, missing_bytes=missing_bytes, model=model)


def test_auto_prefers_a_present_artifact_over_a_higher_ranked_absent_one() -> None:
    included = local_profile(48, "2x24")
    fallback = local_profile(24, "1x24")
    catalog = ProfileCatalog((fallback, included))
    gaps = {
        ("hardware-48gb", 0): absent("Cyber-Tiel Coder 35B-A3B", 22 * GIB),
        ("hardware-48gb", 1): absent("Cyber-Tiel Coder 35B-A3B", 22 * GIB),
        ("hardware-24gb", 0): present("Qwen 3.8 27B GSQ-RCO"),
        ("hardware-24gb", 1): present("Qwen 3.8 27B GSQ-RCO"),
    }

    def probe(profile: Turbofile, index: int) -> ArtifactPresence:
        return gaps[(profile.id, index)]

    choice = catalog.select(hardware(24576, 24576), requested="auto", probe=probe)

    assert choice.mode is SelectionMode.AUTO
    assert choice.profile.id == "hardware-24gb"
    assert choice.initial_rung_index == 1
    assert choice.recommended_profile_id == "hardware-48gb"
    # The reason must name both: what it chose, and what it would have chosen.
    assert "Qwen 3.8 27B GSQ-RCO" in choice.reason
    assert "Cyber-Tiel Coder 35B-A3B" in choice.reason
    assert "22.0 GiB short" in choice.reason
    assert choice.missing_bytes == 22 * GIB


def test_auto_starts_at_the_highest_present_rung_of_the_best_present_profile() -> None:
    """An absent top rung must not disqualify a profile that has a lower one."""
    profile = local_profile(48, "2x24", rungs=3)
    catalog = ProfileCatalog((profile,))

    def probe(_profile: Turbofile, index: int) -> ArtifactPresence:
        return present("Qwen 3.8 27B") if index == 1 else absent("Cyber-Tiel", 10 * GIB)

    choice = catalog.select(hardware(24576, 24576), requested="auto", probe=probe)

    assert (choice.profile.id, choice.initial_rung_index) == ("hardware-48gb", 1)
    assert "best present artifact at this tier" in choice.reason


def test_sharded_model_missing_siblings_is_absent(tmp_path: Path) -> None:
    rows = [
        {
            "destination": f"Big-Model/Big-Model-{index:05d}-of-00004.gguf",
            "path": f"Big-Model-{index:05d}-of-00004.gguf",
            "repo_id": "org/big",
            "size_bytes": 2 * GIB,
        }
        for index in range(1, 5)
    ]
    directory = tmp_path / "Big-Model"
    directory.mkdir()
    for index in (1, 2):
        with open(directory / f"Big-Model-{index:05d}-of-00004.gguf", "wb") as handle:
            handle.truncate(2 * GIB)  # sparse: a real size without real bytes

    state = inspect_artifact(
        directory / "Big-Model-00001-of-00004.gguf", model_root=tmp_path, rows=rows
    )

    assert state.present is False
    assert state.shortfall == 4 * GIB
    assert "2 of 4 shards" in state.reason

    for index in (3, 4):
        with open(directory / f"Big-Model-{index:05d}-of-00004.gguf", "wb") as handle:
            handle.truncate(2 * GIB)
    complete = inspect_artifact(
        directory / "Big-Model-00001-of-00004.gguf", model_root=tmp_path, rows=rows
    )
    assert complete.present is True


def test_symlinked_model_root_still_matches_manifest_rows(tmp_path: Path) -> None:
    """Model roots are often symlinks to another disk; presence must survive it."""
    real = tmp_path / "disk"
    (real / "Bonsai-27B").mkdir(parents=True)
    with open(real / "Bonsai-27B" / "Bonsai-27B-Q1_0.gguf", "wb") as handle:
        handle.truncate(2 * GIB)
    root = tmp_path / "gguf"
    root.symlink_to(real, target_is_directory=True)
    rows = [
        {
            "destination": "Bonsai-27B/Bonsai-27B-Q1_0.gguf",
            "repo_id": "org/bonsai",
            "size_bytes": 2 * GIB,
        }
    ]

    state = inspect_artifact(root / "Bonsai-27B" / "Bonsai-27B-Q1_0.gguf", model_root=root, rows=rows)

    assert state.present is True
    assert state.expected_bytes == 2 * GIB


def test_sharded_incomplete_rung_counts_as_absent_for_selection(tmp_path: Path) -> None:
    directory = tmp_path / "Big-Model"
    directory.mkdir()
    with open(directory / "Big-Model-00001-of-00004.gguf", "wb") as handle:
        handle.truncate(4096)
    rows = [
        {
            "destination": f"Big-Model/Big-Model-{index:05d}-of-00004.gguf",
            "repo_id": "org/big",
            "size_bytes": 2 * GIB,
        }
        for index in range(1, 5)
    ]
    missing = inspect_artifact(directory / "Big-Model-00001-of-00004.gguf", model_root=tmp_path, rows=rows)

    catalog = ProfileCatalog((local_profile(48, "2x24"),))

    def probe(_profile: Turbofile, _index: int) -> ArtifactPresence:
        return ArtifactPresence(
            present=missing.present,
            missing_bytes=missing.shortfall,
            model="Big Model",
        )

    with pytest.raises(WaitingForDownload) as caught:
        catalog.select(hardware(24576, 24576), requested="auto", probe=probe)
    assert caught.value.missing_bytes == 8 * GIB - 4096


def test_nothing_present_yields_waiting_download_with_bytes_not_a_crash() -> None:
    catalog = ProfileCatalog((local_profile(48, "2x24"),))
    short = 41 * GIB + int(0.2 * GIB)

    def probe(_profile: Turbofile, _index: int) -> ArtifactPresence:
        return absent("MiniMax-M3", short)

    with pytest.raises(WaitingForDownload) as caught:
        catalog.select(hardware(24576, 24576), requested="auto", probe=probe)

    waiting = caught.value
    assert waiting.missing_bytes == short
    assert waiting.model == "MiniMax-M3"
    assert waiting.recommended_profile_id == "hardware-48gb"
    assert "41.2 GiB short" in str(waiting)
    assert "waiting-download" in str(waiting)


def test_service_persists_and_reports_the_artifact_aware_choice(tmp_path: Path) -> None:
    profiles = ProfileCatalog(
        load_yaml_profile(ROOT / "runtime-profiles" / f"{size}gb.yaml")
        for size in (16, 24, 48)
    )

    class Backend:
        def __init__(self) -> None:
            self.events: list[object] = []

        def reset_managed(self): self.events.append("reset")
        def activate_local(self, rung_id): self.events.append(("local", rung_id))
        def activate_api(self, main, aux): self.events.append(("api", main, aux))
        def route_aux_to_main(self): self.events.append("shared")
        def route_aux_dedicated(self): self.events.append("dedicated")
        def verify_rung(self, rung_id): return True
        def publish_routes(self, state): self.events.append(("publish", state.rung_index))
        def restore(self, state): self.events.append(("restore", state.rung_index))
        def verify_restore(self, state): return True
        def block_aux_admission(self): self.events.append("block")

    backends: list[Backend] = []

    def factory(profile, state):
        del profile, state
        item = Backend()
        backends.append(item)
        return item

    def probe(profile: Turbofile, _index: int) -> ArtifactPresence:
        if profile.id == "hardware-48gb":
            return absent("Cyber-Tiel Coder 35B-A3B", 20 * GIB)
        return present("Qwen 3.8 27B GSQ-RCO")

    runtime = RuntimeService(
        catalog=profiles,
        resolutions=load_runtime_resolutions(
            ROOT / "runtime-profiles" / "runtime-resolutions.json"
        ),
        requirements_path=ROOT / "runtime-profiles" / "rung-requirements.json",
        controller_state_path=tmp_path / "controller.json",
        route_state_path=tmp_path / "routes.json",
        manager_port=11401,
        backend_factory=factory,
        artifact_probe=probe,
    )
    selection_path = tmp_path / "selection.json"

    controller = runtime.synchronize(selection_path, hardware(24576, 24576))

    assert controller.profile.id == "hardware-24gb"
    persisted = json.loads(selection_path.read_text())
    assert persisted["profile_id"] == "hardware-24gb"
    assert "Cyber-Tiel Coder 35B-A3B" in runtime.selection_reason
    assert "Qwen 3.8 27B GSQ-RCO" in runtime.selection_reason
    assert backends[0].events[0][0] == "local"
