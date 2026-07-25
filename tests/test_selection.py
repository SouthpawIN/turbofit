from __future__ import annotations

from dataclasses import replace

import pytest

from test_runtime_profile import valid_mapping
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.runtime_profile import Turbofile
from turbofit_runtime.selection import (
    ProfileCatalog,
    SelectionMode,
    load_selection,
    save_selection,
)


def hardware(*memory_mb: int) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=65536,
        devices=tuple(
            AcceleratorDevice(
                index=index,
                uuid=f"GPU-{index}",
                name="GPU",
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=memory,
                compute_capability="8.6",
                bus_id=f"0{index}",
            )
            for index, memory in enumerate(memory_mb)
        ),
    )


def profile(class_gb: int, topology: str, *, local: bool) -> Turbofile:
    data = valid_mapping()
    data["id"] = f"hardware-{class_gb}gb"
    data["hardware"].update(
        class_vram_gb=class_gb,
        min_devices=sum(int(part.split("x", 1)[0]) for part in topology.split("+")),
        total_vram_gb=class_gb,
        per_device_min_gb=min(
            int(part.split("x", 1)[1]) for part in topology.split("+")
        ),
        topology=topology,
    )
    if class_gb == 48:
        data["policy"]["expansion_margin_gb_per_card"] = 0.5
    api = {
        "id": "api",
        "context": 131072,
        "aux_mode": "api",
        "evidence": "sha256:" + "f" * 64,
        "main_api_policy": "api:auto",
        "aux_api_policy": "api:auto",
    }
    if local:
        data["rungs"] = [data["rungs"][0], api]
    else:
        data["policy"]["recommendation"] = "api-only-unproven-local"
        data["rungs"] = [api]
    return Turbofile.from_mapping(data)


def catalog() -> ProfileCatalog:
    return ProfileCatalog(
        (
            profile(8, "1x8", local=False),
            profile(16, "1x16", local=False),
            profile(24, "1x24", local=True),
            profile(48, "2x24", local=True),
        )
    )


@pytest.mark.parametrize(
    ("memory", "expected"),
    [((8192,), "hardware-8gb"), ((16384,), "hardware-16gb"), ((24576,), "hardware-24gb"), ((24576, 24576), "hardware-48gb")],
)
def test_auto_selects_canonical_profile_from_physical_topology(
    memory: tuple[int, ...], expected: str
) -> None:
    choice = catalog().select(hardware(*memory), requested="auto")

    assert choice.mode is SelectionMode.AUTO
    assert choice.profile.id == expected
    assert choice.initial_rung_index == len(choice.profile.rungs) - 1
    assert choice.target_ceiling_index == 0


def test_auto_uses_api_only_lower_class_for_unmeasured_topology() -> None:
    choice = catalog().select(hardware(12288), requested="auto")

    assert choice.profile.id == "hardware-8gb"
    assert all(rung.aux_mode.value == "api" for rung in choice.profile.rungs)


def test_manual_local_profile_must_fit_physical_topology() -> None:
    with pytest.raises(ValueError, match="does not fit physical hardware"):
        catalog().select(hardware(16384), requested="hardware-24gb")


def test_manual_api_only_profile_is_safe_on_larger_hardware() -> None:
    choice = catalog().select(hardware(24576), requested="hardware-16gb")

    assert choice.mode is SelectionMode.MANUAL
    assert choice.profile.id == "hardware-16gb"
    assert choice.initial_rung_index == 0


def test_catalog_rejects_duplicate_ids_and_profiles_without_terminal_api() -> None:
    item = profile(8, "1x8", local=False)
    with pytest.raises(ValueError, match="duplicate profile id"):
        ProfileCatalog((item, item))

    local_only = replace(profile(24, "1x24", local=True), rungs=(profile(24, "1x24", local=True).rungs[0],))
    with pytest.raises(ValueError, match="terminal API rung"):
        ProfileCatalog((local_only,))


def test_selection_record_round_trips_atomically(tmp_path) -> None:
    choice = catalog().select(hardware(24576), requested="auto")
    path = tmp_path / "selection.json"

    save_selection(path, choice)
    restored = load_selection(path)

    assert restored == {
        "schema": "turbofit.selection/v1",
        "mode": "auto",
        "requested": "auto",
        "profile_id": "hardware-24gb",
        "target_ceiling_index": 0,
        "initial_rung_index": 1,
    }
    assert not list(tmp_path.glob(".selection.json.*"))
