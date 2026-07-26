from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.recommend import hardware_satisfies


ROOT = Path(__file__).parents[1]
CLASSES = (8, 16, 24, 48, 64, 96, 200, 300)
TOPOLOGIES = {
    8: "1x8",
    16: "1x16",
    24: "1x24",
    48: "2x24",
    64: "2x32",
    96: "4x24",
    200: "2x100",
    300: "3x100",
}


def fingerprint(memory_gb: tuple[int, ...]) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=524288,
        devices=tuple(
            AcceleratorDevice(
                index=index,
                uuid=f"GPU-{index}",
                name="GPU",
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=size * 1024,
                compute_capability="8.6",
                bus_id=f"{index:02d}",
            )
            for index, size in enumerate(memory_gb)
        ),
    )


def test_all_hardware_classes_are_valid_profiles() -> None:
    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert profile.id == f"hardware-{class_gb}gb"
        assert profile.hardware.class_vram_gb == class_gb
        assert profile.hardware.topology == TOPOLOGIES[class_gb]
        assert all(rung.aux_mode.value != "api" for rung in profile.rungs)
        assert all(rung.main_api_policy is None for rung in profile.rungs)
        assert all(rung.aux_api_policy is None for rung in profile.rungs)


def test_every_hardware_class_publishes_a_local_recommendation_ladder() -> None:
    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert profile.rungs
        assert all(rung.aux_mode.value in {"shared-main", "dedicated"} for rung in profile.rungs)
        assert profile.policy.recommendation in {"measured-winner", "portable-local-floor"}


def test_every_rung_evidence_identity_resolves() -> None:
    measured_index = json.loads((ROOT / "runtime-profiles" / "evidence-index.json").read_text())
    class_index = json.loads((ROOT / "runtime-profiles" / "class-evidence-index.json").read_text())
    identities = set(measured_index) | set(class_index)

    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert all(rung.evidence in identities for rung in profile.rungs)


def test_topology_subkeys_are_used_for_physical_matching() -> None:
    profile_48 = load_yaml_profile(ROOT / "runtime-profiles" / "48gb.yaml")

    assert hardware_satisfies(fingerprint((24, 24)), profile_48.hardware)
    assert not hardware_satisfies(fingerprint((48,)), profile_48.hardware)


def test_same_card_count_with_larger_vram_can_use_smaller_proven_envelope() -> None:
    profile_24 = load_yaml_profile(ROOT / "runtime-profiles" / "24gb.yaml")
    profile_48 = load_yaml_profile(ROOT / "runtime-profiles" / "48gb.yaml")

    assert hardware_satisfies(fingerprint((32,)), profile_24.hardware)
    assert hardware_satisfies(fingerprint((32, 32)), profile_48.hardware)
    assert not hardware_satisfies(fingerprint((48,)), profile_48.hardware)


def test_hardware_envelopes_match_their_canonical_topologies() -> None:
    for class_gb, topology in TOPOLOGIES.items():
        sizes: list[int] = []
        for part in topology.split("+"):
            count, size = (int(item) for item in part.split("x"))
            sizes.extend([size] * count)
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert hardware_satisfies(fingerprint(tuple(sizes)), profile.hardware)
