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
MEASURED = {24, 48}


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
        assert profile.rungs[-1].aux_mode.value == "api"
        assert profile.rungs[-1].main_api_policy == "api:auto"
        assert profile.rungs[-1].aux_api_policy == "api:auto"


def test_only_measured_classes_publish_local_recommendation_rungs() -> None:
    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        local_rungs = profile.rungs[:-1]
        if class_gb in MEASURED:
            assert local_rungs
            assert profile.policy.recommendation == "measured-winner"
        else:
            assert local_rungs == ()
            assert profile.policy.recommendation == "api-only-unproven-local"


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


def test_hardware_envelopes_match_their_exact_canonical_topologies() -> None:
    for class_gb, topology in TOPOLOGIES.items():
        sizes: list[int] = []
        for part in topology.split("+"):
            count, size = (int(item) for item in part.split("x"))
            sizes.extend([size] * count)
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert hardware_satisfies(fingerprint(tuple(sizes)), profile.hardware)
