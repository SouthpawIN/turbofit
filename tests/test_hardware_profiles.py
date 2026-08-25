from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.profile_io import load_profile as load_yaml_profile
from turbofit_runtime.recipes import RecipeBook
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
        assert profile.hardware.accelerator == "llama.cpp-local"
        assert profile.hardware.topology == TOPOLOGIES[class_gb]
        assert all(rung.aux_mode.value != "api" for rung in profile.rungs)
        assert all(rung.main_api_policy is None for rung in profile.rungs)
        assert all(rung.aux_api_policy is None for rung in profile.rungs)


def test_every_hardware_class_publishes_a_local_recommendation_ladder() -> None:
    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        assert profile.rungs
        assert all(rung.aux_mode.value in {"shared-main", "dedicated"} for rung in profile.rungs)
        assert profile.policy.recommendation == "evidence-gated"


def test_every_resolved_role_matches_its_rung_context() -> None:
    resolutions = json.loads(
        (ROOT / "runtime-profiles" / "runtime-resolutions.json").read_text()
    )["profiles"]
    recipes = RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cpu")

    for class_gb in CLASSES:
        profile = load_yaml_profile(ROOT / "runtime-profiles" / f"{class_gb}gb.yaml")
        for rung in profile.rungs:
            for role, target in resolutions[profile.id][rung.id].items():
                component = recipes.resolve_component(
                    target["family"], role=role, context=rung.context,
                    gpu=target["gpu"], port=int(target["port"]), alias=target["model_tag"],
                )
                assert component.command[component.command.index("-c") + 1] == str(rung.context)


def test_48gb_ceiling_routes_main_and_aux_at_262k() -> None:
    profile = load_yaml_profile(ROOT / "runtime-profiles" / "48gb.yaml")
    ceiling = profile.rungs[0]

    assert ceiling.context == 262144
    assert ceiling.aux_mode.value == "shared-main"


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


def test_dedicated_8gb_tier_allows_only_the_windows_vulkan_reporting_delta() -> None:
    profile = load_yaml_profile(ROOT / "runtime-profiles" / "8gb.yaml")
    windows_vulkan = HardwareFingerprint(
        os="windows",
        architecture="amd64",
        system_ram_mb=65_455,
        devices=(
            AcceleratorDevice(
                index=0,
                uuid="00000000-0900-0000-0000-000000000000",
                name="AMD Radeon RX 6600",
                vendor="amd",
                backend="vulkan",
                memory_total_mb=8_176,
                compute_capability=None,
                bus_id=None,
            ),
        ),
    )
    undersized = HardwareFingerprint(
        os="windows",
        architecture="amd64",
        system_ram_mb=65_455,
        devices=(
            AcceleratorDevice(
                index=0,
                uuid="00000000-0900-0000-0000-000000000001",
                name="AMD Radeon RX 6600",
                vendor="amd",
                backend="vulkan",
                memory_total_mb=8_175,
                compute_capability=None,
                bus_id=None,
            ),
        ),
    )

    assert hardware_satisfies(windows_vulkan, profile.hardware)
    assert not hardware_satisfies(undersized, profile.hardware)


def test_large_shared_memory_pools_are_not_rejected_by_discrete_card_topology() -> None:
    profile = load_yaml_profile(ROOT / "runtime-profiles" / "96gb.yaml")
    cpu = HardwareFingerprint("linux", "x86_64", 131_072)
    apple = HardwareFingerprint(
        "darwin", "arm64", 131_072,
        devices=(AcceleratorDevice(
            0, "apple-unified-memory", "Apple Silicon Unified Memory",
            "apple", "metal", 122_880, None, None,
        ),),
    )

    assert hardware_satisfies(cpu, profile.hardware)
    assert hardware_satisfies(apple, profile.hardware)


def test_tournament_ranking_uses_current_30_and_50_tps_milestones() -> None:
    ranking = json.loads((ROOT / "references/hardware-tier-tournaments.json").read_text())["ranking"]

    assert ranking == [
        "quality", "context_128k", "server_decode_30_tps",
        "context_262k", "server_decode_50_tps", "context_1m",
    ]
