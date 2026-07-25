from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from turbofit_runtime.runtime_profile import (
    AuxMode,
    HardwareConstraint,
    RoleRoutes,
    RuntimePolicy,
    RuntimeRung,
    Turbofile,
)


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def valid_mapping() -> dict[str, Any]:
    return {
        "schema": "turbofit.runtime/v1",
        "id": "quality-24gb",
        "revision": 1,
        "hardware": {
            "class_vram_gb": 24,
            "min_devices": 1,
            "total_vram_gb": 24,
            "per_device_min_gb": 24,
            "accelerator": "nvidia-cuda",
            "compute_capability_min": "8.6",
            "system_ram_gb": 64,
            "topology": "1x24",
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
            {
                "id": "shared-main-128k",
                "context": 131072,
                "aux_mode": "shared-main",
                "evidence": "sha256:" + "c" * 64,
                "main_manifest": DIGEST_A,
            },
            {
                "id": "api",
                "context": 131072,
                "aux_mode": "api",
                "evidence": "sha256:" + "d" * 64,
                "main_api_policy": "api:auto",
                "aux_api_policy": "api:auto",
            },
        ],
    }


def assert_plain(value: Any) -> None:
    assert not is_dataclass(value)
    assert not isinstance(value, tuple)
    if isinstance(value, dict):
        for key, child in value.items():
            assert isinstance(key, str)
            assert_plain(child)
    elif isinstance(value, list):
        for child in value:
            assert_plain(child)


def test_valid_mapping_builds_frozen_profile_and_preserves_rung_order() -> None:
    profile = Turbofile.from_mapping(valid_mapping())

    assert profile.id == "quality-24gb"
    assert profile.hardware.topology == "1x24"
    assert profile.policy.external_gpu_priority == "absolute"
    assert tuple(rung.id for rung in profile.rungs) == ("shared-main-128k", "api")
    assert isinstance(profile.rungs, tuple)
    with pytest.raises(FrozenInstanceError):
        profile.revision = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        profile.hardware.min_devices = 2  # type: ignore[misc]


def test_to_mapping_is_plain_deterministic_and_round_trips() -> None:
    profile = Turbofile.from_mapping(valid_mapping())

    first = profile.to_mapping()
    second = profile.to_mapping()

    assert first == second
    assert_plain(first)
    assert Turbofile.from_mapping(first) == profile


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("extra",), True),
        (("hardware", "extra"), True),
        (("policy", "extra"), True),
        (("roles", "extra"), True),
        (("rungs", 0, "extra"), True),
    ],
)
def test_unknown_fields_are_rejected_at_every_object(path: tuple[Any, ...], value: Any) -> None:
    mapping = valid_mapping()
    target: Any = mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="unknown field"):
        Turbofile.from_mapping(mapping)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("roles", "main"), "/home/user/model.gguf", "local path"),
        (("roles", "main"), "~/model.gguf", "local path"),
        (("policy", "api_key"), "secret", "secret field"),
        (("rungs", 0, "gpu_index"), 1, "placement field"),
        (("rungs", 0, "main_api_policy"), "Bearer abc", "secret value"),
        (("rungs", 0, "main_api_policy"), "sk-live-secret", "secret value"),
    ],
)
def test_nonportable_or_secret_content_is_rejected(
    path: tuple[Any, ...], value: Any, message: str
) -> None:
    mapping = valid_mapping()
    target: Any = mapping
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        Turbofile.from_mapping(mapping)


def test_urls_api_routes_and_content_addresses_are_portable() -> None:
    mapping = valid_mapping()
    mapping["roles"]["fallback"] = "https://api.example.test/v1"
    mapping["rungs"][-1]["main_api_policy"] = "api:auto"

    profile = Turbofile.from_mapping(mapping)

    assert profile.roles.fallback.startswith("https://")
    assert profile.rungs[0].main_manifest == DIGEST_A


@pytest.mark.parametrize("field", ["class_vram_gb", "total_vram_gb", "per_device_min_gb"])
def test_hardware_capacity_fields_must_be_positive(field: str) -> None:
    mapping = valid_mapping()
    mapping["hardware"][field] = 0
    with pytest.raises(ValueError, match=field):
        Turbofile.from_mapping(mapping)


def test_numeric_fields_reject_booleans_nan_and_infinity() -> None:
    for field, value in (
        ("contraction_dwell_s", float("nan")),
        ("expansion_dwell_s", float("inf")),
        ("expansion_margin_gb_per_card", float("-inf")),
        ("cooldown_s", True),
    ):
        mapping = valid_mapping()
        mapping["policy"][field] = value
        with pytest.raises(ValueError, match=field):
            Turbofile.from_mapping(mapping)

    mapping = valid_mapping()
    mapping["hardware"]["class_vram_gb"] = True
    with pytest.raises(ValueError, match="class_vram_gb"):
        Turbofile.from_mapping(mapping)


def test_hardware_device_count_and_total_capacity_are_consistent() -> None:
    mapping = valid_mapping()
    mapping["hardware"].update(min_devices=2, total_vram_gb=24, per_device_min_gb=16)
    with pytest.raises(ValueError, match="min_devices"):
        Turbofile.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contraction_dwell_s", -1),
        ("expansion_dwell_s", -1),
        ("expansion_margin_gb_per_card", -1),
        ("cooldown_s", -1),
        ("flap_failure_limit", 0),
        ("flap_window_s", -1),
    ],
)
def test_policy_bounds_are_enforced(field: str, value: int) -> None:
    mapping = valid_mapping()
    mapping["policy"][field] = value
    with pytest.raises(ValueError, match=field):
        Turbofile.from_mapping(mapping)


def test_external_gpu_priority_is_absolute() -> None:
    mapping = valid_mapping()
    mapping["policy"]["external_gpu_priority"] = "best-effort"
    with pytest.raises(ValueError, match="external_gpu_priority"):
        Turbofile.from_mapping(mapping)


def test_revision_context_and_profile_identifiers_are_validated() -> None:
    for mutate, message in (
        (lambda data: data.update(revision=0), "revision"),
        (lambda data: data.update(id="Not Portable"), "id"),
        (lambda data: data["rungs"][0].update(context=0), "context"),
    ):
        mapping = valid_mapping()
        mutate(mapping)
        with pytest.raises(ValueError, match=message):
            Turbofile.from_mapping(mapping)


def test_rung_ids_are_unique() -> None:
    mapping = valid_mapping()
    mapping["rungs"][1]["id"] = mapping["rungs"][0]["id"]
    with pytest.raises(ValueError, match="duplicate rung"):
        Turbofile.from_mapping(mapping)


def test_local_rung_manifest_contracts_are_enforced() -> None:
    missing_main = valid_mapping()
    del missing_main["rungs"][0]["main_manifest"]
    with pytest.raises(ValueError, match="main_manifest"):
        Turbofile.from_mapping(missing_main)

    dedicated = valid_mapping()
    dedicated["rungs"][0]["aux_mode"] = "dedicated"
    with pytest.raises(ValueError, match="aux_manifest"):
        Turbofile.from_mapping(dedicated)

    shared_with_aux = valid_mapping()
    shared_with_aux["rungs"][0]["aux_manifest"] = DIGEST_B
    with pytest.raises(ValueError, match="shared-main"):
        Turbofile.from_mapping(shared_with_aux)


def test_manifest_references_are_exact_sha256_content_addresses() -> None:
    mapping = valid_mapping()
    mapping["rungs"][0]["main_manifest"] = "sha256:abc"
    with pytest.raises(ValueError, match="sha256"):
        Turbofile.from_mapping(mapping)


def test_terminal_rung_is_api_only_with_both_policies_and_no_manifests() -> None:
    for mutate, message in (
        (lambda rung: rung.update(aux_mode="shared-main"), "terminal rung"),
        (lambda rung: rung.pop("main_api_policy"), "main_api_policy"),
        (lambda rung: rung.pop("aux_api_policy"), "aux_api_policy"),
        (lambda rung: rung.update(main_manifest=DIGEST_A), "local manifests"),
    ):
        mapping = valid_mapping()
        mutate(mapping["rungs"][-1])
        with pytest.raises(ValueError, match=message):
            Turbofile.from_mapping(mapping)


def test_empty_rung_list_is_rejected() -> None:
    mapping = valid_mapping()
    mapping["rungs"] = []
    with pytest.raises(ValueError, match="rungs"):
        Turbofile.from_mapping(mapping)


def test_schema_example_matches_contract() -> None:
    yaml = pytest.importorskip("yaml")
    path = Path(__file__).parents[1] / "runtime-profiles" / "schema-example.yaml"
    profile = Turbofile.from_mapping(yaml.safe_load(path.read_text()))

    assert profile.hardware.class_vram_gb == 24
    assert profile.rungs[0].aux_mode is AuxMode.SHARED_MAIN
    assert profile.rungs[-1].aux_mode is AuxMode.API
