from __future__ import annotations

from test_selection import hardware
from turbofit_runtime.hardware import HardwareFingerprint
from turbofit_runtime.pressure_probe import (
    normalize_managed_requirements,
    probe_accelerator_pressure,
    probe_nvidia_pressure,
)


CARD_CSV = "0, 24576, 23000\n1, 24576, 1000\n"
PROCESS_CSV = (
    "GPU-0, 111, 22000, llama-server\n"
    "GPU-0, 222, 500, blender\n"
    "GPU-1, 333, 200, Xorg\n"
)


def test_probe_classifies_manager_desktop_and_external_memory_per_card() -> None:
    fingerprint = hardware(24576, 24576)

    def runner(command: list[str]) -> str:
        return PROCESS_CSV if "--query-compute-apps" in command[1] else CARD_CSV

    snapshot = probe_nvidia_pressure(
        fingerprint,
        manager_status={"residents": [{"model_tag": "main", "pid": 111}]},
        managed_required_mb=(22000, 0),
        command_runner=runner,
        desktop_baseline_mb=256,
        safety_reserve_mb=512,
    )

    assert snapshot.cards[0].managed_mb == 22000
    assert snapshot.cards[0].external_mb == 744
    assert snapshot.cards[0].available_for_managed_mb == 23064
    assert snapshot.cards[1].desktop_mb == 256
    assert snapshot.cards[1].external_mb == 744
    assert snapshot.release_targets == (111,)


def test_probe_fails_closed_when_process_inventory_is_unavailable() -> None:
    fingerprint = hardware(24576)

    def runner(command: list[str]) -> str:
        if "--query-compute-apps" in command[1]:
            raise OSError("NVML process query unavailable")
        return "0, 24576, 1000\n"

    snapshot = probe_nvidia_pressure(
        fingerprint,
        manager_status={"active": {"model_tag": "main", "pid": 111}},
        managed_required_mb=(500,),
        command_runner=runner,
        desktop_baseline_mb=256,
        safety_reserve_mb=512,
    )

    assert snapshot.process_data_available is False
    assert snapshot.cards[0].external_mb == 244
    assert snapshot.release_targets == ()


def test_cpu_only_pressure_uses_available_system_ram_as_the_managed_pool() -> None:
    fingerprint = HardwareFingerprint("linux", "x86_64", 65536)

    snapshot = probe_accelerator_pressure(
        fingerprint,
        manager_status={"residents": [{"pid": 111}]},
        managed_required_mb=(),
        system_available_mb=49152,
        desktop_baseline_mb=0,
        safety_reserve_mb=1024,
    )

    assert snapshot.cards[0].total_mb == fingerprint.host_usable_memory_mb
    assert snapshot.cards[0].observed_used_mb == fingerprint.host_usable_memory_mb - 49152
    assert snapshot.cards[0].available_for_managed_mb == 48128
    assert snapshot.process_data_available is False


def test_shared_memory_requirements_collapse_discrete_profile_cards_into_one_pool() -> None:
    cpu = HardwareFingerprint("linux", "x86_64", 393216)

    assert normalize_managed_requirements(cpu, (24576, 24576, 24576)) == (73728,)
