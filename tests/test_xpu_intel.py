from __future__ import annotations

import subprocess

import pytest

from turbofit_runtime.gpu import parse_xpu_gpu_memory_csv
from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint, probe_hardware
from turbofit_runtime.intel_xpu import (
    parse_xpu_inventory,
    parse_xpu_processes,
    parse_xpu_cards,
    probe_xpu_pressure,
)
from turbofit_runtime.pressure import CardMemory, GPUProcess
from turbofit_runtime.pressure_probe import probe_accelerator_pressure


# Fixtures captured from xpu-smi 2.3.0 on a 2x Arc B70 + ARL iGPU box.
INVENTORY_TWO_DISCRETE_ONE_INTEGRATED = (
    "0, 00000000-0000-0004-0000-0000e2238086, Intel(R) Arc(TM) Pro B70 Graphics,"
    " 0000:04:00.0, 32656, 1.17.39395\n"
    "1, 00000000-0000-0009-0000-0000e2238086, Intel(R) Arc(TM) Pro B70 Graphics,"
    " 0000:09:00.0, 32656, 1.17.39395\n"
    "2, 00000000-0000-0200-0000-00067d678086, Intel(R) Graphics,"
    " 0000:00:02.0, 24083.296875, 1.17.39395\n"
)
MEMORY_TWO_DISCRETE_ONE_INTEGRATED = (
    "00000000-0000-0004-0000-0000e2238086, 32656, 9152.6953125\n"
    "00000000-0000-0009-0000-0000e2238086, 32656, 32546.66015625\n"
    "00000000-0000-0200-0000-00067d678086, 24083.296875, 51124.9921875\n"
)
PS_ROWS = (
    "  PID         Command                                    Type      DeviceID    SHR        MEM         Cmp%     Rnd%     Med%     Cpy%     \n"
    "  1641        python                                     Mixed     0           0          6520620     0.0      0.0      0.0      0.0     \n"
    "  1735        Hyprland                                   Mixed     0           1821680    744880      0.0      19.1     0.0      0.0     \n"
    "  137884      firefox                                    Mixed     0           1214384    1389000     0.0      15.4     0.0      0.0     \n"
    "  3639193     VLLM::EngineCore                           Mixed     1           0          33056692    0.0      0.0      0.0      0.0     \n"
    "  3508975     Hermes                                     Mixed     2           122880     167116      0.0      13.0     0.0      0.0     \n"
)

B70_A = AcceleratorDevice(
    index=0, uuid="00000000-0000-0004-0000-0000e2238086",
    name="Intel(R) Arc(TM) Pro B70 Graphics", vendor="intel", backend="vulkan",
    memory_total_mb=32656, compute_capability=None, bus_id="0000:04:00.0",
)
B70_B = AcceleratorDevice(
    index=1, uuid="00000000-0000-0009-0000-0000e2238086",
    name="Intel(R) Arc(TM) Pro B70 Graphics", vendor="intel", backend="vulkan",
    memory_total_mb=32656, compute_capability=None, bus_id="0000:09:00.0",
)
DEVICES = (B70_A, B70_B)


def fingerprint(devices=DEVICES, ram_mb=65536) -> HardwareFingerprint:
    return HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=ram_mb, devices=devices
    )


def test_parse_xpu_inventory_classifies_integrated_by_bdf() -> None:
    rows = parse_xpu_inventory(INVENTORY_TWO_DISCRETE_ONE_INTEGRATED)

    # Rows sort by bus id: the integrated part (0000:00:...) lands first.
    assert [row.integrated for row in rows] == [True, False, False]
    discrete = [row.device for row in rows if not row.integrated]
    assert [d.bus_id for d in discrete] == ["0000:04:00.0", "0000:09:00.0"]
    assert [d.vendor for d in discrete] == ["intel", "intel"]
    assert [d.backend for d in discrete] == ["vulkan", "vulkan"]
    assert discrete[0].memory_total_mb == 32656


def test_parse_xpu_inventory_rounds_fractional_totals() -> None:
    rows = parse_xpu_inventory(
        "0, u1, Intel(R) Arc(TM) A770, 0000:03:00.0, 16012.7, 1.17\n"
    )
    assert rows[0].device.memory_total_mb == 16013


def test_parse_xpu_inventory_empty_payload_raises() -> None:
    with pytest.raises(ValueError):
        parse_xpu_inventory("")


def _xpu_runner(
    inventory=INVENTORY_TWO_DISCRETE_ONE_INTEGRATED,
    memory=MEMORY_TWO_DISCRETE_ONE_INTEGRATED,
):
    def runner(command):
        if command[0] != "xpu-smi":
            # An Intel-only box: the other vendor probes are simply not installed.
            raise FileNotFoundError(command[0])
        spec = command[1] if len(command) > 1 else ""
        if spec.startswith("--query-gpu"):
            fields = spec.split("=", 1)[1]
            if "memory.used" in fields:
                return memory
            assert "driver_version" in fields  # inventory must not ask for usage
            return inventory
        if command[1:] == ["ps"]:
            return PS_ROWS
        raise AssertionError(command)
    return runner


def test_probe_hardware_prefers_xpu_over_vulkaninfo() -> None:
    def runner(command):
        if command[0] == "vulkaninfo":
            raise AssertionError("xpu-smi inventory must not fall through to vulkaninfo")
        return _xpu_runner()(command)

    hardware = probe_hardware(
        command_runner=runner, os_name="linux", architecture="x86_64", system_ram_mb=65536
    )

    assert [device.uuid for device in hardware.devices] == [B70_A.uuid, B70_B.uuid]
    assert hardware.memory_pool_kind == "dedicated"
    # iGPU's shared aperture must not add to the accelerator budget.
    assert hardware.total_vram_mb == 2 * 32656


def test_probe_hardware_integrated_only_box_skips_vulkaninfo() -> None:
    integrated_only = (
        "0, u1, Intel(R) Graphics, 0000:00:02.0, 24083.3, 1.17\n"
    )

    def runner(command):
        if command[0] == "vulkaninfo":
            raise AssertionError("an integrated-only box must not charge a fake heap via vulkaninfo")
        return _xpu_runner(inventory=integrated_only)(command)

    hardware = probe_hardware(
        command_runner=runner, os_name="linux", architecture="x86_64", system_ram_mb=65536
    )

    assert hardware.devices == ()
    assert hardware.memory_pool_kind == "cpu"


def test_probe_hardware_falls_back_to_vulkaninfo_without_xpu() -> None:
    vulkan_raw = (
        "GPU0:\n"
        "    deviceName = Intel(R) Arc(TM) A770 Graphics\n"
        "    deviceUUID = deadbeef\n"
        "    vendorID = 0x8086\n"
        "    memoryHeaps:\n"
        "        memoryHeaps[0]:\n"
        "            flags = MEMORY_HEAP_DEVICE_LOCAL_BIT\n"
        "            size = 17179869184\n"
        "        memoryTypes:\n"
    )

    def runner(command):
        if command[0] == "xpu-smi":
            raise FileNotFoundError("xpu-smi")
        if command[0] == "vulkaninfo":
            return vulkan_raw
        raise FileNotFoundError(command[0])

    hardware = probe_hardware(
        command_runner=runner, os_name="linux", architecture="x86_64", system_ram_mb=32768
    )
    assert len(hardware.devices) == 1
    assert hardware.devices[0].vendor == "intel"


def test_parse_xpu_processes_maps_device_ids_and_kib() -> None:
    processes = parse_xpu_processes(PS_ROWS, DEVICES)

    by_pid = {process.pid: process for process in processes}
    assert 3508975 not in by_pid  # integrated device is out of scope
    assert by_pid[1641] == GPUProcess(
        gpu=0, pid=1641, used_mb=round(6520620 / 1024), name="python", desktop=False
    )
    assert by_pid[1735].desktop is True
    vllm = by_pid[3639193]
    assert vllm.gpu == 1 and vllm.name == "VLLM::EngineCore" and vllm.desktop is False


def test_parse_xpu_cards_skips_integrated_and_rounds() -> None:
    cards = parse_xpu_cards(MEMORY_TWO_DISCRETE_ONE_INTEGRATED, DEVICES)

    assert cards == (
        CardMemory(0, 32656, 9153),
        CardMemory(1, 32656, 32547),
    )


def test_parse_xpu_cards_fingerprint_mismatch_raises() -> None:
    three = DEVICES + (AcceleratorDevice(
        index=3, uuid="missing", name="Ghost", vendor="intel", backend="vulkan",
        memory_total_mb=16384, compute_capability=None, bus_id="0000:0a:00.0",
    ),)
    with pytest.raises(ValueError):
        parse_xpu_cards(MEMORY_TWO_DISCRETE_ONE_INTEGRATED, three)


def test_pressure_probe_end_to_end_attribution() -> None:
    def runner(command):
        if command[0] != "xpu-smi":
            raise FileNotFoundError(command[0])
        if command[1].startswith("--query-gpu"):
            return MEMORY_TWO_DISCRETE_ONE_INTEGRATED
        if command[1:] == ["ps"]:
            return PS_ROWS
        raise AssertionError(command)

    snapshot = probe_accelerator_pressure(
        fingerprint(),
        manager_status={"residents": [{"pid": 3639193, "role": "main"}]},
        managed_required_mb=(16384, 16384),
        command_runner=runner,
    )

    assert snapshot.process_data_available is True
    cards = {card.gpu: card for card in snapshot.cards}
    # The managed VLLM engine owns card 1 entirely; that is signal, not victim.
    assert cards[1].managed_mb == round(33056692 / 1024)
    # The whole card is managed; the residual lands in external by rounding and
    # the desktop baseline, never silently counted as free.
    assert cards[1].external_mb < 16
    # Card 0 shows a python external user and a desktop-class Hyprland user.
    assert cards[0].external_mb >= 1
    assert cards[0].desktop_mb >= 1
    # Available excludes desktop, external, and the safety reserve.
    assert cards[0].available_for_managed_mb <= cards[0].total_mb - cards[0].safety_reserve_mb
    # Unexplained usage on card 1 lands in external, never silently free.
    assert cards[1].observed_used_mb == 32547
    assert snapshot.release_targets == (3639193,)


def test_pressure_probe_degrades_when_ps_missing() -> None:
    def runner(command):
        if command[0] != "xpu-smi":
            raise FileNotFoundError(command[0])
        if command[1:] == ["ps"]:
            raise subprocess.CalledProcessError(1, command)
        return MEMORY_TWO_DISCRETE_ONE_INTEGRATED

    snapshot = probe_accelerator_pressure(
        fingerprint(),
        manager_status={"residents": []},
        managed_required_mb=(0, 0),
        command_runner=runner,
    )
    assert snapshot.process_data_available is False
    assert snapshot.release_targets == ()
    # With no process data, usage is attributed conservatively: the desktop
    # baseline plus an equal amount of "unexplained" external — never free.
    assert all(
        card.external_mb + card.desktop_mb == card.observed_used_mb
        for card in snapshot.cards
    )


def test_probe_xpu_pressure_returns_none_processes_on_ps_failure() -> None:
    def runner(command):
        if command[1:] == ["ps"]:
            raise FileNotFoundError("xpu-smi")
        return MEMORY_TWO_DISCRETE_ONE_INTEGRATED

    cards, processes = probe_xpu_pressure(DEVICES, runner)
    assert len(cards) == 2 and processes is None


def test_parse_xpu_gpu_memory_csv_skips_integrated() -> None:
    samples = parse_xpu_gpu_memory_csv(
        "0, 00000000-0000-0004-0000-0000e2238086, 0000:04:00.0, 32656, 9114.52, 23541.48, 100.00\n"
        "1, 00000000-0000-0009-0000-0000e2238086, 0000:09:00.0, 32656, 32546.66, 109.34, 100.00\n"
        "2, 00000000-0000-0200-0000-00067d678086, 0000:00:02.0, 23751.19, 51534.91, 0, 0.00\n"
    )
    assert [sample.gpu for sample in samples] == [0, 1]
    assert samples[0].total_mb == 32656
    assert samples[0].used_mb == 9115
    assert samples[0].utilization_pct == 100
