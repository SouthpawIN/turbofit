from __future__ import annotations

import subprocess

import pytest

from turbofit_runtime.hardware import (
    AcceleratorDevice,
    HardwareFingerprint,
    parse_nvidia_inventory_csv,
    probe_hardware,
)


def test_parse_nvidia_inventory_preserves_topology_fields() -> None:
    raw = (
        "0, GPU-aaaa, NVIDIA GeForce RTX 3090, 24576, 8.6, 00000000:01:00.0\n"
        "1, GPU-bbbb, NVIDIA GeForce RTX 3090, 24576, 8.6, 00000000:02:00.0\n"
    )

    devices = parse_nvidia_inventory_csv(raw)

    assert devices == (
        AcceleratorDevice(
            index=0,
            uuid="GPU-aaaa",
            name="NVIDIA GeForce RTX 3090",
            vendor="nvidia",
            backend="cuda",
            memory_total_mb=24576,
            compute_capability="8.6",
            bus_id="00000000:01:00.0",
        ),
        AcceleratorDevice(
            index=1,
            uuid="GPU-bbbb",
            name="NVIDIA GeForce RTX 3090",
            vendor="nvidia",
            backend="cuda",
            memory_total_mb=24576,
            compute_capability="8.6",
            bus_id="00000000:02:00.0",
        ),
    )


def test_topology_key_distinguishes_one_48gb_from_two_24gb_cards() -> None:
    one = HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=131072,
        devices=(AcceleratorDevice(0, "a", "A", "nvidia", "cuda", 49152, "8.9", "01"),),
    )
    two = HardwareFingerprint(
        os="linux",
        architecture="x86_64",
        system_ram_mb=131072,
        devices=(
            AcceleratorDevice(0, "b", "B", "nvidia", "cuda", 24576, "8.6", "01"),
            AcceleratorDevice(1, "c", "B", "nvidia", "cuda", 24576, "8.6", "02"),
        ),
    )

    assert one.total_vram_mb == two.total_vram_mb == 49152
    assert one.topology_key == "1x49152mb"
    assert two.topology_key == "2x24576mb"
    assert one.recommendation_key != two.recommendation_key


def test_fingerprint_normalizes_device_order_and_capabilities() -> None:
    high = AcceleratorDevice(1, "b", "B", "nvidia", "cuda", 24576, "8.6", "02")
    low = AcceleratorDevice(0, "a", "A", "nvidia", "cuda", 16384, "8.0", "01")

    fingerprint = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=65536, devices=(high, low)
    )

    assert tuple(device.index for device in fingerprint.devices) == (0, 1)
    assert fingerprint.backends == ("cuda",)
    assert fingerprint.vendors == ("nvidia",)
    assert fingerprint.topology_key == "1x16384mb+1x24576mb"


def test_malformed_nvidia_inventory_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="six NVIDIA inventory columns"):
        parse_nvidia_inventory_csv("0, GPU-a, RTX, 24576")


def test_probe_without_nvidia_returns_honest_no_accelerator_fingerprint() -> None:
    def missing(_command: list[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    fingerprint = probe_hardware(
        command_runner=missing,
        os_name="linux",
        architecture="x86_64",
        system_ram_mb=32768,
    )

    assert fingerprint.devices == ()
    assert fingerprint.backends == ()
    assert fingerprint.topology_key == "no-accelerator"
    assert ":no-accelerator:" in fingerprint.recommendation_key
    assert fingerprint.recommendation_key.endswith("ram-32768mb")


def test_probe_uses_stable_nvidia_query() -> None:
    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(command)
        return "0, GPU-a, RTX 4090, 24576, 8.9, 00000000:01:00.0\n"

    fingerprint = probe_hardware(
        command_runner=run,
        os_name="linux",
        architecture="x86_64",
        system_ram_mb=65536,
    )

    assert commands == [[
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]]
    assert fingerprint.devices[0].name == "RTX 4090"


def test_fingerprint_rejects_boolean_capacities_and_duplicate_device_identity() -> None:
    with pytest.raises(ValueError, match="memory_total_mb"):
        AcceleratorDevice(0, "a", "A", "nvidia", "cuda", True, "8.6", "01")
    with pytest.raises(ValueError, match="system_ram_mb"):
        HardwareFingerprint(os="linux", architecture="x86_64", system_ram_mb=True)

    first = AcceleratorDevice(0, "same", "A", "nvidia", "cuda", 1024, "8.6", "01")
    duplicate = AcceleratorDevice(1, "same", "B", "nvidia", "cuda", 2048, "8.6", "02")
    with pytest.raises(ValueError, match="duplicate accelerator uuid"):
        HardwareFingerprint(
            os="linux", architecture="x86_64", system_ram_mb=8192, devices=(first, duplicate)
        )


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(1, ["nvidia-smi"]), subprocess.TimeoutExpired(["nvidia-smi"], 5)],
)
def test_probe_subprocess_failures_return_no_accelerator(error: Exception) -> None:
    def failed(_command: list[str]) -> str:
        raise error

    fingerprint = probe_hardware(
        command_runner=failed,
        os_name="linux",
        architecture="x86_64",
        system_ram_mb=32768,
    )
    assert fingerprint.topology_key == "no-accelerator"
