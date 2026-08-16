from __future__ import annotations

import subprocess
import json

import pytest

from turbofit_runtime.hardware import (
    AcceleratorDevice,
    HardwareFingerprint,
    _run_command,
    parse_nvidia_inventory_csv,
    parse_rocm_smi_json,
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


def test_parse_rocm_inventory_preserves_identity_capacity_and_bus() -> None:
    raw = json.dumps({
        "card0": {
            "Card series": "AMD Radeon PRO W7900",
            "Unique ID": "0x1234",
            "PCI Bus": "0000:41:00.0",
            "VRAM Total Memory (B)": "51527024640",
        }
    })

    devices = parse_rocm_smi_json(raw)

    assert devices == (
        AcceleratorDevice(
            index=0,
            uuid="0x1234",
            name="AMD Radeon PRO W7900",
            vendor="amd",
            backend="rocm",
            memory_total_mb=49140,
            compute_capability=None,
            bus_id="0000:41:00.0",
        ),
    )


def test_probe_falls_back_to_rocm_when_nvidia_is_absent() -> None:
    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(command)
        if command[0] == "nvidia-smi":
            raise FileNotFoundError(command[0])
        return json.dumps({
            "card0": {
                "Card series": "AMD Radeon RX 7900 XTX",
                "Unique ID": "0xabcd",
                "PCI Bus": "0000:03:00.0",
                "VRAM Total Memory (B)": str(24 * 1024**3),
            }
        })

    fingerprint = probe_hardware(
        command_runner=run,
        os_name="linux",
        architecture="x86_64",
        system_ram_mb=65536,
    )

    assert commands[0][0] == "nvidia-smi"
    assert commands[1][0] == "rocm-smi"
    assert fingerprint.backends == ("rocm",)
    assert fingerprint.total_vram_mb == 24576


def test_probe_apple_silicon_exposes_unified_memory_as_metal_capacity() -> None:
    def missing(_command: list[str]) -> str:
        raise FileNotFoundError("nvidia-smi")

    fingerprint = probe_hardware(
        command_runner=missing,
        os_name="Darwin",
        architecture="arm64",
        system_ram_mb=65536,
    )

    assert fingerprint.os == "darwin"
    assert fingerprint.backends == ("metal",)
    assert fingerprint.total_vram_mb == 57_344
    assert fingerprint.total_vram_mb < fingerprint.system_ram_mb
    assert fingerprint.devices[0].vendor == "apple"


def test_probe_labels_wsl2_without_losing_nvidia_topology() -> None:
    def run(_command: list[str]) -> str:
        return "0, GPU-a, RTX 4090, 24576, 8.9, 00000000:01:00.0\n"

    fingerprint = probe_hardware(
        command_runner=run,
        os_name="Linux",
        architecture="x86_64",
        system_ram_mb=32768,
        kernel_release="5.15.153.1-microsoft-standard-WSL2",
    )

    assert fingerprint.os == "windows-wsl2"
    assert fingerprint.backends == ("cuda",)
    assert fingerprint.total_vram_mb == 24576


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


def test_run_command_retries_nvidia_smi_with_loaded_kernel_compatibility_library() -> None:
    calls: list[dict[str, str] | None] = []

    def check_output(
        _command: list[str],
        *,
        text: bool,
        timeout: int,
        stderr: int,
        env: dict[str, str] | None = None,
    ) -> str:
        assert text is True
        assert timeout == 5
        assert stderr == subprocess.PIPE
        calls.append(env)
        if env is None:
            raise subprocess.CalledProcessError(
                18,
                ["nvidia-smi"],
                stderr="Failed to initialize NVML: Driver/library version mismatch",
            )
        return "0, GPU-a, RTX 3090, 24576, 8.6, 00000000:01:00.0\n"

    result = _run_command(
        ["nvidia-smi"],
        check_output=check_output,
        compatibility_library_dir=lambda: "/home/test/.local/lib/nvidia-580.159.03/usr/lib/x86_64-linux-gnu",
    )

    assert result.startswith("0, GPU-a")
    assert calls[0] is None
    assert calls[1] is not None
    assert calls[1]["LD_LIBRARY_PATH"].startswith(
        "/home/test/.local/lib/nvidia-580.159.03/usr/lib/x86_64-linux-gnu"
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


def test_parse_nvidia_inventory_gb10_unified_memory_falls_back_to_system_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # DGX Spark (GB10) reports memory.total as "[N/A]" via nvidia-smi because
    # the GPU shares unified memory with the CPU. The parser must fall back to
    # system RAM instead of raising "invalid NVIDIA numeric field".
    import turbofit_runtime.hardware as hw

    monkeypatch.setattr(hw, "_system_ram_mb", lambda: 131072)
    raw = "0, GPU-gb10, NVIDIA GB10, [N/A], 12.1, 00000000:01:00.0\n"

    devices = parse_nvidia_inventory_csv(raw)

    assert len(devices) == 1
    assert devices[0].name == "NVIDIA GB10"
    assert devices[0].memory_total_mb == 131072
    assert devices[0].compute_capability == "12.1"


def test_memory_capacity_distinguishes_dedicated_unified_and_cpu_pools() -> None:
    dedicated = HardwareFingerprint(
        os="linux", architecture="x86_64", system_ram_mb=393216,
        devices=(
            AcceleratorDevice(0, "a", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "01"),
            AcceleratorDevice(1, "b", "RTX 3090", "nvidia", "cuda", 24576, "8.6", "02"),
        ),
    )
    unified = HardwareFingerprint(
        os="linux", architecture="aarch64", system_ram_mb=131072,
        devices=(AcceleratorDevice(0, "u", "NVIDIA GB10", "nvidia", "cuda", 131072, "12.1", "01"),),
    )
    cpu = HardwareFingerprint(os="linux", architecture="x86_64", system_ram_mb=65536)

    assert dedicated.memory_pool_kind == "dedicated"
    assert dedicated.host_usable_memory_mb == 385024
    assert dedicated.total_usable_memory_mb == 434176
    assert unified.memory_pool_kind == "unified"
    assert unified.total_usable_memory_mb == unified.host_usable_memory_mb == 124518
    assert cpu.memory_pool_kind == "cpu"
    assert cpu.total_usable_memory_mb == cpu.host_usable_memory_mb == 62259
