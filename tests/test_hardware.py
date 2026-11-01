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
    parse_vulkaninfo_text,
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


def test_probe_falls_back_to_vulkan_on_windows_with_dedicated_heap_capacity() -> None:
    raw = """GPU0:
    vendorID           = 0x1002
    deviceName         = AMD Radeon RX 6600
    deviceUUID         = 00112233-4455-6677-8899-aabbccddeeff
    memoryHeaps: count = 2
    memoryHeaps[0]:
        size   = 8304721920
        budget = 6937617920
        flags: count = 1
            MEMORY_HEAP_DEVICE_LOCAL_BIT
    memoryHeaps[1]:
        size   = 34317664256
        budget = 33512374272
        flags:
            None
    memoryTypes: count = 2
"""

    devices = parse_vulkaninfo_text(raw)
    fingerprint = probe_hardware(
        command_runner=lambda command: (_ for _ in ()).throw(FileNotFoundError(command[0]))
        if command[0] == "nvidia-smi" else raw,
        os_name="Windows",
        architecture="amd64",
        system_ram_mb=65536,
    )

    assert devices[0].vendor == "amd"
    assert devices[0].backend == "vulkan"
    assert devices[0].memory_total_mb == 7920
    assert fingerprint.topology_key == "1x7920mb"
    assert fingerprint.backends == ("vulkan",)


def test_vulkan_probe_drops_integrated_gpu_when_discrete_gpus_present() -> None:
    # A desktop box with discrete cards: the iGPU's device-local heap is
    # shared system RAM (Arrow Lake reports ~46 GiB from a 64 GB machine),
    # so it must not inflate device counts or VRAM totals.
    raw = """GPU0:
    vendorID           = 0x8086
    deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName         = Intel(R) Graphics (BMG G31)
    deviceUUID         = 868023e2-0000-0000-0400-000000000000
    memoryHeaps: count = 1
    memoryHeaps[0]:
        size   = 34242297856
        budget = 34242297856
        flags: count = 1
            MEMORY_HEAP_DEVICE_LOCAL_BIT
    memoryTypes: count = 2
GPU1:
    vendorID           = 0x8086
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = Intel(R) Graphics (ARL)
    deviceUUID         = 8680677d-0600-0000-0002-000000000000
    memoryHeaps: count = 1
    memoryHeaps[0]:
        size   = 49976927232
        budget = 49976927232
        flags: count = 1
            MEMORY_HEAP_DEVICE_LOCAL_BIT
    memoryTypes: count = 2
"""

    devices = parse_vulkaninfo_text(raw)

    assert [device.name for device in devices] == ["Intel(R) Graphics (BMG G31)"]
    assert devices[0].memory_total_mb == 32656


def test_vulkan_probe_keeps_integrated_gpu_when_no_discrete_gpus() -> None:
    raw = """GPU0:
    vendorID           = 0x8086
    deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    deviceName         = Intel(R) Graphics (ARL)
    deviceUUID         = 8680677d-0600-0000-0002-000000000000
    memoryHeaps: count = 1
    memoryHeaps[0]:
        size   = 49976927232
        budget = 49976927232
        flags: count = 1
            MEMORY_HEAP_DEVICE_LOCAL_BIT
    memoryTypes: count = 2
"""

    devices = parse_vulkaninfo_text(raw)

    assert len(devices) == 1
    assert devices[0].name == "Intel(R) Graphics (ARL)"


def test_probe_falls_back_to_vulkan_on_linux_when_cuda_and_rocm_are_absent() -> None:
    raw = """GPU0:
    vendorID           = 0x1002
    deviceName         = AMD Radeon RX 6600
    deviceUUID         = 00112233-4455-6677-8899-aabbccddeeff
    memoryHeaps: count = 2
    memoryHeaps[0]:
        size   = 8304721920
        budget = 6937617920
        flags: count = 1
            MEMORY_HEAP_DEVICE_LOCAL_BIT
    memoryHeaps[1]:
        size   = 34317664256
        budget = 33512374272
        flags:
            None
    memoryTypes: count = 2
"""

    def run(command: list[str]) -> str:
        if command[0] in {"nvidia-smi", "rocm-smi"}:
            raise FileNotFoundError(command[0])
        return raw

    fingerprint = probe_hardware(
        command_runner=run,
        os_name="linux",
        architecture="x86_64",
        system_ram_mb=65536,
    )

    assert fingerprint.backends == ("vulkan",)
    assert fingerprint.topology_key == "1x7920mb"


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


def test_run_command_gives_vulkaninfo_a_longer_timeout() -> None:
    seen: list[int] = []

    def check_output(_command: list[str], *, text: bool, timeout: int, stderr: int) -> str:
        seen.append(timeout)
        return "GPU0:\n"

    _run_command(["vulkaninfo"], check_output=check_output)
    _run_command(["nvidia-smi"], check_output=check_output)

    assert seen == [30, 5]


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


# The RAM probe is the accelerator budget on unified-memory parts (GB10 reports memory.total as
# "[N/A]"; Metal shares system RAM), so it must answer or say it cannot -- never raise, and never
# hand back a negative capacity. Every source is stubbed below so all branches run on any runner.

MEMINFO_128 = "MemTotal:       134217728 kB\nMemFree:         1000000 kB\n"


def _sysconf(values: dict | None):
    # A libc that does not define a name makes os.sysconf raise, not return 0.
    def fake(name: str) -> int:
        if values is None or name not in values:
            raise ValueError(f"unrecognized configuration name: {name}")
        return values[name]

    return fake


def _host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sysconf: dict | None,
    meminfo: str | None,
    windows: bool = False,
) -> None:
    """Pin every source the probe can consult. ``sysconf=None`` is a libc without the names;
    ``windows=False`` takes GlobalMemoryStatusEx away so the POSIX branches run on a Windows
    runner too."""
    import ctypes

    import turbofit_runtime.hardware as hw

    if sysconf is None:
        monkeypatch.delattr(hw.os, "sysconf", raising=False)
    else:
        monkeypatch.setattr(hw.os, "sysconf", _sysconf(sysconf or None), raising=False)
    if not windows:
        monkeypatch.delattr(ctypes, "windll", raising=False)

    def read_text(self, *args, **kwargs):
        if self.as_posix() == "/proc/meminfo" and meminfo is not None:
            return meminfo
        raise FileNotFoundError(self.as_posix())

    monkeypatch.setattr(hw.Path, "read_text", read_text)


class _FakeWindll:
    """GlobalMemoryStatusEx writes through the byref pointer and reports success separately."""

    def __init__(self, *, ok: bool, total_bytes: int) -> None:
        self.kernel32 = self
        self._ok, self._total = ok, total_bytes

    def GlobalMemoryStatusEx(self, ref) -> int:  # noqa: N802 - Win32 spelling
        if self._ok:
            ref._obj.ullTotalPhys = self._total
        return 1 if self._ok else 0


def test_system_ram_is_read_in_process_without_consulting_another_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import turbofit_runtime.hardware as hw

    _host(monkeypatch, sysconf={"SC_PHYS_PAGES": 32768, "SC_PAGE_SIZE": 4096}, meminfo=None)

    assert hw._system_ram_mb() == 128


def test_system_ram_survives_a_libc_that_lacks_the_sysconf_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.sysconf raises ValueError for a name its libc does not define. That escaped the probe and
    # propagated out of probe_hardware; the fallbacks below it were never reached.
    import turbofit_runtime.hardware as hw

    _host(monkeypatch, sysconf={}, meminfo=MEMINFO_128)

    assert hw._system_ram_mb() == 131072


def test_system_ram_treats_an_indeterminate_sysconf_answer_as_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sysconf signals "indeterminate" with -1 rather than raising. Multiplied through, that
    # reached HardwareFingerprint as a negative capacity.
    import turbofit_runtime.hardware as hw

    _host(monkeypatch, sysconf={"SC_PHYS_PAGES": -1, "SC_PAGE_SIZE": 4096}, meminfo=MEMINFO_128)

    assert hw._system_ram_mb() == 131072


def test_system_ram_is_zero_when_every_source_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    import turbofit_runtime.hardware as hw

    _host(monkeypatch, sysconf=None, meminfo=None)

    assert hw._system_ram_mb() == 0


def test_windows_ram_does_not_read_a_failed_call_as_a_zero_byte_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    import turbofit_runtime.hardware as hw

    monkeypatch.setattr(ctypes, "windll", _FakeWindll(ok=False, total_bytes=0), raising=False)
    assert hw._windows_ram_bytes() == 0

    monkeypatch.setattr(ctypes, "windll", _FakeWindll(ok=True, total_bytes=68 * 1024**3),
                        raising=False)
    assert hw._windows_ram_bytes() == 68 * 1024**3


def test_windows_ram_is_reached_when_sysconf_has_no_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The ctypes path was gated on hasattr(os, "sysconf") alone, so it was unreachable on every
    # POSIX host -- including one whose sysconf could not answer.
    import ctypes

    import turbofit_runtime.hardware as hw

    _host(monkeypatch, sysconf={}, meminfo=None, windows=True)
    monkeypatch.setattr(ctypes, "windll", _FakeWindll(ok=True, total_bytes=68 * 1024**3),
                        raising=False)

    assert hw._system_ram_mb() == 68 * 1024


def test_probe_gb10_survives_a_libc_without_the_sysconf_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The end-to-end symptom: on a GB10 the GPU's whole capacity comes from the RAM probe, so a
    # libc whose sysconf lacks SC_PHYS_PAGES took probe_hardware down instead of reading 128 GB.
    _host(monkeypatch, sysconf={}, meminfo=MEMINFO_128)

    def runner(command: list[str]) -> str:
        if command[0] == "nvidia-smi":
            return "0, GPU-gb10, NVIDIA GB10, [N/A], 12.1, 00000000:01:00.0\n"
        raise FileNotFoundError(command[0])

    fingerprint = probe_hardware(command_runner=runner, os_name="linux", architecture="aarch64")

    assert fingerprint.system_ram_mb == 131072
    assert fingerprint.devices[0].memory_total_mb == 131072
    assert fingerprint.memory_pool_kind == "unified"
