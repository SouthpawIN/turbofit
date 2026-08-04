"""Stable, topology-aware hardware identity for runtime recommendations."""
from __future__ import annotations

import csv
import os
import platform
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

NVIDIA_INVENTORY_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,compute_cap,pci.bus_id",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class AcceleratorDevice:
    index: int
    uuid: str
    name: str
    vendor: str
    backend: str
    memory_total_mb: int
    compute_capability: str | None
    bus_id: str | None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("accelerator index must be a non-negative integer")
        if (
            isinstance(self.memory_total_mb, bool)
            or not isinstance(self.memory_total_mb, int)
            or self.memory_total_mb <= 0
        ):
            raise ValueError("memory_total_mb must be a positive integer")
        for name in ("uuid", "name", "vendor", "backend"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True)
class HardwareFingerprint:
    os: str
    architecture: str
    system_ram_mb: int
    devices: tuple[AcceleratorDevice, ...] = ()

    def __post_init__(self) -> None:
        if not self.os or not self.architecture:
            raise ValueError("os and architecture must be non-empty")
        if (
            isinstance(self.system_ram_mb, bool)
            or not isinstance(self.system_ram_mb, int)
            or self.system_ram_mb <= 0
        ):
            raise ValueError("system_ram_mb must be a positive integer")
        ordered = tuple(sorted(self.devices, key=lambda item: (item.bus_id or "", item.index)))
        for field in ("index", "uuid"):
            values = [getattr(device, field) for device in ordered]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate accelerator {field}")
        bus_ids = [device.bus_id for device in ordered if device.bus_id is not None]
        if len(bus_ids) != len(set(bus_ids)):
            raise ValueError("duplicate accelerator bus_id")
        object.__setattr__(self, "devices", ordered)

    @property
    def total_vram_mb(self) -> int:
        return sum(device.memory_total_mb for device in self.devices)

    @property
    def vendors(self) -> tuple[str, ...]:
        return tuple(sorted({device.vendor for device in self.devices}))

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(sorted({device.backend for device in self.devices}))

    @property
    def topology_key(self) -> str:
        if not self.devices:
            return "no-accelerator"
        counts = Counter(device.memory_total_mb for device in self.devices)
        return "+".join(
            f"{counts[memory]}x{memory}mb" for memory in sorted(counts)
        )

    @property
    def recommendation_key(self) -> str:
        backends = "+".join(self.backends) or "none"
        capabilities = "+".join(
            sorted(
                {
                    device.compute_capability
                    for device in self.devices
                    if device.compute_capability
                }
            )
        ) or "unknown"
        return (
            f"{self.os}:{self.architecture}:{backends}:{self.topology_key}:"
            f"cap-{capabilities}:ram-{self.system_ram_mb}mb"
        )


def parse_nvidia_inventory_csv(raw: str) -> tuple[AcceleratorDevice, ...]:
    devices: list[AcceleratorDevice] = []
    for row_number, row in enumerate(csv.reader(StringIO(raw)), start=1):
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 6:
            raise ValueError(
                f"expected six NVIDIA inventory columns on row {row_number}"
            )
        index, uuid, name, memory_total, compute_capability, bus_id = (
            item.strip() for item in row
        )
        try:
            parsed_index = int(index)
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA numeric field on row {row_number}") from exc
        # DGX Spark (GB10) and other unified-memory parts report memory.total
        # as "[N/A]" via nvidia-smi -- the GPU shares system RAM, so fall back
        # to that. Mirrors the existing [N/A] handling for compute_capability
        # and bus_id just below.
        if memory_total in {"", "N/A", "[N/A]"}:
            parsed_memory = _system_ram_mb()
        else:
            try:
                parsed_memory = int(memory_total)
            except ValueError as exc:
                raise ValueError(
                    f"invalid NVIDIA numeric field on row {row_number}"
                ) from exc
        devices.append(
            AcceleratorDevice(
                index=parsed_index,
                uuid=uuid,
                name=name,
                vendor="nvidia",
                backend="cuda",
                memory_total_mb=parsed_memory,
                compute_capability=None
                if compute_capability in {"", "N/A", "[N/A]"}
                else compute_capability,
                bus_id=None if bus_id in {"", "N/A", "[N/A]"} else bus_id,
            )
        )
    return tuple(sorted(devices, key=lambda item: (item.bus_id or "", item.index)))


def probe_hardware(
    *,
    command_runner: Callable[[list[str]], str] | None = None,
    os_name: str | None = None,
    architecture: str | None = None,
    system_ram_mb: int | None = None,
    kernel_release: str | None = None,
) -> HardwareFingerprint:
    runner = command_runner or _run_command
    normalized_os = (os_name or platform.system()).lower()
    normalized_architecture = (architecture or platform.machine()).lower()
    memory_mb = system_ram_mb or _system_ram_mb()
    release = (kernel_release or platform.release()).lower()
    try:
        devices = parse_nvidia_inventory_csv(runner(list(NVIDIA_INVENTORY_QUERY)))
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        devices = ()
    if not devices and normalized_os == "darwin" and normalized_architecture in {"arm64", "aarch64"}:
        # Metal shares system memory with macOS. Reserve 8 GiB so profile matching
        # never treats memory required by the OS as fully allocatable model VRAM.
        usable_unified_memory_mb = max(1, memory_mb - 8 * 1024)
        devices = (
            AcceleratorDevice(
                index=0,
                uuid="apple-unified-memory",
                name="Apple Silicon Unified Memory",
                vendor="apple",
                backend="metal",
                memory_total_mb=usable_unified_memory_mb,
                compute_capability=None,
                bus_id=None,
            ),
        )
    if normalized_os == "linux" and ("microsoft" in release or os.getenv("WSL_INTEROP")):
        normalized_os = "windows-wsl2"
    return HardwareFingerprint(
        os=normalized_os,
        architecture=normalized_architecture,
        system_ram_mb=memory_mb,
        devices=devices,
    )


def _run_command(
    command: list[str],
    *,
    check_output: Callable[..., str] | None = None,
    compatibility_library_dir: Callable[[], str | None] | None = None,
) -> str:
    run = check_output or subprocess.check_output
    try:
        return run(command, text=True, timeout=5, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        if not command or Path(command[0]).name != "nvidia-smi":
            raise
        find_compatibility_dir = compatibility_library_dir or _nvidia_compatibility_library_dir
        library_dir = find_compatibility_dir()
        if not library_dir:
            raise
        env = os.environ.copy()
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = (
            f"{library_dir}:{current}" if current else library_dir
        )
        return run(
            command,
            text=True,
            timeout=5,
            stderr=subprocess.PIPE,
            env=env,
        )


def _nvidia_compatibility_library_dir() -> str | None:
    override = os.getenv("TURBOFIT_NVIDIA_COMPAT_LIB_DIR")
    if override and Path(override).is_dir():
        return override
    try:
        raw_version = Path("/proc/driver/nvidia/version").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"Kernel Module\s+(\d+\.\d+(?:\.\d+)?)", raw_version)
    if not match:
        return None
    candidate = (
        Path.home()
        / ".local"
        / "lib"
        / f"nvidia-{match.group(1)}"
        / "usr"
        / "lib"
        / "x86_64-linux-gnu"
    )
    return str(candidate) if candidate.is_dir() else None


def _system_ram_mb() -> int:
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count // (1024 * 1024))
    # Windows fallback via ctypes GlobalMemoryStatusEx
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullTotalPhys // (1024 * 1024))
