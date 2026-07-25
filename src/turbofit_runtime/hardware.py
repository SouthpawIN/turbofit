"""Stable, topology-aware hardware identity for runtime recommendations."""
from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections import Counter
from dataclasses import dataclass
from io import StringIO
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
            parsed_memory = int(memory_total)
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA numeric field on row {row_number}") from exc
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
) -> HardwareFingerprint:
    runner = command_runner or _run_command
    try:
        devices = parse_nvidia_inventory_csv(runner(list(NVIDIA_INVENTORY_QUERY)))
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        devices = ()
    return HardwareFingerprint(
        os=(os_name or platform.system()).lower(),
        architecture=(architecture or platform.machine()).lower(),
        system_ram_mb=system_ram_mb or _system_ram_mb(),
        devices=devices,
    )


def _run_command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=5)


def _system_ram_mb() -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count // (1024 * 1024))
