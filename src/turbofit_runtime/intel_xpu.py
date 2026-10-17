"""Intel XPU probes via xpu-smi (Level Zero) for Arc discrete GPUs.

``xpu-smi`` mirrors the ``nvidia-smi`` surface Intel GPUs need: a single
``--query-gpu`` inventory/memory query and a ``ps`` subcommand with per-process
device memory. This module is the Intel counterpart of the CUDA probe; the
Level Zero surface is strictly more honest than ``vulkaninfo`` device-local
heap sizes, which report an integrated GPU's shared-system-memory aperture as
dedicated VRAM.

Integrated parts (Intel iGPUs) are identified by their PCI BDF on domain 0 /
bus 00 and excluded from the accelerator inventory: their "memory" is host
RAM counted once through ``host_usable_memory_mb``.
"""
from __future__ import annotations

import re
import subprocess
from typing import Callable

from .hardware import AcceleratorDevice
from .pressure import CardMemory, GPUProcess

XPU_INVENTORY_QUERY = [
    "xpu-smi",
    "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]
XPU_MEMORY_QUERY = [
    "xpu-smi",
    "--query-gpu=uuid,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]
XPU_PROCESS_QUERY = ["xpu-smi", "ps"]
XPU_GPU_MEMORY_QUERY = [
    "xpu-smi",
    "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,"
    "utilization.gpu",
    "--format=csv,noheader,nounits",
]

INTEGRATED_BDF_PREFIX = "0000:00:"
DESKTOP_PROCESS_TOKENS = ("xorg", "wayland", "gnome-shell", "kwin", "hyprland", "quickshell")

# xpu-smi `ps` row: PID Command Type DeviceID SHR MEM Cmp% Rnd% Med% Cpy%
# MEM is in KiB; Command may contain spaces, so the row is matched from both
# ends with the free-form command captured between.
_PS_ROW = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<command>.+?)\s+(?P<type>\S+)\s+(?P<device>\d+)\s+"
    r"(?P<shr>\d+)\s+(?P<mem>\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s*$"
)


def _mib_to_int(raw: str) -> int:
    """xpu-smi reports fractional MiB; every downstream model is integer."""
    return int(round(float(raw.strip())))


class IntelXpuDevice:
    """One discovered Intel device plus the discrete/integrated classification."""

    __slots__ = ("device", "integrated")

    def __init__(self, device: AcceleratorDevice, integrated: bool) -> None:
        self.device = device
        self.integrated = integrated


def is_integrated_bdf(bus_id: str) -> bool:
    return bus_id.startswith(INTEGRATED_BDF_PREFIX)


def parse_xpu_inventory(raw: str) -> tuple[IntelXpuDevice, ...]:
    """Parse ``--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version``.

    Raises ValueError when the payload carries no identifiable Intel devices so
    callers can distinguish "xpu-smi answered" from "xpu-smi is not this box's
    tool".
    """
    rows: list[IntelXpuDevice] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"invalid xpu-smi inventory row: {line}")
        index_text, uuid, name, bus_id, total_text, _driver = fields
        try:
            index = int(index_text)
            total_mb = _mib_to_int(total_text)
        except ValueError as exc:
            raise ValueError(f"invalid xpu-smi inventory numeric field: {line}") from exc
        if not uuid or not name or not bus_id:
            raise ValueError(f"incomplete xpu-smi inventory row: {line}")
        rows.append(IntelXpuDevice(
            AcceleratorDevice(
                index=index,
                uuid=uuid,
                name=name,
                vendor="intel",
                backend="vulkan",
                memory_total_mb=total_mb,
                compute_capability=None,
                bus_id=bus_id,
            ),
            integrated=is_integrated_bdf(bus_id),
        ))
    if not rows:
        raise ValueError("xpu-smi inventory is empty")
    return tuple(sorted(rows, key=lambda item: (item.device.bus_id or "", item.device.index)))


def probe_xpu_devices(
    command_runner: Callable[[list[str]], str] | None = None,
) -> tuple[IntelXpuDevice, ...]:
    return parse_xpu_inventory((command_runner or _run)(list(XPU_INVENTORY_QUERY)))


def parse_xpu_cards(raw: str, devices: tuple[AcceleratorDevice, ...]) -> tuple[CardMemory, ...]:
    """Map uuid-keyed memory rows onto the fingerprint's discrete devices.

    Devices the payload does not carry (the integrated part) never appear; a
    fingerprint device missing from the payload is a probe mismatch. Order
    follows ``devices`` so per-card requirements zip correctly downstream.
    """
    by_uuid = {device.uuid: device for device in devices}
    remaining = dict(by_uuid)
    cards: dict[int, CardMemory] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"invalid xpu-smi memory row: {line}")
        uuid, total_text, used_text = fields
        device = remaining.pop(uuid, None)
        if device is None:
            continue
        try:
            total = _mib_to_int(total_text)
            used = _mib_to_int(used_text)
        except ValueError as exc:
            raise ValueError(f"invalid xpu-smi memory field: {line}") from exc
        if total <= 0:
            raise ValueError(f"xpu-smi reported non-positive total for {uuid}")
        cards[device.index] = CardMemory(
            gpu=device.index, total_mb=total, used_mb=min(max(0, used), total)
        )
    if len(cards) != len(devices):
        raise ValueError("xpu-smi pressure inventory does not match hardware fingerprint")
    return tuple(cards[index] for index in sorted(cards))


def parse_xpu_processes(raw: str, devices: tuple[AcceleratorDevice, ...]) -> tuple[GPUProcess, ...]:
    """Attribute xpu-smi `ps` rows (MEM in KiB) to fingerprint devices by index."""
    by_index = {device.index: device for device in devices}
    processes: list[GPUProcess] = []
    for line in raw.splitlines():
        match = _PS_ROW.match(line)
        if match is None:
            continue
        index = int(match.group("device"))
        if index not in by_index:
            continue  # integrated device or out of scope
        pid = int(match.group("pid"))
        used_mb = max(0, round(int(match.group("mem")) / 1024))
        name = match.group("command").strip()
        lowered = name.lower()
        desktop = any(token in lowered for token in DESKTOP_PROCESS_TOKENS)
        processes.append(GPUProcess(
            gpu=by_index[index].index, pid=pid, used_mb=used_mb, name=name, desktop=desktop
        ))
    return tuple(processes)


def probe_xpu_pressure(
    devices: tuple[AcceleratorDevice, ...],
    command_runner: Callable[[list[str]], str] | None = None,
) -> tuple[tuple[CardMemory, ...], tuple[GPUProcess, ...] | None]:
    """Return per-card memory plus processes; None processes when `ps` failed."""
    cards = parse_xpu_cards((command_runner or _run)(list(XPU_MEMORY_QUERY)), devices)
    try:
        processes: tuple[GPUProcess, ...] | None = parse_xpu_processes(
            (command_runner or _run)(list(XPU_PROCESS_QUERY)), devices
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        processes = None
    return cards, processes


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=5, stderr=subprocess.PIPE)
