"""Bounded NVIDIA pressure probe feeding the ownership-aware pressure model."""
from __future__ import annotations

import csv
import subprocess
from io import StringIO
from typing import Any, Callable

from .hardware import HardwareFingerprint
from .pressure import CardMemory, GPUProcess, PressureSnapshot, measure_pressure

CARD_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used",
    "--format=csv,noheader,nounits",
]
PROCESS_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
    "--format=csv,noheader,nounits",
]


def probe_nvidia_pressure(
    hardware: HardwareFingerprint,
    *,
    manager_status: dict[str, Any],
    managed_required_mb: tuple[int, ...],
    command_runner: Callable[[list[str]], str] | None = None,
    desktop_baseline_mb: int = 256,
    safety_reserve_mb: int = 512,
) -> PressureSnapshot:
    if len(managed_required_mb) not in {0, len(hardware.devices)}:
        raise ValueError("managed requirements must be empty or match device count")
    if desktop_baseline_mb < 0 or safety_reserve_mb < 0:
        raise ValueError("pressure reserves must be non-negative")
    runner = command_runner or _run
    cards = _parse_cards(runner(list(CARD_QUERY)))
    if len(cards) != len(hardware.devices):
        raise RuntimeError("NVIDIA pressure inventory does not match hardware fingerprint")
    uuid_to_index = {device.uuid: device.index for device in hardware.devices}
    try:
        processes = _parse_processes(runner(list(PROCESS_QUERY)), uuid_to_index)
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        processes = None
    manager_pids = frozenset(_collect_pids(manager_status))
    required = managed_required_mb or tuple(0 for _ in hardware.devices)
    managed_resident = {
        device.index: value for device, value in zip(hardware.devices, required, strict=True)
    }
    desktop = {device.index: desktop_baseline_mb for device in hardware.devices}
    safety = {device.index: safety_reserve_mb for device in hardware.devices}
    empty = {device.index: 0 for device in hardware.devices}
    return measure_pressure(
        cards=cards,
        processes=processes,
        managed_pids=set(manager_pids),
        turbohaul_pids=set(manager_pids),
        managed_resident_mb=managed_resident,
        desktop_baseline_mb=desktop,
        safety_reserve_mb=safety,
        reservations_mb=empty,
    )


def _parse_cards(raw: str) -> tuple[CardMemory, ...]:
    cards: list[CardMemory] = []
    for number, row in enumerate(csv.reader(StringIO(raw)), start=1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 3:
            raise ValueError(f"invalid NVIDIA card row {number}")
        try:
            index, total, used = (int(value.strip()) for value in row)
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA card numeric field on row {number}") from exc
        cards.append(CardMemory(index, total, used))
    if not cards:
        raise ValueError("NVIDIA pressure inventory is empty")
    return tuple(cards)


def _parse_processes(raw: str, uuid_to_index: dict[str, int]) -> tuple[GPUProcess, ...]:
    processes: list[GPUProcess] = []
    for number, row in enumerate(csv.reader(StringIO(raw)), start=1):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != 4:
            raise ValueError(f"invalid NVIDIA process row {number}")
        uuid, pid_text, used_text, name = (value.strip() for value in row)
        if uuid not in uuid_to_index:
            raise ValueError(f"unknown NVIDIA UUID on process row {number}")
        try:
            pid, used = int(pid_text), int(used_text)
        except ValueError as exc:
            raise ValueError(f"invalid NVIDIA process numeric field on row {number}") from exc
        lowered = name.lower()
        desktop = any(token in lowered for token in ("xorg", "wayland", "gnome-shell", "kwin"))
        processes.append(GPUProcess(uuid_to_index[uuid], pid, used, name, desktop))
    return tuple(processes)


def _collect_pids(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, list):
        for item in value:
            found.extend(_collect_pids(item))
    elif isinstance(value, dict):
        pid = value.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            found.append(pid)
        for item in value.values():
            found.extend(_collect_pids(item))
    return found


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, timeout=5)
