"""Classify per-card VRAM pressure without targeting external processes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CardMemory:
    gpu: int
    total_mb: int
    used_mb: int

    def __post_init__(self) -> None:
        if self.gpu < 0:
            raise ValueError("gpu must be non-negative")
        if self.total_mb <= 0:
            raise ValueError("total_mb must be positive")
        if self.used_mb < 0 or self.used_mb > self.total_mb:
            raise ValueError("used_mb must be between zero and total_mb")


@dataclass(frozen=True)
class GPUProcess:
    gpu: int
    pid: int
    used_mb: int
    name: str
    desktop: bool = False

    def __post_init__(self) -> None:
        if self.gpu < 0:
            raise ValueError("gpu must be non-negative")
        if self.pid <= 0:
            raise ValueError("pid must be positive")
        if self.used_mb < 0:
            raise ValueError("used_mb must be non-negative")
        if not self.name:
            raise ValueError("name must be non-empty")


@dataclass(frozen=True)
class CardPressure:
    gpu: int
    total_mb: int
    observed_used_mb: int
    managed_mb: int
    external_mb: int
    desktop_mb: int
    safety_reserve_mb: int
    reservation_mb: int
    available_for_managed_mb: int


@dataclass(frozen=True)
class PressureSnapshot:
    cards: tuple[CardPressure, ...]
    process_data_available: bool
    release_targets: tuple[int, ...]


def measure_pressure(
    *,
    cards: Sequence[CardMemory],
    processes: Sequence[GPUProcess] | None,
    managed_pids: set[int] | frozenset[int],
    runtime_pids: set[int] | frozenset[int],
    managed_resident_mb: Mapping[int, int],
    desktop_baseline_mb: Mapping[int, int],
    safety_reserve_mb: Mapping[int, int],
    reservations_mb: Mapping[int, int],
) -> PressureSnapshot:
    ordered_cards = tuple(sorted(cards, key=lambda item: item.gpu))
    if not ordered_cards:
        raise ValueError("cards must not be empty")
    if len({card.gpu for card in ordered_cards}) != len(ordered_cards):
        raise ValueError("duplicate GPU card")
    known_gpus = {card.gpu for card in ordered_cards}
    _validate_budget_mapping(managed_resident_mb, known_gpus, "managed_resident_mb")
    _validate_budget_mapping(desktop_baseline_mb, known_gpus, "desktop_baseline_mb")
    _validate_budget_mapping(safety_reserve_mb, known_gpus, "safety_reserve_mb")
    _validate_budget_mapping(reservations_mb, known_gpus, "reservations_mb")

    owned_pids = set(managed_pids) | set(runtime_pids)
    process_data_available = processes is not None
    process_rows = tuple(processes or ())
    for process in process_rows:
        if process.gpu not in known_gpus:
            raise ValueError(f"process references unknown GPU {process.gpu}")

    release_targets = tuple(
        sorted({process.pid for process in process_rows if process.pid in owned_pids})
    )
    measured: list[CardPressure] = []
    for card in ordered_cards:
        card_processes = tuple(row for row in process_rows if row.gpu == card.gpu)
        managed_observed = sum(
            row.used_mb for row in card_processes if row.pid in owned_pids
        )
        managed = max(managed_observed, managed_resident_mb.get(card.gpu, 0))
        desktop_observed = sum(
            row.used_mb
            for row in card_processes
            if row.desktop and row.pid not in owned_pids
        )
        desktop = max(desktop_baseline_mb.get(card.gpu, 0), desktop_observed)
        external_observed = sum(
            row.used_mb
            for row in card_processes
            if row.pid not in owned_pids and not row.desktop
        )
        accounted = managed + desktop + external_observed
        unexplained = max(0, card.used_mb - accounted)
        external = external_observed + unexplained
        safety = safety_reserve_mb.get(card.gpu, 0)
        reservation = reservations_mb.get(card.gpu, 0)
        available = max(0, card.total_mb - external - desktop - safety - reservation)
        measured.append(
            CardPressure(
                gpu=card.gpu,
                total_mb=card.total_mb,
                observed_used_mb=card.used_mb,
                managed_mb=managed,
                external_mb=external,
                desktop_mb=desktop,
                safety_reserve_mb=safety,
                reservation_mb=reservation,
                available_for_managed_mb=available,
            )
        )
    return PressureSnapshot(
        cards=tuple(measured),
        process_data_available=process_data_available,
        release_targets=release_targets if process_data_available else (),
    )


def _validate_budget_mapping(
    values: Mapping[int, int], known_gpus: set[int], name: str
) -> None:
    for gpu, value in values.items():
        if gpu not in known_gpus:
            raise ValueError(f"{name} references unknown GPU {gpu}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}[{gpu}] must be a non-negative integer")
