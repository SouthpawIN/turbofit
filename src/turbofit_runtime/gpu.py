"""Per-card GPU admission and mandatory clear gates."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class GPUSample:
    gpu: int
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: int

    def to_dict(self) -> dict:
        return {
            "gpu": self.gpu,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "utilization_pct": self.utilization_pct,
        }


@dataclass(frozen=True)
class GPUClearEvent:
    timestamp: str
    label: str
    passed: bool
    ceilings_mb: dict[int, int]
    snapshot: tuple[GPUSample, ...]
    samples_observed: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "passed": self.passed,
            "ceilings_mb": self.ceilings_mb,
            "snapshot": [sample.to_dict() for sample in self.snapshot],
            "samples_observed": self.samples_observed,
        }


class GPUClearTimeout(RuntimeError):
    def __init__(self, event: GPUClearEvent):
        self.event = event
        super().__init__(f"GPU clear gate timed out: {event.to_dict()}")


@dataclass(frozen=True)
class CardFit:
    gpu: int
    required_mb: int
    budget_mb: int
    fits: bool


@dataclass(frozen=True)
class FitResult:
    fits: bool
    cards: dict[int, CardFit]
    reason: str


def parse_nvidia_memory_csv(raw: str) -> tuple[GPUSample, ...]:
    rows = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        values = [int(item.strip()) for item in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"expected five NVIDIA memory columns: {line}")
        gpu, total, used, free, utilization = values
        rows.append(GPUSample(
            gpu=gpu,
            total_mb=total,
            used_mb=used,
            free_mb=free,
            utilization_pct=utilization,
        ))
    return tuple(rows)


def probe_gpus() -> tuple[GPUSample, ...]:
    raw = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return parse_nvidia_memory_csv(raw)


class GPUClearGate:
    def __init__(
        self,
        *,
        sample_fn: Callable[[], Sequence[GPUSample]] = probe_gpus,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sample = sample_fn
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

    def wait(
        self,
        *,
        ceilings_mb: Mapping[int, int],
        settle_samples: int = 3,
        timeout_s: float = 180,
        poll_s: float = 1,
        label: str,
    ) -> GPUClearEvent:
        if settle_samples < 1:
            raise ValueError("settle_samples must be >= 1")
        started = self._monotonic()
        observed = 0
        consecutive = 0
        snapshot: tuple[GPUSample, ...] = ()
        while self._monotonic() - started <= timeout_s:
            snapshot = tuple(self._sample())
            observed += 1
            clear = bool(snapshot) and all(
                sample.used_mb <= ceilings_mb.get(sample.gpu, 1024)
                for sample in snapshot
            )
            consecutive = consecutive + 1 if clear else 0
            if consecutive >= settle_samples:
                return GPUClearEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    label=label,
                    passed=True,
                    ceilings_mb=dict(ceilings_mb),
                    snapshot=snapshot,
                    samples_observed=observed,
                )
            self._sleep(poll_s)
        event = GPUClearEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            label=label,
            passed=False,
            ceilings_mb=dict(ceilings_mb),
            snapshot=snapshot,
            samples_observed=observed,
        )
        raise GPUClearTimeout(event)


def fit_per_card(
    requirements_mb: Mapping[int, int],
    snapshot: Sequence[GPUSample],
    *,
    safety_floor_mb: int = 1024,
    live: bool,
) -> FitResult:
    samples = {sample.gpu: sample for sample in snapshot}
    cards: dict[int, CardFit] = {}
    reasons = []
    for gpu, required in requirements_mb.items():
        sample = samples.get(gpu)
        available = (sample.free_mb if live else sample.total_mb) if sample else 0
        budget = max(0, available - safety_floor_mb)
        fits = required <= budget
        cards[gpu] = CardFit(gpu=gpu, required_mb=required, budget_mb=budget, fits=fits)
        operator = "<=" if fits else ">"
        reasons.append(f"GPU{gpu} {required} MiB {operator} {budget} MiB")
    return FitResult(
        fits=bool(cards) and all(card.fits for card in cards.values()),
        cards=cards,
        reason="; ".join(reasons),
    )
