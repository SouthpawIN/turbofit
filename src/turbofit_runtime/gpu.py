"""Per-card GPU admission and mandatory clear gates."""
from __future__ import annotations

import subprocess
import json
import platform
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
    if platform.system() == "Windows":
        return probe_windows_gpus()
    raw = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return parse_nvidia_memory_csv(raw)


def parse_windows_gpu_memory(raw: str) -> tuple[GPUSample, ...]:
    """Parse PowerShell's adapter inventory plus dedicated usage counters."""
    payload = json.loads(raw)
    adapters = payload.get("adapters") or []
    usage = payload.get("usage") or []
    if not isinstance(adapters, list) or not isinstance(usage, list) or not adapters:
        raise ValueError("Windows GPU probe returned no adapters")
    if len(usage) < len(adapters):
        raise ValueError("Windows GPU probe returned incomplete usage counters")
    samples = []
    for index, adapter in enumerate(adapters):
        total_bytes = int(adapter.get("AdapterRAM") or 0)
        used_bytes = int(usage[index].get("CookedValue") or 0)
        if total_bytes <= 0:
            raise ValueError("Windows GPU adapter has no usable memory size")
        samples.append(GPUSample(
            gpu=index,
            total_mb=total_bytes // (1024 * 1024),
            used_mb=used_bytes // (1024 * 1024),
            free_mb=max(0, total_bytes - used_bytes) // (1024 * 1024),
            utilization_pct=0,
        ))
    return tuple(samples)


def probe_windows_gpus() -> tuple[GPUSample, ...]:
    command = (
        "$adapters = @(Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.AdapterRAM -gt 0 } | "
        "Select-Object Name,AdapterRAM); "
        "$usage = @(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' "
        "-ErrorAction Stop | Select-Object -ExpandProperty CounterSamples | "
        "Select-Object InstanceName,CookedValue); "
        "[PSCustomObject]@{adapters=$adapters;usage=$usage} | "
        "ConvertTo-Json -Compress"
    )
    raw = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True,
    )
    return parse_windows_gpu_memory(raw)


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

    def sample_now(self) -> tuple[GPUSample, ...]:
        """Read one instantaneous sample for baseline-relative campaigns."""
        return tuple(self._sample())

    def wait(
        self,
        *,
        ceilings_mb: Mapping[int, int],
        baseline_mb: Mapping[int, int] | None = None,
        baseline_margin_mb: int = 256,
        settle_samples: int = 3,
        timeout_s: float = 180,
        poll_s: float = 1,
        label: str,
    ) -> GPUClearEvent:
        if settle_samples < 1:
            raise ValueError("settle_samples must be >= 1")
        if baseline_margin_mb < 0:
            raise ValueError("baseline_margin_mb must be >= 0")
        effective_ceilings = {
            gpu: max(
                int(ceiling),
                int(baseline_mb[gpu]) + baseline_margin_mb,
            )
            if baseline_mb is not None and gpu in baseline_mb
            else int(ceiling)
            for gpu, ceiling in ceilings_mb.items()
        }
        started = self._monotonic()
        observed = 0
        consecutive = 0
        snapshot: tuple[GPUSample, ...] = ()
        while self._monotonic() - started <= timeout_s:
            snapshot = tuple(self._sample())
            observed += 1
            clear = bool(snapshot) and all(
                sample.used_mb <= effective_ceilings.get(sample.gpu, 1024)
                for sample in snapshot
            )
            consecutive = consecutive + 1 if clear else 0
            if consecutive >= settle_samples:
                return GPUClearEvent(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    label=label,
                    passed=True,
                    ceilings_mb=effective_ceilings,
                    snapshot=snapshot,
                    samples_observed=observed,
                )
            self._sleep(poll_s)
        event = GPUClearEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            label=label,
            passed=False,
            ceilings_mb=effective_ceilings,
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
