from __future__ import annotations

import pytest

from turbofit_runtime.gpu import (
    GPUClearGate,
    GPUClearTimeout,
    GPUSample,
    fit_per_card,
    parse_nvidia_memory_csv,
)


def sample(gpu: int, used: int, free: int = 24_000, total: int = 24_576) -> GPUSample:
    return GPUSample(gpu=gpu, used_mb=used, free_mb=free, total_mb=total, utilization_pct=0)


def test_parse_nvidia_memory_csv() -> None:
    raw = "0, 24576, 530, 24046, 0\n1, 24576, 57, 24519, 2\n"

    parsed = parse_nvidia_memory_csv(raw)

    assert parsed == (
        GPUSample(gpu=0, total_mb=24576, used_mb=530, free_mb=24046, utilization_pct=0),
        GPUSample(gpu=1, total_mb=24576, used_mb=57, free_mb=24519, utilization_pct=2),
    )


def test_clear_gate_requires_three_consecutive_clear_samples() -> None:
    samples = iter([
        (sample(0, 800), sample(1, 100)),
        (sample(0, 700), sample(1, 90)),
        (sample(0, 650), sample(1, 80)),
    ])
    gate = GPUClearGate(sample_fn=lambda: next(samples), sleep_fn=lambda _: None)

    event = gate.wait(ceilings_mb={0: 1024, 1: 1024}, settle_samples=3, timeout_s=10, label="test")

    assert event.passed is True
    assert event.samples_observed == 3
    assert event.snapshot[0].used_mb == 650


def test_clear_gate_resets_streak_after_vram_spike() -> None:
    samples = iter([
        (sample(0, 800), sample(1, 100)),
        (sample(0, 700), sample(1, 90)),
        (sample(0, 3000), sample(1, 80)),
        (sample(0, 650), sample(1, 80)),
        (sample(0, 640), sample(1, 70)),
        (sample(0, 630), sample(1, 60)),
    ])
    gate = GPUClearGate(sample_fn=lambda: next(samples), sleep_fn=lambda _: None)

    event = gate.wait(ceilings_mb={0: 1024, 1: 1024}, settle_samples=3, timeout_s=10, label="spike")

    assert event.samples_observed == 6
    assert event.snapshot[0].used_mb == 630


def test_clear_gate_timeout_reports_last_snapshot() -> None:
    clock = iter([0.0, 0.5, 1.0, 1.5, 2.0])
    gate = GPUClearGate(
        sample_fn=lambda: (sample(0, 3000), sample(1, 2000)),
        sleep_fn=lambda _: None,
        monotonic_fn=lambda: next(clock),
    )

    with pytest.raises(GPUClearTimeout) as error:
        gate.wait(ceilings_mb={0: 1024, 1: 1024}, settle_samples=2, timeout_s=1, label="busy")

    assert error.value.event.passed is False
    assert error.value.event.snapshot[0].used_mb == 3000


def test_per_card_fit_never_uses_aggregate_vram() -> None:
    snapshot = (
        GPUSample(gpu=0, total_mb=24576, used_mb=20000, free_mb=4576, utilization_pct=0),
        GPUSample(gpu=1, total_mb=24576, used_mb=1000, free_mb=23576, utilization_pct=0),
    )

    result = fit_per_card({0: 6000, 1: 12000}, snapshot, safety_floor_mb=1024, live=True)

    assert result.fits is False
    assert result.cards[0].fits is False
    assert result.cards[1].fits is True
    assert "GPU0" in result.reason


def test_clear_card_fit_uses_total_minus_safety_floor() -> None:
    snapshot = (sample(0, used=500, total=24576), sample(1, used=100, total=24576))

    result = fit_per_card({0: 23000, 1: 23553}, snapshot, safety_floor_mb=1024, live=False)

    assert result.cards[0].fits is True
    assert result.cards[0].budget_mb == 23552
    assert result.cards[1].fits is False
