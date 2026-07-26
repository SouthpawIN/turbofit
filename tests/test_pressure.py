from __future__ import annotations

from turbofit_runtime.pressure import CardMemory, GPUProcess, measure_pressure


def test_managed_and_turbohaul_memory_is_not_external_demand() -> None:
    snapshot = measure_pressure(
        cards=(CardMemory(gpu=0, total_mb=24576, used_mb=18000),),
        processes=(
            GPUProcess(gpu=0, pid=10, used_mb=14000, name="llama-main"),
            GPUProcess(gpu=0, pid=11, used_mb=2000, name="turbohaul-sidecar"),
        ),
        managed_pids={10},
        turbohaul_pids={11},
        managed_resident_mb={0: 16000},
        desktop_baseline_mb={0: 1000},
        safety_reserve_mb={0: 1000},
        reservations_mb={},
    )

    card = snapshot.cards[0]
    assert card.managed_mb == 16000
    assert card.external_mb == 1000
    assert card.available_for_managed_mb == 21576
    assert snapshot.release_targets == (10, 11)


def test_unrelated_gpu_process_reduces_budget_but_is_never_an_action_target() -> None:
    snapshot = measure_pressure(
        cards=(CardMemory(0, 24576, 12000),),
        processes=(
            GPUProcess(0, 20, 8000, "external-training"),
            GPUProcess(0, 21, 3000, "turbofit-main"),
        ),
        managed_pids={21},
        turbohaul_pids=set(),
        managed_resident_mb={0: 3000},
        desktop_baseline_mb={0: 1000},
        safety_reserve_mb={0: 1024},
        reservations_mb={},
    )

    card = snapshot.cards[0]
    assert card.external_mb == 8000
    assert card.available_for_managed_mb == 14552
    assert 20 not in snapshot.release_targets
    assert snapshot.release_targets == (21,)


def test_desktop_baseline_and_inflight_reservation_are_reserved_per_card() -> None:
    snapshot = measure_pressure(
        cards=(CardMemory(0, 24576, 1200), CardMemory(1, 24576, 500)),
        processes=(GPUProcess(0, 30, 700, "desktop", desktop=True),),
        managed_pids=set(),
        turbohaul_pids=set(),
        managed_resident_mb={},
        desktop_baseline_mb={0: 1200, 1: 800},
        safety_reserve_mb={0: 1024, 1: 1024},
        reservations_mb={0: 2048, 1: 4096},
    )

    first, second = snapshot.cards
    assert first.desktop_mb == 1200
    assert first.reservation_mb == 2048
    assert first.available_for_managed_mb == 20304
    assert second.available_for_managed_mb == 18656


def test_missing_process_data_degrades_to_conservative_used_memory_accounting() -> None:
    snapshot = measure_pressure(
        cards=(CardMemory(0, 24576, 10000),),
        processes=None,
        managed_pids={99},
        turbohaul_pids=set(),
        managed_resident_mb={0: 6000},
        desktop_baseline_mb={0: 1000},
        safety_reserve_mb={0: 1024},
        reservations_mb={},
    )

    card = snapshot.cards[0]
    assert snapshot.process_data_available is False
    assert card.managed_mb == 6000
    assert card.external_mb == 3000
    assert card.available_for_managed_mb == 19552
    assert snapshot.release_targets == ()


def test_pressure_never_aggregates_asymmetric_cards() -> None:
    snapshot = measure_pressure(
        cards=(CardMemory(0, 24576, 23000), CardMemory(1, 24576, 1000)),
        processes=(GPUProcess(0, 50, 22000, "external"),),
        managed_pids=set(),
        turbohaul_pids=set(),
        managed_resident_mb={},
        desktop_baseline_mb={0: 500, 1: 500},
        safety_reserve_mb={0: 1024, 1: 1024},
        reservations_mb={},
    )

    assert snapshot.cards[0].available_for_managed_mb == 552
    assert snapshot.cards[1].available_for_managed_mb == 22552


def test_invalid_card_or_process_data_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="total_mb"):
        CardMemory(0, 0, 0)
    with pytest.raises(ValueError, match="used_mb"):
        CardMemory(0, 100, 101)
    with pytest.raises(ValueError, match="used_mb"):
        GPUProcess(0, 1, -1, "bad")
