from __future__ import annotations

from dataclasses import replace

import pytest

from test_runtime_profile import DIGEST_A, valid_mapping
from turbofit_runtime.policy import (
    ActionKind,
    AdaptiveState,
    CapacitySnapshot,
    reconcile,
)
from turbofit_runtime.runtime_profile import Turbofile


def profile() -> Turbofile:
    mapping = valid_mapping()
    mapping["policy"].update(
        contraction_dwell_s=10,
        expansion_dwell_s=20,
        expansion_margin_gb_per_card=1,
        cooldown_s=30,
        flap_failure_limit=2,
        flap_window_s=60,
    )
    mapping["rungs"] = [
        {
            "id": "quality-262k",
            "context": 262144,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "1" * 64,
            "main_manifest": DIGEST_A,
        },
        {
            "id": "lean-128k",
            "context": 131072,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "2" * 64,
            "main_manifest": DIGEST_A,
        },
        {
            "id": "api",
            "context": 131072,
            "aux_mode": "api",
            "evidence": "sha256:" + "3" * 64,
            "main_api_policy": "api:auto",
            "aux_api_policy": "api:auto",
        },
    ]
    return Turbofile.from_mapping(mapping)


def snapshot(
    available: int,
    *,
    succeeded: bool = False,
    failed: bool = False,
) -> CapacitySnapshot:
    return CapacitySnapshot(
        available_mb_per_card=(available,),
        required_mb_by_rung=((20000,), (12000,), ()),
        activation_succeeded=succeeded,
        activation_failed=failed,
    )


def test_single_deficit_sample_does_not_contract() -> None:
    item = profile()
    first = reconcile(AdaptiveState(current_index=0), snapshot(15000), item, now=0)
    second = reconcile(first.state, snapshot(15000), item, now=9)

    assert first.action is ActionKind.NONE
    assert second.action is ActionKind.NONE
    assert second.state.deficit_since == 0


def test_sustained_deficit_contracts_to_first_fitting_rung() -> None:
    item = profile()
    state = reconcile(AdaptiveState(current_index=0), snapshot(15000), item, now=0).state

    plan = reconcile(state, snapshot(15000), item, now=10)

    assert plan.action is ActionKind.ACTIVATE
    assert plan.target_index == 1
    assert plan.state.pending_index == 1


def test_severe_deficit_skips_directly_to_api_rung() -> None:
    item = profile()
    state = reconcile(AdaptiveState(current_index=0), snapshot(5000), item, now=0).state

    plan = reconcile(state, snapshot(5000), item, now=10)

    assert plan.target_index == 2


def test_expansion_is_one_rung_and_requires_margin_and_dwell() -> None:
    item = profile()
    state = AdaptiveState(current_index=2, last_stable_index=2)
    first = reconcile(state, snapshot(14000), item, now=0)
    before_dwell = reconcile(first.state, snapshot(14000), item, now=19)
    plan = reconcile(before_dwell.state, snapshot(14000), item, now=20)

    assert first.action is ActionKind.NONE
    assert before_dwell.action is ActionKind.NONE
    assert plan.action is ActionKind.ACTIVATE
    assert plan.target_index == 1

    no_margin = reconcile(
        AdaptiveState(current_index=2, last_stable_index=2), snapshot(12500), item, now=0
    )
    assert no_margin.state.surplus_since is None


def test_target_ceiling_prevents_expansion_above_operator_limit() -> None:
    item = profile()
    state = AdaptiveState(current_index=1, last_stable_index=1, target_ceiling_index=1)

    plan = reconcile(state, snapshot(24576), item, now=100)

    assert plan.action is ActionKind.NONE
    assert plan.state.surplus_since is None


def test_cooldown_blocks_new_transition() -> None:
    item = profile()
    state = AdaptiveState(
        current_index=1,
        last_stable_index=1,
        target_ceiling_index=0,
        cooldown_until=100,
        surplus_since=50,
    )

    plan = reconcile(state, snapshot(24576), item, now=90)

    assert plan.action is ActionKind.NONE
    assert "cooldown" in plan.reason


def test_activation_success_commits_pending_rung_and_starts_cooldown() -> None:
    item = profile()
    state = AdaptiveState(current_index=0, last_stable_index=0, pending_index=1)

    plan = reconcile(state, snapshot(15000, succeeded=True), item, now=50)

    assert plan.action is ActionKind.NONE
    assert plan.state.current_index == 1
    assert plan.state.last_stable_index == 1
    assert plan.state.pending_index is None
    assert plan.state.cooldown_until == 80


def test_failed_activation_rolls_back_and_repeated_failures_quarantine() -> None:
    item = profile()
    pending = AdaptiveState(current_index=0, last_stable_index=0, pending_index=1)

    first = reconcile(pending, snapshot(15000, failed=True), item, now=10)
    assert first.action is ActionKind.ROLLBACK
    assert first.target_index == 0
    assert first.state.failure_times == (10,)

    second_pending = replace(first.state, pending_index=1, cooldown_until=0)
    second = reconcile(second_pending, snapshot(15000, failed=True), item, now=20)
    assert second.action is ActionKind.ROLLBACK
    assert second.state.quarantine_until == 80

    blocked = reconcile(
        replace(second.state, current_index=2, pending_index=None, cooldown_until=0),
        snapshot(24576),
        item,
        now=30,
    )
    assert blocked.action is ActionKind.NONE
    assert "quarantine" in blocked.reason


def test_replay_is_deterministic() -> None:
    item = profile()
    samples = [(0, 15000), (10, 15000)]

    def replay() -> list[tuple[ActionKind, int | None, AdaptiveState]]:
        state = AdaptiveState(current_index=0)
        output = []
        for now, available in samples:
            plan = reconcile(state, snapshot(available), item, now=now)
            output.append((plan.action, plan.target_index, plan.state))
            state = plan.state
        return output

    assert replay() == replay()


@pytest.mark.parametrize("now", [float("nan"), float("inf"), float("-inf"), True])
def test_reconcile_rejects_nonfinite_or_boolean_time(now: float) -> None:
    with pytest.raises(ValueError, match="now must be finite"):
        reconcile(AdaptiveState(current_index=0), snapshot(24576), profile(), now=now)


def test_state_rejects_future_history_and_capacity_rejects_booleans() -> None:
    item = profile()
    with pytest.raises(ValueError, match="deficit_since cannot be later"):
        reconcile(
            AdaptiveState(current_index=0, deficit_since=11),
            snapshot(24576),
            item,
            now=10,
        )
    with pytest.raises(ValueError, match="available capacity"):
        CapacitySnapshot(
            available_mb_per_card=(True,),
            required_mb_by_rung=((1,), (1,), ()),
        )
