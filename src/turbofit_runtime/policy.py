"""Pure adaptive rung transition policy with hysteresis and rollback plans."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from .fit_check import TierProfile, fit_tier, rung_is_banned, rung_tokens
from .runtime_profile import Turbofile


class ActionKind(str, Enum):
    NONE = "none"
    ACTIVATE = "activate"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class AdaptiveState:
    current_index: int
    last_stable_index: int | None = None
    pending_index: int | None = None
    target_ceiling_index: int = 0
    deficit_since: float | None = None
    surplus_since: float | None = None
    cooldown_until: float = 0
    failure_times: tuple[float, ...] = ()
    quarantine_until: float = 0

    def __post_init__(self) -> None:
        for name in ("current_index", "target_ceiling_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.last_stable_index is None:
            object.__setattr__(self, "last_stable_index", self.current_index)
        for name in ("last_stable_index", "pending_index"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        for name in ("deficit_since", "surplus_since"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("cooldown_until", "quarantine_until"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.failure_times
        ):
            raise ValueError("failure_times must be finite and non-negative")
        object.__setattr__(self, "failure_times", tuple(self.failure_times))


@dataclass(frozen=True)
class CapacitySnapshot:
    available_mb_per_card: tuple[int, ...]
    required_mb_by_rung: tuple[tuple[int, ...], ...]
    activation_succeeded: bool = False
    activation_failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_mb_per_card", tuple(self.available_mb_per_card))
        object.__setattr__(
            self,
            "required_mb_by_rung",
            tuple(tuple(item) for item in self.required_mb_by_rung),
        )
        if not self.available_mb_per_card:
            raise ValueError("available_mb_per_card must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.available_mb_per_card
        ):
            raise ValueError("available capacity must be non-negative integers")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for rung in self.required_mb_by_rung
            for value in rung
        ):
            raise ValueError("rung requirements must be non-negative integers")
        if self.activation_succeeded and self.activation_failed:
            raise ValueError("activation cannot both succeed and fail")


@dataclass(frozen=True)
class ActionPlan:
    action: ActionKind
    target_index: int | None
    reason: str
    state: AdaptiveState


def reconcile(
    state: AdaptiveState,
    snapshot: CapacitySnapshot,
    profile: Turbofile,
    now: float,
) -> ActionPlan:
    _validate_inputs(state, snapshot, profile, now)
    if state.pending_index is not None:
        return _resolve_pending(state, snapshot, profile, now)

    if now < state.cooldown_until:
        return _none(state, "cooldown active")

    current_requirement = snapshot.required_mb_by_rung[state.current_index]
    if not _fits(current_requirement, snapshot.available_mb_per_card, margin_mb=0):
        contracted_state = replace(state, surplus_since=None)
        if contracted_state.deficit_since is None:
            return _none(replace(contracted_state, deficit_since=now), "deficit dwell started")
        elapsed = now - contracted_state.deficit_since
        if elapsed < profile.policy.contraction_dwell_s:
            return _none(contracted_state, "deficit dwell active")
        target = _contraction_target(state.current_index, snapshot, profile)
        if target is None:
            return _none(contracted_state, "no contraction rung fits")
        planned = replace(
            contracted_state,
            pending_index=target,
            last_stable_index=state.current_index,
            deficit_since=None,
        )
        return ActionPlan(ActionKind.ACTIVATE, target, "sustained capacity deficit", planned)

    stable_state = replace(state, deficit_since=None)
    if state.current_index <= state.target_ceiling_index:
        return _none(replace(stable_state, surplus_since=None), "target ceiling reached")
    if now < state.quarantine_until:
        return _none(replace(stable_state, surplus_since=None), "expansion quarantine active")

    margin_mb = round(profile.policy.expansion_margin_gb_per_card * 1024)
    if profile.policy.expansion_mode == "direct":
        target = _expansion_fit_target(state.current_index, snapshot, profile, margin_mb)
    else:
        target = state.current_index - 1
    if target is None or target >= state.current_index:
        return _none(replace(stable_state, surplus_since=None), "no expansion target fits")
    requirement = snapshot.required_mb_by_rung[target]
    if not _fits(requirement, snapshot.available_mb_per_card, margin_mb=margin_mb):
        return _none(replace(stable_state, surplus_since=None), "expansion margin unavailable")
    if stable_state.surplus_since is None:
        return _none(replace(stable_state, surplus_since=now), "surplus dwell started")
    elapsed = now - stable_state.surplus_since
    if elapsed < profile.policy.expansion_dwell_s:
        return _none(stable_state, "surplus dwell active")
    planned = replace(
        stable_state,
        pending_index=target,
        last_stable_index=state.current_index,
        surplus_since=None,
    )
    return ActionPlan(ActionKind.ACTIVATE, target, "sustained expansion headroom", planned)


def _resolve_pending(
    state: AdaptiveState,
    snapshot: CapacitySnapshot,
    profile: Turbofile,
    now: float,
) -> ActionPlan:
    if snapshot.activation_succeeded:
        target = state.pending_index
        assert target is not None
        committed = replace(
            state,
            current_index=target,
            last_stable_index=target,
            pending_index=None,
            deficit_since=None,
            surplus_since=None,
            cooldown_until=now + profile.policy.cooldown_s,
        )
        return _none(committed, "activation committed")
    if snapshot.activation_failed:
        window = profile.policy.flap_window_s if profile.policy.flap_window_s is not None else 300
        limit = profile.policy.flap_failure_limit or 3
        failures = tuple(value for value in state.failure_times if value >= now - window) + (now,)
        quarantine_until = state.quarantine_until
        if len(failures) >= limit:
            quarantine_until = max(quarantine_until, now + window)
        rolled_back = replace(
            state,
            current_index=state.last_stable_index,
            pending_index=None,
            deficit_since=None,
            surplus_since=None,
            cooldown_until=now + profile.policy.cooldown_s,
            failure_times=failures,
            quarantine_until=quarantine_until,
        )
        return ActionPlan(
            ActionKind.ROLLBACK,
            state.last_stable_index,
            "activation failed; rollback required",
            rolled_back,
        )
    return _none(state, "activation pending")


def _contraction_target(
    current_index: int,
    snapshot: CapacitySnapshot,
    profile: Turbofile,
) -> int | None:
    """Live re-fit: select the rung for the tier that fits current usable memory.

    Usable memory comes from the pressure probe (CapacitySnapshot's
    available_mb_per_card). The fit check maps it to a Fit List tier band;
    the deepest-fitting rung carrying that tier's model identity is the
    target. Banned models (Bonsai, 9B) are never selected. Falls back to
    the pre-baked rung ladder only for profiles without tier-identified
    rungs.
    """
    usable_gb = sum(snapshot.available_mb_per_card) / 1024
    tier = fit_tier(usable_gb)
    if tier is not None:
        matching = _rung_indices_for_tier(profile, tier)
        for index in sorted(matching, reverse=True):
            if index == current_index:
                continue
            if _fits(
                snapshot.required_mb_by_rung[index],
                snapshot.available_mb_per_card,
                margin_mb=0,
            ):
                return index
    return _first_unbanned_fitting_contraction(
        current_index,
        snapshot.required_mb_by_rung,
        snapshot.available_mb_per_card,
        tuple(rung.id for rung in profile.rungs),
    )


def _expansion_fit_target(
    current_index: int,
    snapshot: CapacitySnapshot,
    profile: Turbofile,
    margin_mb: int,
) -> int | None:
    """Direct expansion: re-fit straight to the best tier that fits, with margin."""
    usable_gb = (sum(snapshot.available_mb_per_card) - margin_mb) / 1024
    tier = fit_tier(usable_gb)
    if tier is None:
        return current_index - 1
    best: int | None = None
    for index in sorted(_rung_indices_for_tier(profile, tier)):
        if index >= current_index:
            break
        if _fits(
            snapshot.required_mb_by_rung[index],
            snapshot.available_mb_per_card,
            margin_mb=margin_mb,
        ):
            best = index
    return best


def _rung_indices_for_tier(profile: Turbofile, tier: TierProfile) -> tuple[int, ...]:
    rung_ids = tuple(rung.id for rung in profile.rungs)
    indices: list[int] = []
    for index, rung_id in enumerate(rung_ids):
        if rung_is_banned(rung_id):
            continue
        if set(tier.model_tokens) & rung_tokens(rung_id):
            indices.append(index)
    return tuple(indices)


def _first_fitting_contraction(
    current_index: int,
    requirements: tuple[tuple[int, ...], ...],
    available: tuple[int, ...],
) -> int | None:
    for index in range(current_index + 1, len(requirements)):
        if _fits(requirements[index], available, margin_mb=0):
            return index
    return None


def _first_unbanned_fitting_contraction(
    current_index: int,
    requirements: tuple[tuple[int, ...], ...],
    available: tuple[int, ...],
    rung_ids: tuple[str, ...],
) -> int | None:
    for index in range(current_index + 1, len(requirements)):
        if rung_is_banned(rung_ids[index]):
            continue
        if _fits(requirements[index], available, margin_mb=0):
            return index
    return None


def _fits(requirement: tuple[int, ...], available: tuple[int, ...], margin_mb: int) -> bool:
    if not requirement:
        return True
    if len(requirement) != len(available):
        return False
    return all(
        required + margin_mb <= capacity
        for required, capacity in zip(requirement, available, strict=True)
    )


def _validate_inputs(
    state: AdaptiveState,
    snapshot: CapacitySnapshot,
    profile: Turbofile,
    now: float,
) -> None:
    count = len(profile.rungs)
    if len(snapshot.required_mb_by_rung) != count:
        raise ValueError("required_mb_by_rung must match profile rung count")
    for name, index in (
        ("current_index", state.current_index),
        ("last_stable_index", state.last_stable_index),
        ("target_ceiling_index", state.target_ceiling_index),
    ):
        if index is None or not 0 <= index < count:
            raise ValueError(f"{name} is outside the rung ladder")
    if state.pending_index is not None and not 0 <= state.pending_index < count:
        raise ValueError("pending_index is outside the rung ladder")
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
        raise ValueError("now must be finite and non-negative")
    for name in ("deficit_since", "surplus_since"):
        value = getattr(state, name)
        if value is not None and value > now:
            raise ValueError(f"{name} cannot be later than now")
    if any(value > now for value in state.failure_times):
        raise ValueError("failure_times cannot be later than now")


def _none(state: AdaptiveState, reason: str) -> ActionPlan:
    return ActionPlan(ActionKind.NONE, None, reason, state)
