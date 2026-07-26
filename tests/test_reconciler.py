from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from test_runtime_profile import DIGEST_A, DIGEST_B, valid_mapping
from turbofit_runtime.reconciler import (
    ReconcileError,
    ReconcilerState,
    transition,
)
from turbofit_runtime.runtime_profile import Turbofile


DIGEST_C = "sha256:" + "c" * 64


def profile() -> Turbofile:
    mapping = valid_mapping()
    mapping["rungs"] = [
        {
            "id": "dedicated",
            "context": 262144,
            "aux_mode": "dedicated",
            "evidence": "sha256:" + "1" * 64,
            "main_manifest": DIGEST_A,
            "aux_manifest": DIGEST_B,
        },
        {
            "id": "shared-seam",
            "context": 262144,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "2" * 64,
            "main_manifest": DIGEST_A,
        },
        {
            "id": "lean",
            "context": 131072,
            "aux_mode": "shared-main",
            "evidence": "sha256:" + "3" * 64,
            "main_manifest": DIGEST_C,
        },
        {
            "id": "api",
            "context": 131072,
            "aux_mode": "api",
            "evidence": "sha256:" + "4" * 64,
            "main_api_policy": "api:main",
            "aux_api_policy": "api:aux",
        },
    ]
    return Turbofile.from_mapping(mapping)


@dataclass
class FakeBackend:
    events: list[object] = field(default_factory=list)
    drain_ok: bool = True
    unload_ok: bool = True
    verify_ok: bool = True
    restore_ok: bool = True
    owned: tuple[int, ...] = (10, 11)

    def reset_managed(self) -> None:
        self.events.append("reset-managed")

    def block_aux_admission(self) -> None:
        self.events.append("block-aux")

    def drain_aux(self, timeout_s: float) -> bool:
        self.events.append(("drain-aux", timeout_s))
        return self.drain_ok

    def clean_unload_aux(self) -> bool:
        self.events.append("unload-aux")
        return self.unload_ok

    def owned_pids(self) -> tuple[int, ...]:
        self.events.append("owned-pids")
        return self.owned

    def escalate_owned(self, pids: tuple[int, ...]) -> None:
        self.events.append(("escalate-owned", pids))
        self.unload_ok = True

    def activate_local(self, rung_id: str) -> None:
        self.events.append(("activate-local", rung_id))

    def activate_api(self, main_policy: str, aux_policy: str) -> None:
        self.events.append(("activate-api", main_policy, aux_policy))

    def route_aux_to_main(self) -> None:
        self.events.append("route-aux-to-main")

    def route_aux_dedicated(self) -> None:
        self.events.append("route-aux-dedicated")

    def verify_rung(self, rung_id: str) -> bool:
        self.events.append(("verify", rung_id))
        return self.verify_ok

    def publish_routes(self, state: ReconcilerState) -> None:
        self.events.append(("publish", state.rung_index, state.main_target, state.aux_target))

    def restore(self, state: ReconcilerState) -> None:
        self.events.append(("restore", state.rung_index))

    def verify_restore(self, state: ReconcilerState) -> bool:
        self.events.append(("verify-restore", state.rung_index))
        return self.restore_ok


def state(index: int = 0) -> ReconcilerState:
    return ReconcilerState(
        profile_id="quality-24gb",
        rung_index=index,
        main_target="local:main",
        aux_target="local:aux" if index == 0 else "local:main",
    )


def test_dedicated_to_shared_main_drains_aux_and_preserves_main_kv() -> None:
    backend = FakeBackend()

    result = transition(state(0), 1, profile(), backend, drain_timeout_s=30)

    assert result.rung_index == 1
    assert backend.events == [
        "block-aux",
        ("drain-aux", 30),
        "unload-aux",
        "route-aux-to-main",
        ("verify", "shared-seam"),
        ("publish", 1, "local:main", "local:main"),
    ]
    assert not any(event[0] == "activate-local" for event in backend.events if isinstance(event, tuple))


def test_shared_to_dedicated_activates_missing_aux_before_verification() -> None:
    backend = FakeBackend()

    result = transition(state(1), 0, profile(), backend)

    assert result.rung_index == 0
    assert backend.events == [
        ("activate-local", "dedicated"),
        "route-aux-dedicated",
        ("verify", "dedicated"),
        ("publish", 0, "local:main", "local:aux"),
    ]


def test_active_aux_drain_failure_rolls_back_without_publishing() -> None:
    backend = FakeBackend(drain_ok=False)

    with pytest.raises(ReconcileError, match="drain"):
        transition(state(0), 1, profile(), backend)

    assert ("restore", 0) in backend.events
    assert ("verify-restore", 0) in backend.events
    assert not any(event[0] == "publish" for event in backend.events if isinstance(event, tuple))


def test_context_or_main_change_activates_then_verifies_before_publish() -> None:
    backend = FakeBackend()

    result = transition(state(1), 2, profile(), backend)

    assert result.rung_index == 2
    assert backend.events == [
        ("activate-local", "lean"),
        "route-aux-to-main",
        ("verify", "lean"),
        ("publish", 2, "local:main", "local:main"),
    ]


def test_api_terminal_rung_activates_policies_and_publishes_after_verification() -> None:
    backend = FakeBackend()

    result = transition(state(2), 3, profile(), backend)

    assert result.main_target == "api:main"
    assert result.aux_target == "api:aux"
    assert backend.events == [
        ("activate-api", "api:main", "api:aux"),
        ("verify", "api"),
        ("publish", 3, "api:main", "api:aux"),
    ]


def test_verification_failure_restores_previous_state() -> None:
    backend = FakeBackend(verify_ok=False)

    with pytest.raises(ReconcileError, match="verification"):
        transition(state(1), 2, profile(), backend)

    assert backend.events[-2:] == [("restore", 1), ("verify-restore", 1)]
    assert not any(event[0] == "publish" for event in backend.events if isinstance(event, tuple))


def test_unload_escalation_targets_only_backend_reported_owned_pids() -> None:
    backend = FakeBackend(unload_ok=False, owned=(10, 11))

    transition(state(0), 1, profile(), backend)

    assert ("escalate-owned", (10, 11)) in backend.events
    assert ("escalate-owned", (99,)) not in backend.events


def test_noop_transition_has_no_backend_effects() -> None:
    backend = FakeBackend()
    current = state(1)

    assert transition(current, 1, profile(), backend) is current
    assert backend.events == []


@pytest.mark.parametrize("target, timeout", [(-1, 60), (4, 60), (1, -0.1)])
def test_transition_rejects_invalid_bounds_before_backend_effects(
    target: int, timeout: float
) -> None:
    backend = FakeBackend()

    with pytest.raises(ValueError):
        transition(state(0), target, profile(), backend, drain_timeout_s=timeout)
    assert backend.events == []
