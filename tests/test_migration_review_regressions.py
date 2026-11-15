"""Controller-owned regressions for independently reviewed findings."""
import socket
from types import SimpleNamespace
import pytest
from test_native_lifecycle import lifecycle
from test_native_identity import backend
from test_native_residency_processes import component
from test_gateway_compatibility import GATEWAY
from turbofit_runtime.native_lifecycle import LifecycleError
from turbofit_runtime.reconciler import transition, ReconcileError


def test_failed_wake_has_bounded_cooldown(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    attempts = []
    def fail():
        attempts.append(1)
        raise RuntimeError('synthetic load failure')
    with pytest.raises(RuntimeError):
        lc.wake('main', fail)
    for _ in range(5):
        with pytest.raises(LifecycleError, match='cooldown'):
            lc.wake('main', fail)
    assert len(attempts) == 1
    now[0] += 5
    with pytest.raises(RuntimeError):
        lc.wake('main', fail)
    assert len(attempts) == 2


def test_real_transition_publishes_current_rung_before_wake(tmp_path, monkeypatch):
    runtime = backend(tmp_path)
    monkeypatch.setattr(runtime, 'activate_local', lambda rung: None)
    monkeypatch.setattr(runtime, 'verify_rung', lambda rung: True)
    updated = transition(runtime.current_state, 1, runtime.profile, runtime)
    assert runtime.current_state == updated
    assert updated.rung_index == 1
    observed = []
    def inspect(role, item, context):
        observed.append((role, item, context))
        raise LookupError('stop before model launch')
    monkeypatch.setattr(runtime, '_component', inspect)
    with pytest.raises(LookupError, match='before model'):
        runtime.ensure_role('main')
    assert observed[0][2] == runtime.profile.rungs[1].context
    assert observed[0][1] == runtime._roles(runtime.profile.rungs[1].id)['main']


def test_occupied_foreign_port_is_not_adopted_or_replaced(tmp_path):
    runtime = backend(tmp_path)
    proposed = component('main')
    with socket.socket() as owner:
        owner.bind(('127.0.0.1', proposed.port))
        owner.listen()
        with pytest.raises(ReconcileError, match='occupied'):
            runtime._start(proposed)
        assert not runtime._owned
        assert owner.fileno() != -1


def test_cloud_routes_cannot_inherit_local_budget_policy():
    assert GATEWAY.reasoning_policy_for({
        'is_api': True, 'alias': 'api-fallback',
        'reasoning_policy': {'hard_cap': 4096},
    }) is None
