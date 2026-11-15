"""Deterministic residency tests. HTTP backends are explicitly synthetic fixtures."""
import concurrent.futures
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request
import pytest
from turbofit_runtime.native_lifecycle import (
    IdleLifecycle, LifecycleEndpoint, LifecycleError, LifecycleUnavailable,
    lifecycle_request, load_endpoint,
)
from test_gateway_compatibility import fake_stack, post, GATEWAY


def lifecycle(tmp_path, now=None, **kwargs):
    now = now or [1000.0]
    return IdleLifecycle(state_dir=tmp_path, clock=lambda: now[0],
                         wall=lambda: 100000 + now[0], **kwargs)


def test_full_idle_window_and_tokens_protect_inflight(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    assert lc.idle_roles() == ()
    a, b = lc.acquire("main"), lc.acquire("main")
    now[0] += 200
    assert lc.idle_roles() == ("aux",)
    lc.release(a)
    lc.release(a)  # idempotent release cannot remove another request's lease
    assert lc.lease_count("main") == 1
    lc.release(b)
    now[0] += 119
    assert lc.idle_roles() == ("aux",)
    now[0] += 1
    assert lc.idle_roles() == ("main", "aux")


def test_restart_uses_wall_clock_not_previous_boot_monotonic(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    lc.release(lc.acquire("main"))
    restart = IdleLifecycle(state_dir=tmp_path, clock=lambda: 3.0,
                            wall=lambda: 101060.0)
    assert restart.idle_roles() == ()
    restart.clock = lambda: 63.0
    restart.wall = lambda: 101120.0
    assert restart.idle_roles() == ("main", "aux")


@pytest.mark.parametrize("corrupt", [False, True])
def test_crash_with_lease_or_corrupt_state_fails_closed(tmp_path, corrupt):
    lc = lifecycle(tmp_path)
    lc.acquire("main")
    if corrupt:
        (tmp_path / "lifecycle-state.json").write_text("{broken")
    restart = lifecycle(tmp_path)
    assert restart.idle_roles() == ()
    with pytest.raises(LifecycleError):
        restart.acquire("main")
    with restart.controller_tick() as allowed:
        assert not allowed


def test_abandoned_lease_does_not_orphan_restart(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    token = lc.acquire("main")
    assert lc.lease_count("main") == 1
    # A dead holder never releases; a day later a fresh owner must proceed.
    now[0] += 25 * 3600
    restart = lifecycle(tmp_path, now)
    assert restart.acquire("main")
    assert token not in restart._leases


def test_legacy_unstamped_lease_still_fails_closed(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    lc.acquire("main")
    state_file = tmp_path / "lifecycle-state.json"
    data = json.loads(state_file.read_text())
    data["leases"] = {token: "main" for token in data["leases"]}
    state_file.write_text(json.dumps(data))
    restart = lifecycle(tmp_path, now)
    with pytest.raises(LifecycleError):
        restart.acquire("main")


def test_singleton(tmp_path):
    first, second = lifecycle(tmp_path), lifecycle(tmp_path)
    first.acquire_singleton()
    try:
        with pytest.raises(LifecycleError):
            second.acquire_singleton()
    finally:
        first.release_singleton()
    second.acquire_singleton()
    second.release_singleton()


@pytest.mark.parametrize("fails", [False, True])
def test_coalesced_wake_finishes_and_propagates_failure(tmp_path, fails):
    lc = lifecycle(tmp_path, wake_timeout_s=2)
    entered, finish, joined = threading.Event(), threading.Event(), threading.Event()
    starts = []
    def start():
        starts.append(1)
        entered.set()
        assert finish.wait(2)
        if fails:
            raise RuntimeError("load failed")
    def waiter():
        joined.set()
        return lc.wake("main", start)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lc.wake, "main", start)
        assert entered.wait(1)
        second = pool.submit(waiter)
        assert joined.wait(1)
        # Waiting must release the lock; this acquisition detects the old deadlock.
        assert lc._lock.acquire(timeout=0.3)
        lc._lock.release()
        finish.set()
        if fails:
            with pytest.raises(RuntimeError):
                first.result(timeout=2)
            with pytest.raises(LifecycleError):
                second.result(timeout=2)
        else:
            first.result(timeout=2)
            second.result(timeout=2)
    assert starts == [1]
    assert not lc.status()["roles"]["main"]["waking"]


def test_idle_transition_holds_admission_lock(tmp_path):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    now[0] += 121
    held, finish, admitted = threading.Event(), threading.Event(), threading.Event()
    def stop(role):
        held.set()
        assert finish.wait(2)
        return True
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        freeing = pool.submit(lc.release_idle, stop)
        assert held.wait(1)
        admission = pool.submit(lambda: (lc.acquire("main"), admitted.set()))
        assert not admitted.wait(0.05)
        finish.set()
        freeing.result(timeout=2)
        admission.result(timeout=2)


def test_endpoint_auth_validation_and_failed_wake_release(tmp_path):
    lc = lifecycle(tmp_path)
    endpoint = LifecycleEndpoint(lc, ensure_ready=lambda role: (_ for _ in ()).throw(RuntimeError()))
    endpoint.start()
    try:
        ep = load_endpoint(tmp_path)
        assert (tmp_path / "lifecycle-endpoint.json").stat().st_mode & 0o777 == 0o600
        with pytest.raises(LifecycleUnavailable):
            lifecycle_request(ep, {"action": "acquire", "role": "main"})
        assert lc.lease_count("main") == 0
        with pytest.raises(LifecycleUnavailable):
            lifecycle_request({**ep, "token": "wrong"}, {"action": "status"})
        with pytest.raises(LifecycleUnavailable):
            lifecycle_request(ep, {"action": "acquire", "role": "ghost"})
    finally:
        endpoint.stop()
    with pytest.raises(LifecycleUnavailable):
        load_endpoint(tmp_path)
    with pytest.raises(ValueError):
        LifecycleEndpoint(lc, host="0.0.0.0")


@pytest.mark.parametrize("stream", [False, True])
def test_actual_gateway_holds_lease_through_upstream_response(tmp_path, monkeypatch, stream, fake_stack):
    lc = lifecycle(tmp_path / "native")
    starts = []
    endpoint = LifecycleEndpoint(lc, ensure_ready=lambda role: starts.append(role))
    endpoint.start()
    monkeypatch.setenv("TURBOFIT_NATIVE_STATE", str(lc.state_dir))
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "1")
    try:
        client, seen = fake_stack
        original = GATEWAY.GatewayHandler._proxy_to
        def observe(self, *args, **kwargs):
            assert lc.lease_count("main") == 1
            with lc.controller_tick() as allowed:
                assert not allowed
            result = original(self, *args, **kwargs)
            assert lc.lease_count("main") == 1
            return result
        monkeypatch.setattr(GATEWAY.GatewayHandler, "_proxy_to", observe)
        response = post(client, {"model": "active:main", "messages": [], "stream": stream})
        try:
            body = response.read()
            assert response.status == 200
            if stream:
                assert b"[DONE]" in body
        finally:
            response.close()
        assert starts == ["main"]
        assert lc.lease_count("main") == 0
    finally:
        endpoint.stop()


def test_configured_missing_controller_returns_503_without_backend_request(tmp_path, monkeypatch, fake_stack):
    monkeypatch.setenv("TURBOFIT_NATIVE_STATE", str(tmp_path / "missing"))
    monkeypatch.setenv("TURBOFIT_LIFECYCLE_REQUIRED", "1")
    client, seen = fake_stack
    response = post(client, {"model": "active:main", "messages": []})
    assert response.status == 503
    response.read()
    assert "payload" not in seen


def test_shared_aux_lease_protects_main(tmp_path):
    lc = lifecycle(tmp_path)
    starts = []
    endpoint = LifecycleEndpoint(lc, ensure_ready=starts.append, resolve_role=lambda role: "main")
    endpoint.start()
    try:
        ep = load_endpoint(tmp_path)
        lease = lifecycle_request(ep, {"action": "acquire", "role": "aux"})
        assert lease["role"] == "main" and starts == ["main"]
        assert lc.lease_count("main") == 1 and lc.lease_count("aux") == 0
        lifecycle_request(ep, {"action": "release", "token": lease["token"]})
    finally:
        endpoint.stop()


def controller_module():
    path = Path(__file__).parents[1] / "scripts/turbofit-controller"
    loader = importlib.machinery.SourceFileLoader("controller_residency_test", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_real_controller_tick_defers_during_request_then_calls_idle_release(tmp_path, monkeypatch):
    mod = controller_module()
    lc = lifecycle(tmp_path)
    controller = SimpleNamespace(state=SimpleNamespace(adaptive=SimpleNamespace(current_index=0)),
                                 requirements=SimpleNamespace(required_mb_by_rung=[1]))
    calls = []
    runtime = SimpleNamespace(observe_residency=lambda owner: None, synchronize=lambda *args: controller,
                              tick=lambda *args, **kw: calls.append("tick"),
                              release_idle_residency=lambda owner: calls.append("release"))
    monkeypatch.setattr(mod, "probe_hardware", lambda: None)
    monkeypatch.setattr(mod, "probe_accelerator_pressure", lambda *a, **k: None)
    args = SimpleNamespace(selection=tmp_path / "selection", runtime_state_dir=tmp_path)
    token = lc.acquire("main")
    mod.run_tick(runtime, args, lc)
    assert calls == []
    lc.release(token)
    mod.run_tick(runtime, args, lc)
    assert calls == ["tick", "release"]
    with pytest.raises(SystemExit):
        mod.parse_args(["--once", "--idle-release", "120"])
