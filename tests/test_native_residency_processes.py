"""Real child process/native backend tests with synthetic HTTP, NOT model inference."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import pytest
from test_native_identity import backend
from turbofit_runtime.native_lifecycle import IdleLifecycle, LifecycleEndpoint, lifecycle_request, load_endpoint
from turbofit_runtime.reconciler import ReconcileError


def component(role, unhealthy=False):
    with socket.socket() as holder:
        holder.bind(('127.0.0.1', 0))
        port = holder.getsockname()[1]
    script = Path(__file__).parent / 'fixtures/native_http_child.py'
    command = [sys.executable, str(script), '--alias', role, '--port', str(port)]
    if unhealthy:
        command.append('--unhealthy')
    return SimpleNamespace(role=role, command=tuple(command), alias=role,
                           port=port, model_path=None, projector_path=None, gpu='')


def test_real_native_children_idle_stop_and_coalesced_request_wake(tmp_path, monkeypatch):
    from dataclasses import replace

    from turbofit_runtime.runtime_profile import AuxMode

    runtime = backend(tmp_path)
    # Two GPU lanes are only real on a dedicated-aux rung; a shared-main rung
    # deliberately serves aux from the main process.
    runtime.profile = replace(
        runtime.profile,
        rungs=tuple(replace(rung, aux_mode=AuxMode.DEDICATED) for rung in runtime.profile.rungs),
    )
    runtime.verification_timeout_s = 3
    components = {role: component(role) for role in ('main', 'aux')}
    monkeypatch.setattr(runtime, '_roles', lambda rung: {r: {} for r in components})
    monkeypatch.setattr(runtime, '_component', lambda role, item, context: components[role])
    now = [1000.0]
    lc = IdleLifecycle(state_dir=tmp_path / 'lifecycle', clock=lambda: now[0], idle_timeout_s=1)
    lc.acquire_singleton()
    endpoint = LifecycleEndpoint(lc, ensure_ready=runtime.ensure_role)
    endpoint.start()
    processes = []
    try:
        ep = load_endpoint(lc.state_dir)
        with ThreadPoolExecutor(max_workers=2) as pool:
            leases = list(pool.map(lambda _: lifecycle_request(ep, {'action': 'acquire', 'role': 'main'}), range(2)))
        first = runtime._owned['main']
        processes.append(first.process)
        assert first.process.poll() is None
        aux = lifecycle_request(ep, {'action': 'acquire', 'role': 'aux'})
        processes.append(runtime._owned['aux'].process)
        assert first.pid != runtime._owned['aux'].pid
        now[0] += 2
        lc.release_idle(runtime.stop_role)
        assert all(p.poll() is None for p in processes)
        for lease in [*leases, aux]:
            lifecycle_request(ep, {'action': 'release', 'token': lease['token']})
        now[0] += 2
        lc.release_idle(runtime.stop_role)
        assert all(p.poll() is not None for p in processes)
        assert not runtime._owned
        second_lease = lifecycle_request(ep, {'action': 'acquire', 'role': 'main'})
        processes.append(runtime._owned['main'].process)
        assert runtime._owned['main'].pid != first.pid
        lifecycle_request(ep, {'action': 'release', 'token': second_lease['token']})
    finally:
        endpoint.stop()
        runtime.reset_managed()
        lc.release_singleton()
        for process in processes:
            process.wait(timeout=3)
    assert all(p.poll() is not None for p in processes)


def test_real_native_health_timeout_reclaims_only_failed_child(tmp_path, monkeypatch):
    runtime = backend(tmp_path)
    runtime.verification_timeout_s = 0.4
    child = component('main', unhealthy=True)
    monkeypatch.setattr(runtime, '_roles', lambda rung: {'main': {}})
    monkeypatch.setattr(runtime, '_component', lambda *args: child)
    processes = []
    start = runtime._start
    def track(value):
        result = start(value)
        processes.append(result.process)
        return result
    monkeypatch.setattr(runtime, '_start', track)
    try:
        with pytest.raises(ReconcileError, match='verification'):
            runtime.ensure_role('main')
        assert not runtime._owned
        assert processes and processes[0].poll() is not None
    finally:
        runtime.reset_managed()
        for process in processes:
            process.wait(timeout=3)
