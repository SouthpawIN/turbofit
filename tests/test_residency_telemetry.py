"""Owner-to-catalogue telemetry. Child servers are synthetic, never models."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_model_metadata import GATEWAY, catalogue
from test_native_identity import backend
from test_native_residency_processes import component
from turbofit_runtime.native_lifecycle import IdleLifecycle, LifecycleEndpoint


def observation(state='ready', alias='Exact/Main:Q4', context=4096):
    return {'main': {'backing_model': alias, 'context_length': context, 'residency': state}}


def test_owner_events_freshness_restart_and_recipe_change(tmp_path):
    now = [0.0]
    lc = IdleLifecycle(state_dir=tmp_path, clock=lambda: now[0], wall=lambda: 100 + now[0], idle_timeout_s=1)
    lc.observe(observation())
    assert lc.status()['roles']['main']['residency'] == 'ready'
    now[0] = 16
    assert lc.status()['roles']['main']['residency'] == 'unknown'
    assert lc.status()['roles']['main']['observed_at'] == 100
    lc.observe(observation())
    lc.release_idle(lambda role: True)
    assert lc.status()['roles']['main']['residency'] == 'idle'
    lc.observe(observation('unknown'))
    assert lc.status()['roles']['main']['residency'] == 'idle'
    entered, done = threading.Event(), threading.Event()
    def start():
        entered.set()
        assert done.wait(2)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(lc.wake, 'main', start)
        assert entered.wait(1)
        assert lc.status()['roles']['main']['residency'] == 'loading'
        now[0] += 16
        assert lc.status()['roles']['main']['residency'] == 'unknown'
        lc.observe(observation())  # Owner tick heartbeats an actual in-flight wake.
        assert lc.status()['roles']['main']['residency'] == 'loading'
        done.set()
        future.result(timeout=2)
    assert lc.status()['roles']['main']['residency'] == 'ready'
    def fail():
        raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError):
        lc.wake('main', fail)
    assert lc.status()['roles']['main']['residency'] == 'error'
    now[0] += 2
    lc.release_idle(lambda role: True)
    lc.observe(observation('unknown'))
    assert lc.status()['roles']['main']['residency'] == 'error'
    lc.observe(observation('unknown', context=8192))
    assert lc.status()['roles']['main']['residency'] == 'unknown'
    assert IdleLifecycle(state_dir=tmp_path).status()['roles']['main']['residency'] == 'unknown'


@pytest.mark.parametrize('change', [
    {'backing_model': 'wrong'}, {'context_length': 8192}, {'context_length': True},
    {'observed_at': 10**1000},
    {'freshness': {'age_s': 10**1000, 'max_age_s': 15, 'stale': False}},
    {'freshness': {'age_s': 0, 'max_age_s': 10**1000, 'stale': False}},
    {'freshness': {'age_s': 16, 'max_age_s': 15, 'stale': False}},
    {'freshness': {'age_s': float('nan'), 'max_age_s': 15, 'stale': False}},
    {'freshness': {'age_s': True, 'max_age_s': 15, 'stale': False}},
    {'freshness': {'age_s': 0, 'max_age_s': 15, 'stale': 'false'}},
])
def test_catalogue_rejects_unbound_or_invalid_freshness(catalogue, monkeypatch, change):
    from turbofit_runtime import native_lifecycle
    item = dict(observation()['main'], observed_at=100,
                freshness={'age_s': 0, 'max_age_s': 15, 'stale': False})
    item.update(change)
    monkeypatch.setenv('TURBOFIT_LIFECYCLE_REQUIRED', '1')
    monkeypatch.setattr(native_lifecycle, 'load_endpoint', lambda _: {})
    monkeypatch.setattr(native_lifecycle, 'lifecycle_request', lambda *a, **k: {'orphaned': False, 'roles': {'main': item}})
    model = GATEWAY.provider_models()[0]
    assert model['metadata']['residency'] == 'unknown'
    assert model['metadata']['backing_model'] == 'Exact/Main:Q4'


def test_real_child_owner_health_idle_failure_to_real_catalogue(catalogue, tmp_path, monkeypatch):
    runtime = backend(tmp_path)
    runtime.verification_timeout_s = 3
    child = component('main')
    monkeypatch.setattr(runtime, '_roles', lambda rung: {'main': {'context': 4096}})
    monkeypatch.setattr(runtime, '_component', lambda *args: child)
    state, routes = catalogue
    routes['main']['alias'] = 'main'
    routes['aux'] = {'kind': 'shared-main'}
    state.write_text(json.dumps({'active': 'fixture', 'routes': routes}))
    now = [0.0]
    lc = IdleLifecycle(state_dir=tmp_path/'lc', clock=lambda: now[0], idle_timeout_s=1)
    lc.observe(runtime.residency_snapshot())
    endpoint = LifecycleEndpoint(lc, ensure_ready=runtime.ensure_role)
    endpoint.start()
    monkeypatch.setenv('TURBOFIT_LIFECYCLE_REQUIRED', '1')
    monkeypatch.setenv('TURBOFIT_NATIVE_STATE', str(lc.state_dir))
    try:
        assert GATEWAY.provider_models()[0]['metadata']['residency'] == 'unknown'
        lc.wake('main', lambda: runtime.ensure_role('main'))
        lc.observe(runtime.residency_snapshot())
        assert all(m['metadata']['residency'] == 'ready' for m in GATEWAY.provider_models())
        resident = runtime._owned['main']
        resident.process.terminate()  # Only our fixture child, not an external PID.
        resident.process.wait(timeout=3)
        lc.observe(runtime.residency_snapshot())
        assert GATEWAY.provider_models()[0]['metadata']['residency'] == 'error'
        lc.wake('main', lambda: runtime.ensure_role('main'))
        lc.observe(runtime.residency_snapshot())
        now[0] = 2
        lc.release_idle(runtime.release_idle_role)
        lc.observe(runtime.residency_snapshot())
        before = dict(lc._observations)
        assert all(m['metadata']['residency'] == 'idle' for m in GATEWAY.provider_models())
        assert not runtime._owned
        assert lc._observations == before
        now[0] = 18
        models = GATEWAY.provider_models()
        assert all(m['metadata']['residency'] == 'unknown' for m in models)
        assert all(m['metadata']['backing_model'] == 'main' for m in models)
        # shared-main aux rides the main model: not a separate catalog
        # entry (the catalog keys on id only), but its identity must still
        # surface in the role metadata.
        aux_meta = GATEWAY.provider_model_metadata()['aux']
        assert aux_meta['mode'] == 'shared-main'
        assert aux_meta['backing_model'] == 'main'
    finally:
        endpoint.stop()
        runtime.reset_managed()


def test_reconstruction_through_real_service_preserves_crash_not_idle(tmp_path, monkeypatch):
    from test_runtime_service import service
    from test_selection import hardware
    from turbofit_runtime.selection import save_selection
    from turbofit_runtime.native_lifecycle import LifecycleError
    service_runtime, _ = service(tmp_path)
    hw = hardware(24576)
    choice = service_runtime.catalog.select(hw, requested='auto')
    selection = tmp_path/'selection.json'
    save_selection(selection, choice)
    service_runtime.synchronize(selection, hw)  # Persist real controller state.
    child = component('main')
    def factory(profile, state):
        native = backend(tmp_path)
        native.profile, native.current_state = profile, state
        native.verification_timeout_s = 3
        monkeypatch.setattr(native, '_roles', lambda rung: {'main': {'context': 4096}})
        monkeypatch.setattr(native, '_component', lambda *a: child)
        return native
    service_runtime.backend_factory = factory
    service_runtime.synchronize(selection, hw)
    now = [0.0]
    lc = IdleLifecycle(state_dir=tmp_path/'lc', clock=lambda: now[0], idle_timeout_s=1)
    service_runtime.observe_residency(lc)
    now[0] = 2
    service_runtime.release_idle_residency(lc)
    assert lc.status()['roles']['main']['residency'] == 'unknown'  # No-op is not unload.
    lc.wake('main', lambda: service_runtime.ensure_residency('main'))
    service_runtime.observe_residency(lc)
    old = service_runtime.controller.backend
    process = old._owned['main'].process
    try:
        process.terminate()
        process.wait(timeout=3)
        service_runtime.synchronize(selection, hw)  # Real constructor recovery loses signalling authority.
        assert not service_runtime.controller.backend._owned
        service_runtime.observe_residency(lc)
        assert lc.status()['roles']['main']['residency'] == 'error'
        now[0] += 2
        with pytest.raises(LifecycleError):
            service_runtime.release_idle_residency(lc)
        service_runtime.synchronize(selection, hw)  # Dead record was cleaned; error still survives.
        service_runtime.observe_residency(lc)
        service_runtime.release_idle_residency(lc)
        assert lc.status()['roles']['main']['residency'] == 'error'
    finally:
        old.reset_managed()


@pytest.mark.parametrize('mismatch', ['command', 'port'])
def test_exact_recipe_and_endpoint_required_by_snapshot_and_ensure(tmp_path, monkeypatch, mismatch):
    from turbofit_runtime.native_backend import OwnedRuntime
    from turbofit_runtime.reconciler import ReconcileError
    runtime = backend(tmp_path)
    desired = component('main')
    monkeypatch.setattr(runtime, '_roles', lambda rung: {'main': {'context': 4096}})
    monkeypatch.setattr(runtime, '_component', lambda *a: desired)
    runtime._owned['main'] = OwnedRuntime(role='main', pid=123, alias=desired.alias,
        port=desired.port + (1 if mismatch == 'port' else 0),
        command=desired.command + (('--wrong',) if mismatch == 'command' else ()))
    monkeypatch.setattr(runtime, '_is_owned', lambda resident: True)
    probes = []
    monkeypatch.setattr(runtime, '_healthy', lambda resident: probes.append(resident.port) or True)
    monkeypatch.setattr(runtime, '_stop', lambda *a, **k: False)
    assert runtime.residency_snapshot()['main']['residency'] == 'error'
    with pytest.raises(ReconcileError, match='replace'):
        runtime.ensure_role('main')
    assert probes == []


def test_idle_identity_loss_between_precheck_and_stop_is_error(tmp_path, monkeypatch):
    from test_native_identity import owned
    runtime = backend(tmp_path)
    runtime._owned['aux'] = owned()
    ownership = iter([True, False])
    monkeypatch.setattr(runtime, '_is_owned', lambda resident: next(ownership))
    assert runtime.release_idle_role('aux') == 'error'
    assert not runtime._owned
