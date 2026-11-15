"""Exercise the actual scheduler entry: its repair path must not undo idle release."""
from dataclasses import replace
from types import SimpleNamespace
from test_controller import controller_for, pressure
from test_selection import catalog
from test_native_lifecycle import lifecycle, controller_module
from turbofit_runtime.reconciler import ReconcilerState


def test_scheduler_does_not_reheat_idle_roles(tmp_path, monkeypatch):
    now = [1000.0]
    lc = lifecycle(tmp_path, now)
    controller, backend = controller_for(catalog(), 'hardware-48gb', (24576, 24576))
    controller.state = replace(controller.state,
        adaptive=replace(controller.state.adaptive, current_index=0, last_stable_index=0,
                         target_ceiling_index=0, pending_index=None),
        reconciler=ReconcilerState(controller.profile.id, 0, 'local:main', 'local:aux'))
    monkeypatch.setattr(backend, 'verify_rung', lambda rung: False)
    stopped = []
    runtime = SimpleNamespace(observe_residency=lambda owner: None, synchronize=lambda *a: controller, tick=controller.tick,
                              release_idle_residency=lambda owner: owner.release_idle(lambda role: stopped.append(role) or True))
    module = controller_module()
    monkeypatch.setattr(module, 'probe_hardware', lambda: None)
    monkeypatch.setattr(module, 'probe_accelerator_pressure', lambda *a, **kw: pressure(24576, 24576))
    args = SimpleNamespace(runtime_state_dir=tmp_path/'native', selection=tmp_path/'selection')
    now[0] += 200
    assert module.run_tick(runtime, args, lc) is None
    assert set(stopped) == {'main', 'aux'}
    assert not backend.events
    # Main becomes active/recent, aux stays intentionally idle: pressure policy
    # may run, but its generic missing-runtime repair must not resurrect aux.
    lease = lc.acquire('main'); lc.release(lease)
    module.run_tick(runtime, args, lc)
    assert not any(isinstance(e, tuple) and e[0] == 'local' for e in backend.events)
