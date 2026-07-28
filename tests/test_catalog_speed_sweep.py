from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "tools/benchmarks/catalog-speed-sweep.py"


def load_sweep():
    spec = spec_from_file_location("catalog_speed_sweep", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_port_does_not_abort_when_docker_cleanup_times_out(monkeypatch) -> None:
    sweep = load_sweep()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("docker", "rm", "-f"):
            raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sweep, "run", fake_run)
    monkeypatch.setattr(sweep.time, "sleep", lambda _: None)

    sweep.cleanup_port()

    assert any(args[0] == "fuser" for args in calls)


def test_restore_after_sweep_runs_even_if_cleanup_raises(monkeypatch) -> None:
    sweep = load_sweep()
    restored = []
    monkeypatch.setattr(sweep, "cleanup_port", lambda: (_ for _ in ()).throw(RuntimeError("cleanup")))
    monkeypatch.setattr(sweep, "restore_services", lambda names: restored.extend(names))

    try:
        sweep.restore_after_sweep(["gateway.service"])
    except RuntimeError:
        pass

    assert restored == ["gateway.service"]
