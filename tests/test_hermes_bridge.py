"""TurboFit adopts Hermes' local runtime where Hermes has the primitive."""

from __future__ import annotations

import types

import pytest

from turbofit_runtime import hermes_bridge


def _stub_modules(*, ladder=None, budget=None, supervisor_cls=None):
    """A minimal stand-in for hermes_cli.local_runtime."""
    modules: dict[str, types.ModuleType] = {}
    if ladder is not None:
        context_policy = types.ModuleType("hermes_cli.local_runtime.context_policy")
        context_policy.ladder = ladder
        modules["hermes_cli.local_runtime.context_policy"] = context_policy
    if budget is not None:
        hardware = types.ModuleType("hermes_cli.local_runtime.hardware")
        hardware.probe_budget = budget
        modules["hermes_cli.local_runtime.hardware"] = hardware
    if supervisor_cls is not None:
        supervisor = types.ModuleType("hermes_cli.local_runtime.supervisor")
        supervisor.LlamaServerSupervisor = supervisor_cls
        modules["hermes_cli.local_runtime.supervisor"] = supervisor
    return modules


@pytest.fixture()
def without_hermes(monkeypatch):
    def absent(name):
        raise ImportError(name)

    monkeypatch.setattr(hermes_bridge, "import_module", absent)
    return hermes_bridge


@pytest.fixture()
def with_hermes(monkeypatch):
    class FakeSupervisor:
        def __init__(self, binary, models_dir, **kwargs):
            self.binary = binary
            self.models_dir = models_dir
            self.kwargs = kwargs
            self.unloaded: list[str] = []
            self.swept = False

        def unload_model(self, model_id):
            self.unloaded.append(model_id)

        def sweep_idle(self):
            self.swept = True
            return ["qwen3-8-27b-gsq-rco-iq3s"]

    modules = _stub_modules(
        ladder=lambda native: [65536, 131072, native],
        budget=lambda: {"vram_free_bytes": 24 * 1024**3},
        supervisor_cls=FakeSupervisor,
    )

    def resolve(name):
        if name in modules:
            return modules[name]
        raise ImportError(name)

    monkeypatch.setattr(hermes_bridge, "import_module", resolve)
    return hermes_bridge, FakeSupervisor


def test_without_hermes_every_primitive_is_reported_missing(without_hermes):
    report = without_hermes.report()

    assert report["hermes_local_runtime"] is False
    assert report["adopted"] == []
    assert set(report["missing"]) == set(hermes_bridge.PRIMITIVES)
    assert "0 of" in report["summary"]


def test_the_fallback_ladder_is_used_when_hermes_has_none(without_hermes):
    assert without_hermes.context_ladder(262_144) == [262_144, 131_072, 65_536, 32_768]
    assert without_hermes.context_ladder(32_768) == [32_768]


def test_hermes_ladder_wins_and_is_ordered_largest_first(with_hermes):
    bridge, _ = with_hermes

    assert bridge.context_ladder(262_144) == [262_144, 131_072, 65_536]


def test_report_names_what_each_adopted_primitive_replaces(with_hermes):
    bridge, _ = with_hermes
    report = bridge.report()

    assert report["adopted"] == ["context_ladder", "hardware_budget", "supervisor"]
    assert "native_backend" in report["replaces"]["supervisor"]
    assert "pressure sensing" in " ".join(report["kept_by_turbofit"])
    assert "3 of" in report["summary"]


def test_budget_comes_from_hermes(with_hermes):
    bridge, _ = with_hermes

    assert bridge.probe_budget() == {"vram_free_bytes": 24 * 1024**3}


def test_a_broken_budget_probe_is_survivable(monkeypatch):
    def explode():
        raise RuntimeError("driver fell over")

    modules = _stub_modules(budget=explode)
    monkeypatch.setattr(
        hermes_bridge, "import_module",
        lambda name: modules[name] if name in modules else (_ for _ in ()).throw(ImportError(name)),
    )

    assert hermes_bridge.probe_budget() is None


def test_the_supervisor_owns_the_process_and_unload(with_hermes):
    bridge, fake = with_hermes
    supervisor = bridge.supervisor("/opt/bin/llama-server", "/models")

    assert isinstance(supervisor, fake)
    assert supervisor.binary == "/opt/bin/llama-server"
    assert bridge.unload(supervisor, "qwen3-8-27b-gsq-rco-iq3s") is True
    assert supervisor.unloaded == ["qwen3-8-27b-gsq-rco-iq3s"]
    assert bridge.idle_release(supervisor) == ["qwen3-8-27b-gsq-rco-iq3s"]
    assert supervisor.swept is True


def test_unload_reports_false_when_the_supervisor_cannot(without_hermes):
    class Bare:
        pass

    assert without_hermes.unload(Bare(), "anything") is False
    assert without_hermes.idle_release(Bare()) == []


def test_a_hermes_install_beside_the_service_is_put_on_the_path(tmp_path, monkeypatch):
    root = tmp_path / "hermes-agent"
    (root / "hermes_cli").mkdir(parents=True)
    monkeypatch.setattr(hermes_bridge, "HERMES_ROOTS", ("{HERMES_SOURCE}", "{HERMES_HOME}/nope"))
    monkeypatch.setenv("HERMES_SOURCE", str(root))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setattr(hermes_bridge, "_path_checked", False)

    def absent(name):
        raise ImportError(name)

    monkeypatch.setattr(hermes_bridge, "import_module", absent)

    found = hermes_bridge.ensure_hermes_on_path()

    assert found == root
    assert str(root) in __import__("sys").path


def test_a_missing_hermes_install_is_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(hermes_bridge, "HERMES_ROOTS", ("{HERMES_SOURCE}",))
    monkeypatch.setenv("HERMES_SOURCE", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(hermes_bridge, "_path_checked", False)
    monkeypatch.setattr(
        hermes_bridge, "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    assert hermes_bridge.ensure_hermes_on_path() is None
