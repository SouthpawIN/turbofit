"""TF6: process identity hardening for owned native runtimes."""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import pytest

from turbofit_runtime import native_backend as nb
from turbofit_runtime.native_backend import NativeRuntimeBackend, OwnedRuntime
from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.reconciler import ReconcilerState
from turbofit_runtime.routes import load_runtime_resolutions

ROOT = Path(__file__).parents[1]


def backend(tmp_path: Path, identity_reader=None) -> NativeRuntimeBackend:
    profile = load_yaml_profile(ROOT / "runtime-profiles/24gb.yaml")
    kwargs = {}
    if identity_reader is not None:
        kwargs["identity_reader"] = identity_reader
    return NativeRuntimeBackend(
        profile=profile,
        resolutions=load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json"),
        recipe_book=RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cpu"),
        route_state_path=tmp_path / "routes.json",
        state_dir=tmp_path / "native",
        manager_port=8092,
        current_state=ReconcilerState(profile.id, 0, "local:main", "local:main"),
        verification_timeout_s=0,
        **kwargs,
    )


def owned(identity: str | None = "linux:12345") -> OwnedRuntime:
    return OwnedRuntime(
        role="aux",
        pid=4242,
        alias="owned",
        port=8093,
        command=("llama-server", "--alias", "owned", "--port", "8093"),
        start_identity=identity,
    )


def test_reused_pid_with_matching_cmdline_and_different_identity_is_not_signalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same binary/alias/port on a recycled PID must never be signalled."""
    runtime = backend(tmp_path, identity_reader=lambda pid: "linux:99999")
    runtime._owned["aux"] = owned("linux:12345")
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(nb.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(nb.NativeRuntimeBackend, "_command_line", staticmethod(lambda pid: "llama-server --alias owned --port 8093"))

    assert runtime._stop("aux") is True
    assert signalled == []


def test_genuine_owned_process_with_matching_identity_is_reclaimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = backend(tmp_path, identity_reader=lambda pid: "linux:12345")
    runtime._owned["aux"] = owned("linux:12345")
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(nb.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    cmdline = {"alive": True}

    def fake_cmdline(pid: int) -> str:
        return "llama-server --alias owned --port 8093" if cmdline["alive"] else ""

    monkeypatch.setattr(nb.NativeRuntimeBackend, "_command_line", staticmethod(fake_cmdline))
    original_kill = nb.os.kill

    def kill_clears(pid: int, sig: int) -> None:
        original_kill(pid, sig)
        cmdline["alive"] = False

    monkeypatch.setattr(nb.os, "kill", kill_clears)

    assert runtime._stop("aux") is True
    assert signalled == [(4242, signal.SIGTERM)]


def test_unreadable_identity_fails_safe_and_never_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows fail-safe: when identity cannot be proven, never signal."""
    runtime = backend(tmp_path, identity_reader=lambda pid: None)
    runtime._owned["aux"] = owned("linux:12345")
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(nb.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(nb.NativeRuntimeBackend, "_command_line", staticmethod(lambda pid: "llama-server --alias owned --port 8093"))

    assert runtime._stop("aux") is True
    assert signalled == []
    assert not (tmp_path / "native" / "aux.json").exists()


def test_legacy_record_without_identity_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = backend(tmp_path, identity_reader=lambda pid: "linux:55555")
    runtime._owned["aux"] = owned(None)
    monkeypatch.setattr(nb.NativeRuntimeBackend, "_command_line", staticmethod(lambda pid: "llama-server --alias owned --port 8093"))

    assert runtime._is_owned(runtime._owned["aux"]) is False
    assert not (tmp_path / "native" / "aux.json").exists()


def test_recovered_record_rejects_reused_pid_even_with_same_cmdline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "native"
    state_dir.mkdir(parents=True)
    (state_dir / "aux.json").write_text(json.dumps({
        "pid": 4242,
        "alias": "owned",
        "port": 8093,
        "command": ["llama-server", "--alias", "owned", "--port", "8093"],
        "start_identity": "linux:12345",
    }))
    monkeypatch.setattr(nb.NativeRuntimeBackend, "_command_line", staticmethod(lambda pid: "llama-server --alias owned --port 8093"))
    runtime = backend(tmp_path, identity_reader=lambda pid: "linux:99999")

    assert "aux" not in runtime._owned


def test_invalid_stale_pid_file_fails_safely(tmp_path: Path) -> None:
    state_dir = tmp_path / "native"
    state_dir.mkdir(parents=True)
    (state_dir / "aux.json").write_text("{not json")

    runtime = backend(tmp_path)  # must not raise

    assert "aux" not in runtime._owned


def test_start_persists_identity_of_spawned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 4242

        def poll(self):
            return None

    component = type("C", (), {})()
    component.role = "aux"
    component.command = ("/bin/true",)
    component.model_path = None
    component.projector_path = None
    component.alias = "owned"
    component.port = 8093
    component.gpu = "0"
    monkeypatch.setattr(nb.Path, "is_file", lambda self: True)
    monkeypatch.setattr(nb.subprocess, "Popen", lambda *a, **k: FakeProcess())

    runtime = backend(tmp_path, identity_reader=lambda pid: "linux:777")
    started = runtime._start(component)

    assert started.start_identity == "linux:777"
    persisted = json.loads((tmp_path / "native" / "aux.json").read_text())
    assert persisted["start_identity"] == "linux:777"


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc; Linux only")
def test_process_start_identity_reads_live_linux_process() -> None:
    identity = nb.process_start_identity(os.getpid())
    assert identity is not None and identity.startswith("linux:")


def test_process_start_identity_missing_process_is_none() -> None:
    assert nb.process_start_identity(2**22) is None