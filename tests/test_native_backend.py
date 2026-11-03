from __future__ import annotations

from pathlib import Path

import pytest

from turbofit_runtime.native_backend import NativeRuntimeBackend, OwnedRuntime
from turbofit_runtime.profile_io import load_yaml_profile
from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.reconciler import ReconcileError, ReconcilerState
from turbofit_runtime.routes import load_runtime_resolutions

ROOT = Path(__file__).parents[1]


def backend(tmp_path: Path) -> NativeRuntimeBackend:
    profile = load_yaml_profile(ROOT / "runtime-profiles/24gb.yaml")
    return NativeRuntimeBackend(
        profile=profile,
        resolutions=load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json"),
        recipe_book=RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cpu"),
        route_state_path=tmp_path / "routes.json",
        state_dir=tmp_path / "native",
        manager_port=8092,
        current_state=ReconcilerState(profile.id, 0, "local:main", "local:main"),
        verification_timeout_s=0,
    )


def test_native_resolution_compiles_loopback_process_with_alias(tmp_path: Path) -> None:
    runtime = backend(tmp_path)
    item = runtime._roles("local-gsq-rco-iq3s-262144")["main"]

    component = runtime._component("main", item, 262144)

    command = list(component.command)
    assert component.kind == "process"
    assert component.port == 8092
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--alias") + 1] == "gsq-rco-iq3s-262144-main"
    assert "--jinja" in command
    assert component.gpu == "0"


def test_native_backend_refuses_unowned_process_escalation(tmp_path: Path) -> None:
    runtime = backend(tmp_path)
    runtime._retiring_aux = OwnedRuntime(
        role="aux",
        pid=123,
        alias="owned",
        port=8093,
        command=("llama-server", "--alias", "owned", "--port", "8093"),
    )

    with pytest.raises(ReconcileError, match="not owned"):
        runtime.escalate_owned((999,))


def test_runtime_resolution_pins_native_family_gpu_and_port(tmp_path: Path) -> None:
    runtime = backend(tmp_path)
    item = runtime._roles("local-gsq-rco-iq3s-262144")["main"]

    assert item["family"] == "Qwen 3.8 27B GSQ-RCO IQ3_S"
    assert item["gpu"] == "0"
    assert item["port"] == 8092


def shared_main_backend(tmp_path: Path) -> NativeRuntimeBackend:
    """A shared-main profile whose resolutions still carry an aux lane."""
    profile = load_yaml_profile(ROOT / "runtime-profiles/48gb.yaml")
    return NativeRuntimeBackend(
        profile=profile,
        resolutions=load_runtime_resolutions(ROOT / "runtime-profiles/runtime-resolutions.json"),
        recipe_book=RecipeBook.load(ROOT / "references/model-recipes.json", backend_name="cpu"),
        route_state_path=tmp_path / "routes.json",
        state_dir=tmp_path / "native",
        manager_port=8092,
        current_state=ReconcilerState(profile.id, 0, "local:main", "local:main"),
        verification_timeout_s=0,
    )


def test_shared_main_rung_launches_one_process_not_two(tmp_path: Path) -> None:
    """A shared-main rung serves aux from main; a second server doubles VRAM."""
    runtime = shared_main_backend(tmp_path)
    rung = runtime.profile.rungs[0]
    assert rung.aux_mode.value == "shared-main"
    assert set(runtime._roles(rung.id)) == {"main", "aux"}, "fixture must carry an aux lane"

    started: list[str] = []

    def fake_start(component):
        started.append(component.role)
        owned = OwnedRuntime(
            role=component.role, pid=4242, alias=component.alias,
            port=component.port, command=tuple(component.command),
        )
        runtime._owned[component.role] = owned
        return owned

    runtime._start = fake_start
    runtime.activate_local(rung.id)

    assert started == ["main"]


def test_shared_main_rung_verifies_without_an_aux_process(tmp_path: Path) -> None:
    runtime = shared_main_backend(tmp_path)
    rung = runtime.profile.rungs[0]
    item = runtime._roles(rung.id)["main"]
    runtime._owned["main"] = OwnedRuntime(
        role="main", pid=4242, alias=str(item["model_tag"]), port=int(item["port"]),
        command=tuple(runtime._component("main", item, rung.context).command),
    )
    runtime._wait_healthy = lambda owned: True

    assert runtime.verify_rung(rung.id) is True


def test_shared_main_aux_wake_targets_main_instead_of_starting_aux(tmp_path: Path) -> None:
    runtime = shared_main_backend(tmp_path)
    rung = runtime.profile.rungs[0]
    item = runtime._roles(rung.id)["main"]
    component = runtime._component("main", item, rung.context)
    runtime._owned["main"] = OwnedRuntime(
        role="main", pid=4242, alias=component.alias, port=component.port,
        command=tuple(component.command),
    )
    runtime._healthy = lambda owned: True
    runtime._is_owned = lambda owned: True
    runtime._stop = lambda role, **kwargs: True

    def must_not_start(component):
        raise AssertionError("aux wake must not start a second server on a shared-main rung")

    runtime._start = must_not_start
    runtime.ensure_role("aux")
