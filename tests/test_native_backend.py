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
    item = runtime._roles("local-bonsai-262144")["main"]

    component = runtime._component("main", item, 262144)

    command = list(component.command)
    assert component.kind == "process"
    assert component.port == 8092
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--alias") + 1] == "bonsai-27b-1bit-262k-main"
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
    item = runtime._roles("local-bonsai-262144")["main"]

    assert item["family"] == "bonsai-27b"
    assert item["gpu"] == "0"
    assert item["port"] == 8092
