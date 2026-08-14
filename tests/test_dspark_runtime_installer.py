from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "install-dspark-runtime"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "install_dspark_runtime",
        SCRIPT,
        loader=SourceFileLoader("install_dspark_runtime", str(SCRIPT)),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_upstream_revision_and_runtime_location() -> None:
    module = _load_module()

    assert len(module.LLAMA_CPP_REVISION) == 40
    assert module.LLAMA_CPP_REVISION == "1c3c9674de4d455f1e571bed808252af54932767"
    assert module.LLAMA_CPP_MINIMUM_BUILD == 10269
    assert module.LLAMA_CPP_REPOSITORY == "https://github.com/ggml-org/llama.cpp.git"
    assert module.runtime_root(Path("/runtime")) == Path(
        "/runtime/llama.cpp-1c3c9674de4d455f1e571bed808252af54932767"
    )
    assert module.binary_path(Path("C:/build"), platform_name="Windows") == Path("C:/build/bin/llama-server.exe")
    assert module.binary_path(Path("/build"), platform_name="Linux") == Path("/build/bin/llama-server")
    assert all(path.is_file() for path in module.PATCHES)


@pytest.mark.parametrize(
    ("backend", "required_flag"),
    [
        ("cuda", "-DGGML_CUDA=ON"),
        ("rocm", "-DGGML_HIP=ON"),
        ("metal", "-DGGML_METAL=ON"),
        ("cpu", "-DGGML_CPU=ON"),
    ],
)
def test_build_configuration_is_backend_specific(backend: str, required_flag: str) -> None:
    module = _load_module()

    arguments = module.cmake_arguments(Path("/src"), Path("/build"), backend)

    assert arguments[:4] == ["cmake", "-S", "/src", "-B"]
    assert arguments[4] == "/build"
    assert required_flag in arguments
    assert "-DBUILD_SHARED_LIBS=OFF" in arguments
    assert "-DLLAMA_BUILD_SERVER=ON" in arguments
    assert "-DLLAMA_BUILD_TESTS=OFF" in arguments


def test_runtime_help_must_expose_dspark_and_jinja() -> None:
    module = _load_module()

    module.verify_help("--spec-type draft-dspark\n--jinja, --no-jinja\n")
    with pytest.raises(RuntimeError, match="draft-dspark"):
        module.verify_help("--jinja\n")
    with pytest.raises(RuntimeError, match="jinja"):
        module.verify_help("--spec-type draft-dspark\n")


def test_auto_backend_prefers_available_accelerator(monkeypatch) -> None:
    module = _load_module()

    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/tool" if name == "nvidia-smi" else None)
    assert module.resolve_backend("auto", platform_name="linux") == "cuda"

    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/tool" if name in {"rocminfo", "amd-smi"} else None)
    assert module.resolve_backend("auto", platform_name="linux") == "rocm"
    assert module.resolve_backend("auto", platform_name="darwin") == "metal"
