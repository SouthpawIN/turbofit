from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.runtime_backends import LemonadeClient, llama_environment


def test_rocm_environment_selects_hip_devices_without_cuda() -> None:
    env = llama_environment("rocm", devices="0,1", base={"KEEP": "yes", "CUDA_VISIBLE_DEVICES": "9"})

    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["HIP_VISIBLE_DEVICES"] == "0,1"
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert env["KEEP"] == "yes"


def test_cuda_environment_selects_cuda_without_hip() -> None:
    env = llama_environment("cuda", devices="2", base={"HIP_VISIBLE_DEVICES": "9"})

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "HIP_VISIBLE_DEVICES" not in env
    assert "ROCR_VISIBLE_DEVICES" not in env


def test_lemonade_client_uses_openai_compatible_api_and_explicit_rocm_load() -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def transport(method: str, url: str, payload: dict | None, _timeout: float) -> tuple[int, dict]:
        calls.append((method, url, payload))
        if url.endswith("/health"):
            return 200, {"status": "ok", "version": "9.3.3"}
        if url.endswith("/models"):
            return 200, {"data": [{"id": "user.coder"}]}
        return 200, {"status": "loaded"}

    client = LemonadeClient("http://127.0.0.1:13305/api/v1", transport=transport)

    assert client.health()["status"] == "ok"
    assert client.models()[0]["id"] == "user.coder"
    client.load(
        "user.coder",
        context=131072,
        backend="rocm",
        llama_args=("--cache-type-k", "q4_0"),
    )

    method, url, payload = calls[-1]
    assert method == "POST"
    assert url == "http://127.0.0.1:13305/api/v1/load"
    assert payload == {
        "model_name": "user.coder",
        "ctx_size": 131072,
        "llamacpp_backend": "rocm",
        "llamacpp_args": "--cache-type-k q4_0",
    }


def test_lemonade_client_rejects_server_error() -> None:
    def transport(_method: str, _url: str, _payload: dict | None, _timeout: float) -> tuple[int, dict]:
        return 503, {"error": "unavailable"}

    client = LemonadeClient("http://127.0.0.1:13305/v1", transport=transport)

    try:
        client.health()
    except RuntimeError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("expected health failure")
