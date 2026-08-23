from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.freetoken import (
    FREETOKEN_REVISION,
    FREETOKEN_VERSION,
    FreeTokenClient,
    build_freetoken_command,
    evaluate_freetoken_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]


def test_freetoken_accepts_current_nvidia_linux_contract() -> None:
    result = evaluate_freetoken_compatibility(
        os_name="linux",
        architecture="x86_64",
        accelerator_backends=("cuda",),
        driver_version="580.159.03",
        nvcc_version="13.0",
    )

    assert result.compatible is True
    assert result.status == "candidate"
    assert result.blockers == ()


def test_freetoken_rejects_cuda_12_and_non_nvidia_hosts() -> None:
    cuda12 = evaluate_freetoken_compatibility(
        os_name="linux",
        architecture="x86_64",
        accelerator_backends=("cuda",),
        driver_version="580.159.03",
        nvcc_version="12.0",
    )
    rocm = evaluate_freetoken_compatibility(
        os_name="linux",
        architecture="x86_64",
        accelerator_backends=("rocm",),
        driver_version="",
        nvcc_version="",
    )

    assert cuda12.compatible is False
    assert "CUDA toolkit 13+ required" in cuda12.blockers
    assert rocm.compatible is False
    assert "NVIDIA CUDA backend required" in rocm.blockers


def test_freetoken_command_is_loopback_identity_bound_and_text_only() -> None:
    command = build_freetoken_command(
        binary="/runtime/bin/ft",
        model="/models/qwen",
        alias="qwen3-6-35b-a3b",
        port=1919,
        context=131_072,
        memory_ratio=0.85,
        moe=True,
        vision=False,
    )

    assert command == (
        "/runtime/bin/ft", "serve", "--model", "/models/qwen",
        "--host", "127.0.0.1", "--port", "1919",
        "--served-model-name", "qwen3-6-35b-a3b",
        "--max-seq-len-override", "131072", "--memory-ratio", "0.85",
        "--moe-backend", "auto",
    )
    with pytest.raises(ValueError, match="text-only"):
        build_freetoken_command(
            binary="ft", model="model", alias="alias", port=1919,
            context=65_536, memory_ratio=0.9, moe=True, vision=True,
        )
    with pytest.raises(ValueError, match="MoE"):
        build_freetoken_command(
            binary="ft", model="model", alias="alias", port=1919,
            context=65_536, memory_ratio=0.9, moe=False, vision=False,
        )


def test_freetoken_client_verifies_health_model_and_stats() -> None:
    def transport(method: str, url: str, timeout: float) -> tuple[int, dict]:
        assert method == "GET"
        if url.endswith("/health"):
            return 200, {"status": "ok", "model": "qwen"}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": "qwen", "context_length": 131072}]}
        if url.endswith("/v1/stats"):
            return 200, {"throughput": {"decode_tps": 39.3}}
        raise AssertionError(url)

    client = FreeTokenClient("http://127.0.0.1:1919", transport=transport)

    assert client.health()["status"] == "ok"
    assert client.models()[0]["id"] == "qwen"
    assert client.stats()["throughput"]["decode_tps"] == 39.3


def test_freetoken_manifest_is_pinned_and_candidate_only() -> None:
    manifest = json.loads((ROOT / "references" / "native-runtimes.json").read_text())
    runtime = next(item for item in manifest["runtimes"] if item["id"] == "freetoken")

    assert runtime["repository"] == "https://github.com/FlashML-org/FreeToken.git"
    assert runtime["revision"] == FREETOKEN_REVISION == "0ab982f10905fa775962a4eddcb44caa50065251"
    assert runtime["version"] == FREETOKEN_VERSION == "0.1.2"
    assert runtime["status"] == "candidate"
    assert runtime["auto_promote"] is False
