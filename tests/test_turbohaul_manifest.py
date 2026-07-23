from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from turbofit_runtime.turbohaul import (
    ChecksumCache,
    ComponentSpec,
    TurbohaulCompiler,
    UnsupportedTurbohaulMethod,
)


def test_checksum_cache_reuses_unchanged_file_and_invalidates_on_change(tmp_path: Path) -> None:
    target = tmp_path / "model.gguf"
    target.write_bytes(b"first")
    calls = []

    def hash_fn(path: Path) -> str:
        calls.append(path.read_bytes())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    cache = ChecksumCache(tmp_path / "checksums.json", hash_fn=hash_fn)

    first = cache(target)
    second = cache(target)
    target.write_bytes(b"second")
    third = cache(target)

    assert first == second
    assert third != first
    assert calls == [b"first", b"second"]


def test_compile_mtp_component_to_content_addressed_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model-bytes")
    projector = tmp_path / "mmproj.gguf"
    projector.write_bytes(b"projector-bytes")
    component = ComponentSpec(
        role="main",
        model_tag="grm-main-128k",
        model_path=model,
        projector_path=projector,
        context=131_072,
        expected_vram_mb=21_153,
        gpu=1,
        method="mtp",
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        auto_place_eligible=False,
        vision=True,
    )

    compiled = TurbohaulCompiler().compile_component(component)

    assert compiled.manifest["model_tag"] == "grm-main-128k"
    assert compiled.manifest["gguf_blob_sha256"] == hashlib.sha256(b"model-bytes").hexdigest()
    assert compiled.manifest["mmproj_blob_sha256"] == hashlib.sha256(b"projector-bytes").hexdigest()
    assert compiled.manifest["gguf_size_bytes"] == len(b"model-bytes")
    assert compiled.manifest["context_size"] == 131_072
    assert compiled.manifest["expected_vram_bytes"] == 21_153 * 1024 * 1024
    assert compiled.manifest["auto_place"] is False
    flags = compiled.manifest["llama_server_flags"]
    assert flags["ctx_size"] == 131_072
    assert flags["main_gpu"] == 1
    assert flags["split_mode"] == "none"
    assert flags["spec_type"] == "draft-mtp"
    assert flags["flash_attn"] is True
    assert flags["jinja"] is True


def test_compile_measured_kv_override_in_bytes_per_token(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 1024)
    component = ComponentSpec(
        role="main",
        model_tag="hybrid-262k",
        model_path=model,
        projector_path=None,
        context=262_144,
        expected_vram_mb=16_000,
        gpu=0,
        method="baseline",
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        auto_place_eligible=True,
        vision=False,
        arch="qwen35",
        hybrid_kv_ratio=0.25,
        kv_bytes_per_token=13_824.0,
    )

    manifest = TurbohaulCompiler().compile_component(component).manifest

    assert manifest["arch"] == "qwen35"
    assert manifest["hybrid_kv_ratio"] == 0.25
    assert manifest["kv_bytes_per_token"] == 13_824.0
    assert manifest["auto_place"] is True


def test_dspark_is_rejected_until_turbohaul_allowlist_supports_it(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    component = ComponentSpec(
        role="main",
        model_tag="ternary-dspark",
        model_path=model,
        projector_path=None,
        context=65_536,
        expected_vram_mb=14_000,
        gpu=0,
        method="dspark",
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        auto_place_eligible=False,
        vision=False,
    )

    with pytest.raises(UnsupportedTurbohaulMethod, match="dspark"):
        TurbohaulCompiler().compile_component(component)


def test_pair_runtime_string_is_deterministic() -> None:
    compiler = TurbohaulCompiler()

    value = compiler.runtime_string("grm-carwin-128k", ("grm-main-128k", "carwin-aux-128k"))

    assert value == "turbofit-runtime use grm-carwin-128k --backend turbohaul --models grm-main-128k,carwin-aux-128k"
