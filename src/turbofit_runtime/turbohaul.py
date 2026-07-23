"""Compile successful Turbofit components into Turbohaul v0.7 manifests."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class UnsupportedTurbohaulMethod(ValueError):
    pass


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    model_tag: str
    model_path: Path
    projector_path: Path | None
    context: int
    expected_vram_mb: int
    gpu: int
    method: str
    cache_type_k: str
    cache_type_v: str
    auto_place_eligible: bool
    vision: bool
    arch: str = ""
    hybrid_kv_ratio: float = 1.0
    kv_bytes_per_token: float | None = None


@dataclass(frozen=True)
class CompiledComponent:
    manifest: dict
    sources: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TurbohaulCompiler:
    SUPPORTED_METHODS = frozenset({"baseline", "mtp"})

    def compile_component(self, component: ComponentSpec) -> CompiledComponent:
        if component.method not in self.SUPPORTED_METHODS:
            raise UnsupportedTurbohaulMethod(
                f"{component.method} is not in Turbohaul v0.7's closed speculative allowlist"
            )
        if not component.model_path.is_file():
            raise FileNotFoundError(component.model_path)
        if component.vision and not component.projector_path:
            raise ValueError("vision component requires projector_path")
        if component.projector_path and not component.projector_path.is_file():
            raise FileNotFoundError(component.projector_path)
        if component.expected_vram_mb <= 0:
            raise ValueError("expected_vram_mb must be measured and positive")

        flags = {
            "ctx_size": component.context,
            "n_gpu_layers": 999,
            "split_mode": "none",
            "main_gpu": component.gpu,
            "parallel": 1,
            "cache_type_k": component.cache_type_k,
            "cache_type_v": component.cache_type_v,
            "flash_attn": True,
            "no_context_shift": True,
            "cache_reuse": 256,
            "slot_prompt_similarity": 0.5,
            "no_perf": True,
            "jinja": True,
        }
        if component.method == "mtp":
            flags["spec_type"] = "draft-mtp"

        manifest = {
            "model_tag": component.model_tag,
            "display_name": component.model_tag,
            "description": (
                f"Turbofit measured {component.role} runtime; "
                f"context={component.context}; method={component.method}"
            ),
            "gguf_blob_sha256": sha256_file(component.model_path),
            "mmproj_blob_sha256": sha256_file(component.projector_path) if component.projector_path else "",
            "gguf_size_bytes": component.model_path.stat().st_size,
            "context_size": component.context,
            "expected_vram_bytes": component.expected_vram_mb * 1024 * 1024,
            "auto_place": component.auto_place_eligible,
            "arch": component.arch,
            "hybrid_kv_ratio": component.hybrid_kv_ratio,
            "revision": 1,
            "llama_server_flags": flags,
            "prompt_template": {"system_default": "", "stop_tokens": []},
        }
        if component.kv_bytes_per_token is not None:
            if component.kv_bytes_per_token < 1024:
                raise ValueError("kv_bytes_per_token is bytes/token and must be >= 1024")
            manifest["kv_bytes_per_token"] = component.kv_bytes_per_token
        return CompiledComponent(
            manifest=manifest,
            sources={
                "gguf": str(component.model_path),
                "mmproj": str(component.projector_path) if component.projector_path else "",
            },
        )

    @staticmethod
    def runtime_string(profile: str, model_tags: tuple[str, ...]) -> str:
        if not model_tags:
            raise ValueError("at least one model tag is required")
        return (
            f"turbofit-runtime use {profile} --backend turbohaul "
            f"--models {','.join(model_tags)}"
        )
