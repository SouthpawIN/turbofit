"""Strict hybrid CPU RAM + GPU VRAM runtime configurations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "turbofit.hybrid-models/v1"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ALLOWED_STATUS = {"configured-unmeasured", "validated", "blocked"}
_ALLOWED_ENGINES = {"llama.cpp", "llama.cpp-minimax-m3", "ik_llama.cpp"}


@dataclass(frozen=True)
class ArtifactFile:
    filename: str
    expected_sha256: str
    size_bytes: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactFile":
        _exact(value, {"filename", "expected_sha256", "size_bytes"}, "artifact file")
        item = cls(str(value["filename"]), str(value["expected_sha256"]), value["size_bytes"])
        if not item.filename or Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
            raise ValueError("artifact filename must be relative and traversal-free")
        if not _HASH_RE.fullmatch(item.expected_sha256):
            raise ValueError("artifact expected_sha256 must be a sha256 identity")
        if not isinstance(item.size_bytes, int) or isinstance(item.size_bytes, bool) or item.size_bytes <= 0:
            raise ValueError("artifact size_bytes must be a positive integer")
        return item


@dataclass(frozen=True)
class ArtifactSource:
    repo_id: str
    revision: str
    files: tuple[ArtifactFile, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactSource":
        _exact(value, {"repo_id", "revision", "files"}, "artifact source")
        raw_files = value["files"]
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError("artifact source files must be a non-empty list")
        source = cls(
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            files=tuple(ArtifactFile.from_mapping(item) for item in raw_files),
        )
        if source.repo_id.count("/") != 1 or any(part in {"", ".", ".."} for part in source.repo_id.split("/")):
            raise ValueError("repo_id must be owner/repository")
        if not _REVISION_RE.fullmatch(source.revision):
            raise ValueError("artifact revision must be an immutable commit")
        if len({item.filename for item in source.files}) != len(source.files):
            raise ValueError("artifact filenames must be unique")
        return source


@dataclass(frozen=True)
class LaunchPolicy:
    gpu_layers: int
    cpu_moe_layers: int | str | None
    split_mode: str
    tensor_split: tuple[float, ...]
    threads: int
    threads_batch: int
    batch_size: int
    ubatch_size: int
    cache_type_k: str
    cache_type_v: str
    mmap: bool
    extra_args: tuple[str, ...]
    mlock: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LaunchPolicy":
        fields = {
            "gpu_layers", "cpu_moe_layers", "split_mode", "tensor_split",
            "threads", "threads_batch", "batch_size", "ubatch_size",
            "cache_type_k", "cache_type_v", "mmap", "extra_args", "mlock",
        }
        _exact(value, fields, "launch policy")
        raw_split = value["tensor_split"]
        if not isinstance(raw_split, list) or not raw_split:
            raise ValueError("tensor_split must be a non-empty list")
        raw_extra = value["extra_args"]
        if not isinstance(raw_extra, list):
            raise ValueError("extra_args must be a list")
        policy = cls(
            gpu_layers=value["gpu_layers"],
            cpu_moe_layers=value["cpu_moe_layers"],
            split_mode=str(value["split_mode"]),
            tensor_split=tuple(raw_split) if isinstance(raw_split, list) else (),
            threads=value["threads"],
            threads_batch=value["threads_batch"],
            batch_size=value["batch_size"],
            ubatch_size=value["ubatch_size"],
            cache_type_k=str(value["cache_type_k"]),
            cache_type_v=str(value["cache_type_v"]),
            mmap=value["mmap"],
            extra_args=tuple(raw_extra),
            mlock=value["mlock"],
        )
        for name in ("gpu_layers", "threads", "threads_batch", "batch_size", "ubatch_size"):
            _positive_int(getattr(policy, name), name)
        if policy.cpu_moe_layers is not None:
            if policy.cpu_moe_layers != "all":
                _positive_int(policy.cpu_moe_layers, "cpu_moe_layers")
        if policy.split_mode not in {"layer", "row", "graph"}:
            raise ValueError("split_mode must be layer, row, or graph")
        if not policy.tensor_split or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in policy.tensor_split):
            raise ValueError("tensor_split values must be positive")
        forbidden = ("-m", "--model", "--port", "--host", "--api-key", "--token", "--password")
        for argument in policy.extra_args:
            if not isinstance(argument, str) or not argument:
                raise ValueError("extra_args values must be non-empty strings")
            lowered = argument.lower()
            if any(lowered == item or lowered.startswith(item + "=") for item in forbidden):
                raise ValueError("extra_args cannot override identity, network, or secret arguments")
        if not isinstance(policy.mmap, bool) or not isinstance(policy.mlock, bool):
            raise ValueError("mmap and mlock must be booleans")
        if policy.mlock and policy.mmap:
            raise ValueError("hybrid configurations cannot mlock a memory-mapped large model")
        return policy


@dataclass(frozen=True)
class HybridConfiguration:
    id: str
    status: str
    evidence: str | None
    context: int
    min_system_ram_mib: int
    required_available_ram_mib: int
    min_vram_mb_per_card: tuple[int, ...]
    engine: str
    launch: LaunchPolicy

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HybridConfiguration":
        _exact(
            value,
            {"id", "status", "evidence", "context", "min_system_ram_mib", "required_available_ram_mib", "min_vram_mb_per_card", "engine", "launch"},
            "hybrid configuration",
        )
        raw_vram = value["min_vram_mb_per_card"]
        item = cls(
            id=str(value["id"]),
            status=str(value["status"]),
            evidence=value["evidence"],
            context=value["context"],
            min_system_ram_mib=value["min_system_ram_mib"],
            required_available_ram_mib=value["required_available_ram_mib"],
            min_vram_mb_per_card=tuple(raw_vram) if isinstance(raw_vram, list) else (),
            engine=str(value["engine"]),
            launch=LaunchPolicy.from_mapping(value["launch"]),
        )
        if not _SLUG_RE.fullmatch(item.id):
            raise ValueError("invalid hybrid configuration id")
        if item.status not in _ALLOWED_STATUS:
            raise ValueError("invalid hybrid configuration status")
        if item.status == "validated" and not isinstance(item.evidence, str):
            raise ValueError("validated configuration requires evidence")
        if item.evidence is not None and not _HASH_RE.fullmatch(str(item.evidence)):
            raise ValueError("configuration evidence must be a sha256 identity")
        _positive_int(item.context, "context")
        _positive_int(item.min_system_ram_mib, "min_system_ram_mib")
        _positive_int(item.required_available_ram_mib, "required_available_ram_mib")
        if item.required_available_ram_mib > item.min_system_ram_mib:
            raise ValueError("required available RAM cannot exceed minimum system RAM")
        if not item.min_vram_mb_per_card:
            raise ValueError("min_vram_mb_per_card must not be empty")
        for amount in item.min_vram_mb_per_card:
            _positive_int(amount, "min_vram_mb_per_card")
        if item.engine not in _ALLOWED_ENGINES:
            raise ValueError("unsupported hybrid runtime engine")
        return item

    def fits(
        self,
        *,
        total_system_ram_mib: int,
        vram_mb_per_card: tuple[int, ...],
        available_system_ram_mib: int | None = None,
    ) -> bool:
        available_ram = (
            total_system_ram_mib
            if available_system_ram_mib is None
            else available_system_ram_mib
        )
        return (
            total_system_ram_mib >= self.min_system_ram_mib
            and available_ram >= self.required_available_ram_mib
            and len(vram_mb_per_card) == len(self.min_vram_mb_per_card)
            and all(actual >= required for actual, required in zip(vram_mb_per_card, self.min_vram_mb_per_card))
        )

    def command(self, *, binary: str, model_path: str, port: int) -> tuple[str, ...]:
        command = [
            binary, "-m", model_path, "--port", str(port), "--host", "127.0.0.1",
            "-c", str(self.context), "-ngl", str(self.launch.gpu_layers),
            "--split-mode", self.launch.split_mode,
            "--tensor-split", ",".join(_number(value) for value in self.launch.tensor_split),
            "--threads", str(self.launch.threads),
            "--threads-batch", str(self.launch.threads_batch),
            "-b", str(self.launch.batch_size), "-ub", str(self.launch.ubatch_size),
            "--flash-attn", "on",
            "--cache-type-k", self.launch.cache_type_k,
            "--cache-type-v", self.launch.cache_type_v,
            "--parallel", "1",
        ]
        if self.engine != "ik_llama.cpp":
            command.extend(["--fit", "off"])
        if self.launch.cpu_moe_layers == "all":
            command.append("--cpu-moe")
        elif self.launch.cpu_moe_layers is not None:
            command.extend(["--n-cpu-moe", str(self.launch.cpu_moe_layers)])
        if not self.launch.mmap:
            command.append("--no-mmap")
        if self.launch.mlock:
            command.append("--mlock")
        command.extend(self.launch.extra_args)
        return tuple(command)


@dataclass(frozen=True)
class HybridModel:
    id: str
    name: str
    architecture: str
    runtime_requirement: str
    parameter_count_b: float
    native_context: int
    source: ArtifactSource
    configurations: tuple[HybridConfiguration, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HybridModel":
        _exact(
            value,
            {"id", "name", "architecture", "runtime_requirement", "parameter_count_b", "native_context", "source", "configurations"},
            "hybrid model",
        )
        raw_configs = value["configurations"]
        if not isinstance(raw_configs, list) or not raw_configs:
            raise ValueError("hybrid model configurations must be a non-empty list")
        item = cls(
            id=str(value["id"]), name=str(value["name"]), architecture=str(value["architecture"]),
            runtime_requirement=str(value["runtime_requirement"]),
            parameter_count_b=float(value["parameter_count_b"]), native_context=value["native_context"],
            source=ArtifactSource.from_mapping(value["source"]),
            configurations=tuple(HybridConfiguration.from_mapping(config) for config in raw_configs),
        )
        if not _SLUG_RE.fullmatch(item.id) or not item.name.strip() or not item.architecture.strip() or not item.runtime_requirement.strip():
            raise ValueError("invalid hybrid model identity")
        if item.parameter_count_b <= 0:
            raise ValueError("parameter_count_b must be positive")
        _positive_int(item.native_context, "native_context")
        if len({config.id for config in item.configurations}) != len(item.configurations):
            raise ValueError("duplicate hybrid configuration id")
        if any(config.context > item.native_context for config in item.configurations):
            raise ValueError("configuration exceeds native context without scaling evidence")
        return item

    def configuration(self, config_id: str) -> HybridConfiguration:
        for config in self.configurations:
            if config.id == config_id:
                return config
        raise KeyError(config_id)


@dataclass(frozen=True)
class HybridCatalog:
    models: dict[str, HybridModel]

    @classmethod
    def load(cls, path: str | Path) -> "HybridCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("hybrid catalog root must be an object")
        _exact(raw, {"schema", "models"}, "hybrid catalog")
        if raw["schema"] != SCHEMA:
            raise ValueError("unsupported hybrid catalog schema")
        raw_models = raw["models"]
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("hybrid catalog models must be a non-empty list")
        models = tuple(HybridModel.from_mapping(item) for item in raw_models)
        if len({model.id for model in models}) != len(models):
            raise ValueError("duplicate hybrid model id")
        return cls({model.id: model for model in models})


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields do not match schema")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
