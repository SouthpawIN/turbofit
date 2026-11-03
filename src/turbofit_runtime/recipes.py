"""Model-family launch recipes for the Main:Aux campaign."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .allowed_lineup import expert_residency
from .hardware import HardwareFingerprint
from .schema import MatrixRow


# Typed launch overrides: validated keys only, never a raw shell escape.
# Values are ints, floats or bools that map 1:1 onto native launcher flags.
LAUNCH_OVERRIDE_FLAGS: dict[str, tuple[str, ...]] = {
    "threads": ("--threads",),
    "threads_batch": ("--threads-batch",),
    "batch_size": ("-b",),
    "ubatch_size": ("-ub",),
    "cache_ram": ("--cache-ram",),
    "cache_reuse": ("--cache-reuse",),
    "no_perf": ("--no-perf",),
    "parallel": ("--parallel",),
    "checkpoints": ("--ctx-checkpoints",),
    "slot_similarity": ("--slot-prompt-similarity",),
    "mtp_n_max": (),
    "no_context_shift": ("--no-context-shift",),
    "cache_type_k": ("--cache-type-k",),
    "cache_type_v": ("--cache-type-v",),
}
_LAUNCH_OVERRIDE_INT_KEYS = frozenset({"threads", "threads_batch", "parallel", "checkpoints", "mtp_n_max",
                                     "batch_size", "ubatch_size", "cache_ram", "cache_reuse"})
_LAUNCH_OVERRIDE_FLOAT_KEYS = frozenset({"slot_similarity"})
_LAUNCH_OVERRIDE_STR_KEYS = frozenset({"cache_type_k", "cache_type_v"})
_LAUNCH_OVERRIDE_BOOL_KEYS = frozenset({"no_context_shift", "no_perf"})
_MTP_N_MAX_RE = re.compile(r"^mtp:n_max=(\d+),p_min=(0(?:\.\d+)?)$")


def _validated_launch_overrides(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(raw) - set(LAUNCH_OVERRIDE_FLAGS)
    if unknown:
        raise ValueError(f"unknown launch override in {where}: {sorted(unknown)[0]}")
    validated: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _LAUNCH_OVERRIDE_INT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"launch override {key} in {where} must be a non-negative integer")
            if key in {"parallel", "batch_size", "ubatch_size"} and not 1 <= value <= 65536:
                raise ValueError(f"launch override {key} must be in 1..65536")
        elif key in _LAUNCH_OVERRIDE_FLOAT_KEYS:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"launch override {key} in {where} must be a finite number")
            validated[key] = float(value)
            if key == "slot_similarity" and not 0 <= value <= 1:
                raise ValueError("slot similarity must be in 0..1")
            continue
        elif key in _LAUNCH_OVERRIDE_STR_KEYS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"launch override {key} in {where} must be a non-empty string")
        elif key in _LAUNCH_OVERRIDE_BOOL_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"launch override {key} in {where} must be a boolean")
        validated[key] = value
    return validated


def _merge_launch_overrides(
    spec_overrides: Any, context_overrides: Any, context: int, family: str
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (spec_overrides, context_overrides):
        if source is None:
            continue
        validated = _validated_launch_overrides(
            source, f"{family} context {context}"
        )
        merged.update(validated)
    return merged


def resolve_native_backend(
    backend_name: str | None = None,
    *,
    platform_name: str | None = None,
    which=shutil.which,
) -> str:
    """Select the native accelerator without assuming NVIDIA is present."""
    explicit = str(
        backend_name or os.environ.get("TURBOFIT_ACCELERATOR_BACKEND") or ""
    ).lower()
    if explicit:
        if explicit not in {"cuda", "rocm", "metal", "vulkan", "cpu"}:
            raise ValueError(f"unsupported native backend: {explicit}")
        return explicit
    host = str(platform_name or sys.platform).lower()
    if host == "darwin":
        return "metal"
    if which("nvidia-smi"):
        return "cuda"
    if host.startswith("linux") and (
        which("rocm-smi") or which("amd-smi") or which("rocminfo")
    ):
        return "rocm"
    if which("vulkaninfo"):
        return "vulkan"
    return "cpu"


def resolve_native_binary(binary: str | Path, *, platform_name: str | None = None) -> str:
    """Resolve a manifest binary to the layout produced by the native builder."""
    path = Path(str(binary)).expanduser()
    host = str(platform_name or sys.platform).lower()
    if not host.startswith("win"):
        return str(path)
    if path.suffix.lower() == ".exe":
        return str(path)
    candidates = (
        path,
        path.parent / "Release" / f"{path.name}.exe",
        path.with_suffix(".exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    # Visual Studio's default multi-config generator places the executable here.
    return str(candidates[1])


@dataclass(frozen=True)
class ResolvedComponent:
    role: str
    family: str
    alias: str
    kind: str
    method: str
    gpu: str
    port: int
    command: tuple[str, ...]
    model_path: str = ""
    projector_path: str = ""
    context: int = 0


@dataclass(frozen=True)
class ResolvedRecipe:
    row_id: str
    profile_name: str
    main_alias: str
    aux_alias: str
    aux_mode: str
    components: tuple[ResolvedComponent, ...]
    main_context: int = 0
    aux_context: int = 0
    aux_residency: dict | None = None


class UnknownRecipeError(ValueError):
    """An unresolvable recipe name, carrying the closest declared candidates."""

    def __init__(self, requested: str, known: tuple[str, ...]) -> None:
        normalized = _normalized_recipe_name(requested)
        scored = sorted(
            ((_closeness(normalized, _normalized_recipe_name(name)), name) for name in known),
            reverse=True,
        )
        self.requested = requested
        self.candidates = tuple(name for score, name in scored if score > 0)[:5]
        hint = f"; did you mean one of {list(self.candidates)}" if self.candidates else ""
        super().__init__(f"unknown model family: {requested}{hint}")


def _normalized_recipe_name(name: str) -> str:
    """Fold the slugs that appear in the wild onto one comparable key.

    Hand-written profiles and resolutions use several spellings of the same
    model (``qwen38-27b-unleashed``, ``qwen3.8-27b-unleashed``,
    ``Qwen 3.8 27B``), so comparison is alphanumeric-only and case-folded.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _declared_aliases(entry: Any) -> tuple[str, ...]:
    if not isinstance(entry, Mapping):
        return ()
    found: list[str] = []
    alias = entry.get("alias")
    if isinstance(alias, str) and alias:
        found.append(alias)
    extra = entry.get("aliases")
    if isinstance(extra, list):
        found.extend(str(item) for item in extra if isinstance(item, str) and item)
    return tuple(found)


def _closeness(left: str, right: str) -> int:
    """Length of the longest common prefix, used only to order suggestions."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


class RecipeBook:
    def __init__(
        self,
        data: dict,
        *,
        platform_name: str | None = None,
        backend_name: str | None = None,
        hardware: HardwareFingerprint | None = None,
    ) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported recipe schema: {data.get('schema_version')}")
        self.data = data
        self.models = data.get("models") or {}
        self.variants = data.get("variants") or {}
        self.recipes_by_alias = self._build_alias_index()
        self.platform_name = platform_name or sys.platform
        self.hardware = hardware
        self.model_root = Path(
            os.environ.get("TURBOFIT_MODEL_ROOT", "~/Models/storage/gguf")
        ).expanduser()
        self.backend_name = resolve_native_backend(
            backend_name,
            platform_name=self.platform_name,
        )
        binaries = data.get("native_binaries") or {}
        try:
            self.atomic_binary = str(binaries[self.backend_name])
        except KeyError:
            self.atomic_binary = str(data["atomic_binary"])

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        platform_name: str | None = None,
        backend_name: str | None = None,
        hardware: HardwareFingerprint | None = None,
    ) -> "RecipeBook":
        return cls(
            json.loads(Path(path).read_text()),
            platform_name=platform_name,
            backend_name=backend_name,
            hardware=hardware,
        )

    @staticmethod
    def _context_method(spec: dict, context: int) -> str:
        try:
            return str(spec["methods"][str(context)])
        except KeyError as exc:
            raise ValueError(f"no method recipe for context {context}") from exc

    def _build_alias_index(self) -> dict[str, tuple[str, str]]:
        """Map every declared spelling of a recipe onto its canonical entry.

        Exact keys win over aliases so a variant alias can never shadow a
        model family of the same name.
        """
        index: dict[str, tuple[str, str]] = {}
        for kind, entries in (("model", self.models), ("variant", self.variants)):
            for key, entry in entries.items():
                for alias in _declared_aliases(entry):
                    index.setdefault(alias, (kind, key))
                    index.setdefault(_normalized_recipe_name(alias), (kind, key))
                index.setdefault(_normalized_recipe_name(key), (kind, key))
        return index

    def _spec(self, name: str) -> tuple[str, dict]:
        variant = self.variants.get(name)
        if variant is not None:
            family = str(variant.get("family", ""))
            if family not in self.models:
                raise ValueError(f"unknown recipe family for variant {name}: {family}")
            return family, {**self.models[family], **{key: value for key, value in variant.items() if key != "family"}}
        if name not in self.models:
            resolved = self.recipes_by_alias.get(name) or self.recipes_by_alias.get(
                _normalized_recipe_name(name)
            )
            if resolved is None:
                raise UnknownRecipeError(name, self._known_recipe_names())
            kind, key = resolved
            if kind == "variant":
                return self._spec(key)
            name = key
        return name, dict(self.models[name])

    def _known_recipe_names(self) -> tuple[str, ...]:
        names = set(self.variants) | set(self.models)
        for entry in list(self.variants.values()) + list(self.models.values()):
            for alias in _declared_aliases(entry):
                names.add(alias)
        return tuple(sorted(names))


    def _component(
        self,
        family: str,
        role: str,
        context: int,
        gpu: str,
        *,
        port_override: int | None = None,
        alias_override: str | None = None,
    ) -> ResolvedComponent:
        resolved_family, spec = self._spec(family)
        if self.hardware is not None:
            available = {str(device.index) for device in self.hardware.devices}
            requested = [item for item in gpu.split(",") if item in available]
            if not available:
                gpu = ""
            elif self.hardware.memory_pool_kind == "unified":
                gpu = str(self.hardware.devices[0].index)
            else:
                gpu = ",".join(requested) or str(self.hardware.devices[0].index)
        method = self._context_method(spec, context)
        context_override = (spec.get("context_overrides") or {}).get(str(context)) or {}
        launch_overrides = _merge_launch_overrides(
            spec.get("launch_overrides"),
            context_override.get("launch_overrides"),
            context,
            family,
        )
        kind = str(spec["kind"])
        alias = alias_override or str(spec["alias"])
        port = port_override or int(spec["port"])
        component_binaries = spec.get("native_binaries") or {}
        binary_value = component_binaries.get(self.backend_name)
        if binary_value is None and spec.get("binary"):
            binary_value = str(spec["binary"])
            for backend in ("cuda", "rocm", "metal", "vulkan", "cpu"):
                binary_value = binary_value.replace(
                    f"/build-{backend}/", f"/build-{self.backend_name}/"
                )
        if binary_value is None:
            binary_value = self.atomic_binary
        binary = resolve_native_binary(binary_value, platform_name=self.platform_name)
        if kind != "process":
            raise ValueError(f"unsupported recipe kind: {kind}")
        root_value = str(spec.get("model_root", "")).replace(
            "${TURBOFIT_MODEL_ROOT}", str(self.model_root)
        )
        root = Path(root_value).expanduser()

        def artifact(value: str) -> str:
            path = Path(
                value.replace("${TURBOFIT_MODEL_ROOT}", str(self.model_root))
            ).expanduser()
            return str(path if path.is_absolute() or not root else root / path)

        model = artifact(str(spec["model"]))
        projector = artifact(str(spec.get("projector", ""))) if spec.get("projector") else ""
        fit_value = context_override.get("fit", spec.get("fit", "on"))
        fit = "on" if fit_value is True or str(fit_value).lower() in {"on", "true", "1"} else "off"
        default_batch = 512 if spec.get("large", False) else 2048
        default_ubatch = 64 if spec.get("large", False) else 512
        batch_size = int(context_override.get("batch_size", spec.get("batch_size", default_batch)))
        ubatch_size = int(context_override.get("ubatch_size", spec.get("ubatch_size", default_ubatch)))
        if not 1 <= ubatch_size <= batch_size:
            raise ValueError(f"invalid batch/ubatch recipe for {family}: {batch_size}/{ubatch_size}")
        runtime_flavor = str(spec.get("runtime_flavor", "mainline"))
        gpu_layers = context_override.get(
            "gpu_layers", 99 if runtime_flavor == "ik" else "auto"
        )
        if self.backend_name == "cpu":
            gpu_layers = 0
        if runtime_flavor == "ik":
            command = [
                binary, "-m", model,
                "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
                "-c", str(context), "-ngl", str(gpu_layers),
                "-fa", "on", "--jinja", "-b", str(batch_size), "-ub", str(ubatch_size),
                "--parallel", "1",
            ]
            if fit == "on":
                command.append("--fit")
            if spec.get("dsa"):
                command.extend(["-mla", "1", "-dsa", "-fidx"])
            if spec.get("worst_graph_tokens") is not None:
                command.extend(["-wgt", str(spec["worst_graph_tokens"])])
        elif runtime_flavor == "prism":
            # PrismML llama.cpp fork for Ternary Bonsai 2 (CUDA, Metal, CPU)
            # Ternary models need the prism fork — mainline llama.cpp cannot load them
            command = [
                binary, "-m", model,
                "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
                "-c", str(context), "-ngl", str(gpu_layers),
                "-fa", "on", "--jinja", "-b", str(batch_size), "-ub", str(ubatch_size),
                "--parallel", "1", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
            ]
            if fit == "on":
                command.append("--fit")
        elif runtime_flavor == "strata":
            # Strata engine: <binary> <binary_args> --family <family> --model <quant>
            # Uses expert tiering: hot experts on GPU, all experts in RAM
            binary_args = spec.get("binary_args", ["setup.py", "serve"])
            if "serve" in binary_args:
                binary_args = [arg for arg in binary_args if arg != "serve"]
            strata_family = spec.get("strata_family", "qwen")
            strata_model = spec.get("strata_model", "IQ2_XS")
            command = [binary, *binary_args, "--family", strata_family, "--model", strata_model]
            command.extend([
                "--context", str(context),
                "--port", str(port),
                "--gpu", "0",
                "--yes", "--no-browser",
            ])
            if spec.get("data_dir"):
                command.extend(["--data-dir", spec["data_dir"]])
        elif runtime_flavor == "mainline":
            command = [
                binary, "-m", model,
                "--host", "127.0.0.1", "--port", str(port), "--alias", alias,
                "-c", str(context), "-ngl", str(gpu_layers), "--fit", fit, "-fa", "on", "--jinja",
                "-b", str(batch_size), "-ub", str(ubatch_size),
                "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
            ]
        else:
            raise ValueError(f"unsupported runtime flavor for {family}: {runtime_flavor}")
        if spec.get("no_mmap"):
            command.append("--no-mmap")
        if spec.get("mlock"):
            command.append("--mlock")
        multi_device = (
            self.hardware is None
            or (
                self.hardware.memory_pool_kind == "dedicated"
                and len([item for item in gpu.split(",") if item]) > 1
            )
        )
        accelerator_split = self.backend_name in {"cuda", "rocm", "vulkan"} and multi_device
        if context_override.get("split_mode") and accelerator_split:
            command.extend(["--split-mode", str(context_override["split_mode"])])
        if context_override.get("tensor_split") and accelerator_split:
            command.extend(["--tensor-split", str(context_override["tensor_split"])])
        if context_override.get("main_gpu") is not None and self.backend_name in {"cuda", "rocm", "vulkan"}:
            command.extend(["--main-gpu", str(context_override["main_gpu"])])
        n_cpu_moe = context_override.get("n_cpu_moe", spec.get("n_cpu_moe"))
        if n_cpu_moe is not None:
            command.extend([ "--n-cpu-moe", str(n_cpu_moe)])
        if context_override.get("cpu_moe", spec.get("cpu_moe")) is True:
            command.append("--cpu-moe")
        # Aux hard-drive/RAM streaming (allowed_lineup.expert_residency), applied to
        # the auxiliary role only: offload MoE experts to the host and keep weights
        # memory-mapped so the aux streams from the hard drive when host RAM is tight.
        if role == "aux" and self.hardware is not None:
            residency = expert_residency(host_ram_gb=self.hardware.system_ram_mb / 1024)
            if residency.get("mmap") is True and "--no-mmap" in command:
                command.remove("--no-mmap")
            if residency.get("mlock") is True and "--mlock" not in command:
                command.append("--mlock")
            if residency.get("cpu_moe") is True and "--cpu-moe" not in command:
                command.append("--cpu-moe")
        scaling = spec.get("context_scaling") or {}
        native_context = int(spec.get("native_context", context))
        host_kv = bool(
            self.hardware is not None
            and self.hardware.memory_pool_kind == "dedicated"
            and context > native_context
            and self.hardware.host_usable_memory_mb >= 32 * 1024
        )
        if context_override.get("kv_offload") is False or host_kv:
            command.append("--no-kv-offload")
        if context > native_context and scaling:
            command.extend([
                "--rope-scaling", str(scaling["method"]),
                "--rope-scale", str(scaling["scale"]),
                "--yarn-orig-ctx", str(scaling["original_context"]),
            ])
        if method == "mtp":
            draft = artifact(str(spec.get("draft", ""))) if spec.get("draft") else ""
            if draft:
                command.extend(["--model-draft", draft])
            overrides = launch_overrides
            if runtime_flavor == "ik":
                spec_type = "mtp:n_max=4,p_min=0.5"
                if "mtp_n_max" in overrides:
                    match = _MTP_N_MAX_RE.match(spec_type)
                    n_max = int(overrides["mtp_n_max"])
                    p_min = match.group(2) if match else "0.5"
                    spec_type = f"mtp:n_max={n_max},p_min={p_min}"
                command.extend(["--spec-type", spec_type])
            else:
                command.extend(["--spec-type", "draft-mtp"])
                if "mtp_n_max" in overrides:
                    command.extend(["--spec-draft-n-max", str(overrides["mtp_n_max"])])
        if method in {"dspark", "dflash2"}:
            draft = artifact(str(spec.get("draft", ""))) if spec.get("draft") else ""
            label = "DFlash2" if method == "dflash2" else "DSpark"
            if not draft:
                raise ValueError(f"{label} recipe for {family} requires a draft model")
            if method == "dflash2" and spec.get("dflash_target_family") != resolved_family:
                raise ValueError(f"{family} requires a dedicated DFlash2 checkpoint bound to its target family")
            command.extend([
                "--model-draft", draft,
                "--spec-type", "draft-dflash" if method == "dflash2" else "draft-dspark",
                "--spec-draft-n-max", str(spec.get("draft_n_max", 7 if method == "dflash2" else 4)),
                "-ngld", str(
                    0 if self.backend_name == "cpu"
                    else context_override.get("draft_gpu_layers", "auto")
                ),
            ])
        if projector:
            command.extend(["--mmproj", projector])
        for override_key, flag_name in (
            ("threads", "--threads"),
            ("threads_batch", "--threads-batch"),
            ("batch_size", "-b"),
            ("ubatch_size", "-ub"),
            ("cache_ram", "--cache-ram"),
            ("cache_reuse", "--cache-reuse"),
            ("parallel", "--parallel"),
            ("checkpoints", "--ctx-checkpoints"),
            ("slot_similarity", "--slot-prompt-similarity"),
            ("cache_type_k", "--cache-type-k"),
            ("cache_type_v", "--cache-type-v"),
        ):
            if override_key in launch_overrides:
                value = launch_overrides[override_key]
                if isinstance(value, float):
                    value = str(value)
                if flag_name in command:
                    command[command.index(flag_name) + 1] = str(value)
                else:
                    command.extend([flag_name, str(value)])
        if launch_overrides.get("no_context_shift") is True:
            command.append("--no-context-shift")
        if launch_overrides.get("no_perf") is True:
            command.append("--no-perf")
        if {"batch_size", "ubatch_size"} & launch_overrides.keys():
            if int(command[command.index("-ub") + 1]) > int(command[command.index("-b") + 1]):
                raise ValueError("microbatch must not exceed batch size")
        return ResolvedComponent(
            role=role, family=family, alias=alias, kind=kind, method=method,
            gpu=gpu, port=port, command=tuple(command),
            model_path=model, projector_path=projector, context=context,
        )

    def resolve_component(
        self,
        family: str,
        *,
        role: str,
        gpu: str,
        port: int,
        context: int,
        alias: str | None = None,
        aux_context: int | None = None,
    ) -> ResolvedComponent:
        """Resolve one native process without exposing private recipe structure."""
        if role not in {"main", "aux"}:
            raise ValueError("role must be main or aux")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in 1..65535")
        if context <= 0:
            raise ValueError("context must be positive")
        if role == "aux" and aux_context is not None:
            if aux_context <= 0:
                raise ValueError("context must be positive")
            context = aux_context
        return self._component(
            family,
            role,
            context,
            gpu,
            port_override=port,
            alias_override=alias,
        )

    def resolve_named(
        self,
        row_id: str,
        main_name: str,
        aux_name: str,
        context: int,
        *,
        aux_context: int | None = None,
    ) -> ResolvedRecipe:
        """Resolve a named main/aux/context combination without a MatrixRow."""
        return self._resolve_values(
            row_id, main_name, aux_name, context, aux_context=aux_context
        )

    def resolve(self, row: MatrixRow) -> ResolvedRecipe:
        return self._resolve_values(row.id, row.main, row.aux, row.context)

    def resolve_catalog_configuration(self, value: dict) -> ResolvedRecipe:
        required = {"id", "main", "auxiliary", "context", "status"}
        optional = {"auxiliary_context"}
        keys = set(value)
        if not required <= keys or keys - required - optional:
            raise ValueError("invalid catalog configuration")
        if value.get("status") != "candidate":
            raise ValueError("invalid catalog configuration")
        context = value["context"]
        if isinstance(context, bool) or not isinstance(context, int):
            raise ValueError("catalog context must be an integer")
        aux_context = value.get("auxiliary_context")
        if aux_context is not None:
            if isinstance(aux_context, bool) or not isinstance(aux_context, int):
                raise ValueError("catalog auxiliary context must be an integer")
        return self._resolve_values(
            str(value["id"]), str(value["main"]), str(value["auxiliary"]), context,
            aux_context=aux_context,
        )

    def _resolve_values(
        self,
        row_id: str,
        main_name: str,
        aux_name: str,
        context: int,
        *,
        aux_context: int | None = None,
    ) -> ResolvedRecipe:
        _, main_spec = self._spec(main_name)
        main_large = bool(main_spec.get("large", False))
        main_override = (main_spec.get("context_overrides") or {}).get(str(context)) or {}
        override_gpu = str(main_override.get("gpu", ""))
        if self.hardware is not None:
            device_ids = [str(device.index) for device in self.hardware.devices]
            if not device_ids:
                auto_main_gpu = auto_aux_gpu = ""
            elif self.hardware.memory_pool_kind == "unified" or len(device_ids) == 1:
                auto_main_gpu = auto_aux_gpu = device_ids[0]
            else:
                auto_aux_gpu = device_ids[0]
                auto_main_gpu = (
                    ",".join(device_ids)
                    if aux_name == "auto" and (
                        main_large or context > int(main_spec.get("native_context", context))
                    )
                    else device_ids[-1]
                )
            # Portable hardware planning supersedes topology-specific catalog hints.
            override_gpu = ""
        else:
            auto_aux_gpu = "0"
            auto_main_gpu = "0,1" if main_large else ("0" if aux_name == "auto" else "1")
        if aux_name == "auto":
            if aux_context is not None and aux_context != context:
                raise ValueError(
                    "shared-main profile cannot use an auxiliary context distinct from the main context"
                )
            main_gpu = override_gpu or auto_main_gpu
            main = self._component(main_name, "main", context, main_gpu, port_override=11605)
            return ResolvedRecipe(
                row_id=row_id,
                profile_name=row_id,
                main_alias=main.alias,
                aux_alias=f"auto:{main.alias}",
                aux_mode="shared-main",
                components=(main,),
                main_context=context,
                aux_context=context,
            )
        aux_role_context = context if aux_context is None else aux_context
        aux = self._component(aux_name, "aux", aux_role_context, auto_aux_gpu, port_override=11610)
        main = self._component(
            main_name,
            "main",
            context,
            override_gpu or auto_main_gpu,
            port_override=11605,
        )
        aux_residency = (
            expert_residency(host_ram_gb=self.hardware.system_ram_mb / 1024)
            if self.hardware is not None
            else None
        )
        return ResolvedRecipe(
            row_id=row_id,
            profile_name=row_id,
            main_alias=main.alias,
            aux_alias=aux.alias,
            aux_mode="dedicated",
            components=(aux, main),
            main_context=context,
            aux_context=aux_role_context,
            aux_residency=aux_residency,
        )
