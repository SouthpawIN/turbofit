"""Model-family launch recipes for the Main:Aux campaign."""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .hardware import HardwareFingerprint
from .schema import MatrixRow


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


@dataclass(frozen=True)
class ResolvedRecipe:
    row_id: str
    profile_name: str
    main_alias: str
    aux_alias: str
    aux_mode: str
    components: tuple[ResolvedComponent, ...]


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

    def _spec(self, name: str) -> tuple[str, dict]:
        variant = self.variants.get(name)
        if variant is not None:
            family = str(variant.get("family", ""))
            if family not in self.models:
                raise ValueError(f"unknown recipe family for variant {name}: {family}")
            return family, {**self.models[family], **{key: value for key, value in variant.items() if key != "family"}}
        if name not in self.models:
            raise ValueError(f"unknown model family: {name}")
        return name, dict(self.models[name])

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
        _, spec = self._spec(family)
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
        binary = str(Path(str(binary_value)).expanduser())
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
            if spec.get("no_mmap"):
                command.append("--no-mmap")
            if spec.get("dsa"):
                command.extend(["-mla", "1", "-dsa", "-fidx"])
            if spec.get("worst_graph_tokens") is not None:
                command.extend(["-wgt", str(spec["worst_graph_tokens"])])
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
            command.extend(["--n-cpu-moe", str(n_cpu_moe)])
        if context_override.get("cpu_moe", spec.get("cpu_moe")) is True:
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
            command.extend([
                "--spec-type",
                "mtp:n_max=4,p_min=0.5" if runtime_flavor == "ik" else "draft-mtp",
            ])
        if method == "dspark":
            draft = artifact(str(spec.get("draft", ""))) if spec.get("draft") else ""
            if not draft:
                raise ValueError(f"DSpark recipe for {family} requires a draft model")
            command.extend([
                "--model-draft", draft,
                "--spec-type", "draft-dspark",
                "--spec-draft-n-max", str(spec.get("draft_n_max", 4)),
                "-ngld", str(
                    0 if self.backend_name == "cpu"
                    else context_override.get("draft_gpu_layers", "auto")
                ),
            ])
        if projector:
            command.extend(["--mmproj", projector])
        return ResolvedComponent(
            role=role, family=family, alias=alias, kind=kind, method=method,
            gpu=gpu, port=port, command=tuple(command),
            model_path=model, projector_path=projector,
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
    ) -> ResolvedComponent:
        """Resolve one native process without exposing private recipe structure."""
        if role not in {"main", "aux"}:
            raise ValueError("role must be main or aux")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in 1..65535")
        if context <= 0:
            raise ValueError("context must be positive")
        return self._component(
            family,
            role,
            context,
            gpu,
            port_override=port,
            alias_override=alias,
        )

    def resolve(self, row: MatrixRow) -> ResolvedRecipe:
        return self._resolve_values(row.id, row.main, row.aux, row.context)

    def resolve_catalog_configuration(self, value: dict) -> ResolvedRecipe:
        required = {"id", "main", "auxiliary", "context", "status"}
        if set(value) != required or value.get("status") != "candidate":
            raise ValueError("invalid catalog configuration")
        context = value["context"]
        if isinstance(context, bool) or not isinstance(context, int):
            raise ValueError("catalog context must be an integer")
        return self._resolve_values(
            str(value["id"]), str(value["main"]), str(value["auxiliary"]), context
        )

    def _resolve_values(self, row_id: str, main_name: str, aux_name: str, context: int) -> ResolvedRecipe:
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
            main_gpu = override_gpu or auto_main_gpu
            main = self._component(main_name, "main", context, main_gpu, port_override=11605)
            return ResolvedRecipe(
                row_id=row_id,
                profile_name=row_id,
                main_alias=main.alias,
                aux_alias=f"auto:{main.alias}",
                aux_mode="shared-main",
                components=(main,),
            )
        aux = self._component(aux_name, "aux", context, auto_aux_gpu, port_override=11610)
        main = self._component(
            main_name,
            "main",
            context,
            override_gpu or auto_main_gpu,
            port_override=11605,
        )
        return ResolvedRecipe(
            row_id=row_id,
            profile_name=row_id,
            main_alias=main.alias,
            aux_alias=aux.alias,
            aux_mode="dedicated",
            components=(aux, main),
        )
