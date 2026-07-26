"""Model-family launch recipes for the Main:Aux campaign."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import MatrixRow


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
    image: str = ""
    environment: dict[str, str] | None = None
    mounts: tuple[str, ...] = ()
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
    def __init__(self, data: dict) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported recipe schema: {data.get('schema_version')}")
        self.data = data
        self.models = data.get("models") or {}
        self.atomic_binary = str(data["atomic_binary"])

    @classmethod
    def load(cls, path: Path | str) -> "RecipeBook":
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _context_method(spec: dict, context: int) -> str:
        try:
            return str(spec["methods"][str(context)])
        except KeyError as exc:
            raise ValueError(f"no method recipe for context {context}") from exc

    def _component(self, family: str, role: str, context: int, gpu: str) -> ResolvedComponent:
        try:
            spec = self.models[family]
        except KeyError as exc:
            raise ValueError(f"unknown model family: {family}") from exc
        method = self._context_method(spec, context)
        context_override = (spec.get("context_overrides") or {}).get(str(context)) or {}
        kind = str(spec["kind"])
        alias = str(spec["alias"])
        port = int(spec["port"])
        if kind == "docker":
            root = Path(str(spec["model_root"]))
            model = str(spec["model"])
            projector = str(spec.get("projector", ""))
            environment = {
                "PORT": str(port),
                "CTX": str(context),
                "MODEL": f"/models/{model}",
                "MAIN_GPU": "0",
                "NGL": "99",
            }
            if projector:
                environment["MMPROJ"] = f"/models/{projector}"
            if method == "dspark":
                environment.update({
                    "DRAFT_MODEL": f"/models/{spec['draft']}",
                    "DRAFT_NGL": "99",
                    "SPEC_DRAFT_N_MAX": "4",
                })
            command: list[str] = []
            if context_override.get("split_mode"):
                command.extend(["--split-mode", str(context_override["split_mode"])])
            if context_override.get("tensor_split"):
                command.extend(["--tensor-split", str(context_override["tensor_split"])])
            scaling = spec.get("context_scaling") or {}
            native_context = int(spec.get("native_context", context))
            if context > native_context and scaling:
                command.extend([
                    "--rope-scaling", str(scaling["method"]),
                    "--rope-scale", str(scaling["scale"]),
                    "--yarn-orig-ctx", str(scaling["original_context"]),
                ])
            return ResolvedComponent(
                role=role, family=family, alias=alias, kind=kind, method=method,
                gpu=gpu, port=port, command=tuple(command), image=str(spec["image"]),
                environment=environment, mounts=(f"{root}:/models:ro",),
                model_path=str(root / model),
                projector_path=str(root / projector) if projector else "",
            )
        if kind != "process":
            raise ValueError(f"unsupported recipe kind: {kind}")
        model = str(spec["model"])
        projector = str(spec.get("projector", ""))
        fit = str(context_override.get("fit", "on"))
        command = [
            self.atomic_binary, "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(context), "-ngl", str(context_override.get("gpu_layers", 99)), "--fit", fit, "-fa", "on",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
        ]
        if context_override.get("split_mode"):
            command.extend(["--split-mode", str(context_override["split_mode"])])
        if context_override.get("tensor_split"):
            command.extend(["--tensor-split", str(context_override["tensor_split"])])
        if context_override.get("main_gpu") is not None:
            command.extend(["--main-gpu", str(context_override["main_gpu"])])
        if context_override.get("kv_offload") is False:
            command.append("--no-kv-offload")
        if context_override.get("n_cpu_moe") is not None:
            command.extend(["--n-cpu-moe", str(context_override["n_cpu_moe"])])
        scaling = spec.get("context_scaling") or {}
        native_context = int(spec.get("native_context", context))
        if context > native_context and scaling:
            command.extend([
                "--rope-scaling", str(scaling["method"]),
                "--rope-scale", str(scaling["scale"]),
                "--yarn-orig-ctx", str(scaling["original_context"]),
            ])
        if method == "mtp":
            command.extend(["--spec-type", "draft-mtp"])
        if projector:
            command.extend(["--mmproj", projector])
        return ResolvedComponent(
            role=role, family=family, alias=alias, kind=kind, method=method,
            gpu=gpu, port=port, command=tuple(command),
            model_path=model, projector_path=projector,
        )

    def resolve(self, row: MatrixRow) -> ResolvedRecipe:
        main_spec = self.models.get(row.main)
        if not main_spec:
            raise ValueError(f"no recipe for main family: {row.main}")
        main_large = bool(main_spec.get("large", False))
        main_override = (main_spec.get("context_overrides") or {}).get(str(row.context)) or {}
        override_gpu = str(main_override.get("gpu", ""))
        if row.aux == "auto":
            main_gpu = override_gpu or ("0,1" if main_large else "0")
            main = self._component(row.main, "main", row.context, main_gpu)
            return ResolvedRecipe(
                row_id=row.id,
                profile_name=row.id,
                main_alias=main.alias,
                aux_alias=f"auto:{main.alias}",
                aux_mode="shared-main",
                components=(main,),
            )
        aux = self._component(row.aux, "aux", row.context, "0")
        main = self._component(row.main, "main", row.context, override_gpu or ("0,1" if main_large else "1"))
        return ResolvedRecipe(
            row_id=row.id,
            profile_name=row.id,
            main_alias=main.alias,
            aux_alias=aux.alias,
            aux_mode="dedicated",
            components=(aux, main),
        )
