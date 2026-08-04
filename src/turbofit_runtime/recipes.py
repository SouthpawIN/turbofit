"""Model-family launch recipes for the Main:Aux campaign."""
from __future__ import annotations

import json
import sys
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
    def __init__(self, data: dict, *, platform_name: str | None = None) -> None:
        if data.get("schema_version") != 1:
            raise ValueError(f"unsupported recipe schema: {data.get('schema_version')}")
        self.data = data
        self.models = data.get("models") or {}
        self.variants = data.get("variants") or {}
        self.atomic_binary = str(data["atomic_binary"])
        self.platform_name = platform_name or sys.platform

    @classmethod
    def load(cls, path: Path | str, *, platform_name: str | None = None) -> "RecipeBook":
        return cls(json.loads(Path(path).read_text()), platform_name=platform_name)

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

    def _component(self, family: str, role: str, context: int, gpu: str) -> ResolvedComponent:
        _, spec = self._spec(family)
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
                    "SPEC_DRAFT_N_MAX": str(spec.get("draft_n_max", 4)),
                })
            command: list[str] = ["--jinja"]
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
            if self.platform_name == "darwin":
                native_command = [
                    self.atomic_binary, "-m", str(root / model),
                    "--host", "127.0.0.1", "--port", str(port),
                    "-c", str(context), "-ngl", "99", "-fa", "on",
                    *command,
                ]
                if method == "dspark":
                    native_command.extend([
                        "--model-draft", str(root / str(spec["draft"])),
                        "--spec-type", "draft-dspark",
                        "--spec-draft-n-max", str(spec.get("draft_n_max", 4)),
                        "-ngld", "99",
                    ])
                if projector:
                    native_command.extend(["--mmproj", str(root / projector)])
                return ResolvedComponent(
                    role=role, family=family, alias=alias, kind="process", method=method,
                    gpu=gpu, port=port, command=tuple(native_command),
                    model_path=str(root / model),
                    projector_path=str(root / projector) if projector else "",
                )
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
            "-c", str(context), "-ngl", str(context_override.get("gpu_layers", 99)), "--fit", fit, "-fa", "on", "--jinja",
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
        if method == "dspark":
            draft = str(spec.get("draft", ""))
            if not draft:
                raise ValueError(f"DSpark recipe for {family} requires a draft model")
            command.extend([
                "--model-draft", draft,
                "--spec-type", "draft-dspark",
                "--spec-draft-n-max", str(spec.get("draft_n_max", 4)),
                "-ngld", "99",
            ])
        if projector:
            command.extend(["--mmproj", projector])
        return ResolvedComponent(
            role=role, family=family, alias=alias, kind=kind, method=method,
            gpu=gpu, port=port, command=tuple(command),
            model_path=model, projector_path=projector,
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
        if aux_name == "auto":
            main_gpu = override_gpu or ("0,1" if main_large else "0")
            main = self._component(main_name, "main", context, main_gpu)
            return ResolvedRecipe(
                row_id=row_id,
                profile_name=row_id,
                main_alias=main.alias,
                aux_alias=f"auto:{main.alias}",
                aux_mode="shared-main",
                components=(main,),
            )
        aux = self._component(aux_name, "aux", context, "0")
        main = self._component(main_name, "main", context, override_gpu or ("0,1" if main_large else "1"))
        return ResolvedRecipe(
            row_id=row_id,
            profile_name=row_id,
            main_alias=main.alias,
            aux_alias=aux.alias,
            aux_mode="dedicated",
            components=(aux, main),
        )
