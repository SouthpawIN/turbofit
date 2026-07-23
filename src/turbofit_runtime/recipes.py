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
            return ResolvedComponent(
                role=role, family=family, alias=alias, kind=kind, method=method,
                gpu=gpu, port=port, command=(), image=str(spec["image"]),
                environment=environment, mounts=(f"{root}:/models:ro",),
                model_path=str(root / model),
                projector_path=str(root / projector) if projector else "",
            )
        if kind != "process":
            raise ValueError(f"unsupported recipe kind: {kind}")
        model = str(spec["model"])
        projector = str(spec.get("projector", ""))
        command = [
            self.atomic_binary, "-m", model,
            "--host", "127.0.0.1", "--port", str(port),
            "-c", str(context), "-ngl", "99", "--fit", "on", "-fa", "on",
            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
        ]
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
        if row.aux == "auto":
            main_gpu = "0,1" if main_large else "0"
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
        main = self._component(row.main, "main", row.context, "0,1" if main_large else "1")
        return ResolvedRecipe(
            row_id=row.id,
            profile_name=row.id,
            main_alias=main.alias,
            aux_alias=aux.alias,
            aux_mode="dedicated",
            components=(aux, main),
        )
