"""Strict candidate catalog and complete main/aux/context configuration matrix."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CATALOG_SCHEMA = "turbofit.model-catalog/v2"
MATRIX_SCHEMA = "turbofit.configuration-matrix/v1"
CONTEXTS = (65_536, 131_072, 262_144, 1_048_576)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class ModelVariant:
    id: str
    name: str
    family: str
    source: str
    revision: str
    artifact: str
    quantization: str
    vision: bool
    moe: bool
    runtime_features: tuple[str, ...]
    roles: tuple[str, ...]
    upstream_source: str | None = None
    upstream_revision: str | None = None
    supported_contexts: tuple[int, ...] = CONTEXTS

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelVariant":
        required = {
            "id", "name", "family", "source", "revision", "artifact", "quantization",
            "vision", "moe", "runtime_features", "roles",
        }
        optional = {"upstream_source", "upstream_revision", "supported_contexts"}
        if not required <= set(value) or set(value) - required - optional:
            raise ValueError("model variant fields do not match catalog schema")
        item = cls(
            id=str(value["id"]),
            name=str(value["name"]),
            family=str(value["family"]),
            source=str(value["source"]),
            revision=str(value["revision"]),
            artifact=str(value["artifact"]),
            quantization=str(value["quantization"]),
            vision=value["vision"],
            moe=value["moe"],
            runtime_features=tuple(value["runtime_features"]),
            roles=tuple(value["roles"]),
            upstream_source=str(value["upstream_source"]) if value.get("upstream_source") else None,
            upstream_revision=str(value["upstream_revision"]) if value.get("upstream_revision") else None,
            supported_contexts=tuple(value.get("supported_contexts", CONTEXTS)),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not _SLUG_RE.fullmatch(self.id):
            raise ValueError(f"invalid model id: {self.id}")
        for name in ("name", "family", "source", "artifact", "quantization"):
            if not getattr(self, name).strip():
                raise ValueError(f"model {self.id} has empty {name}")
        if not self.source.startswith("https://huggingface.co/"):
            raise ValueError(f"model {self.id} source must be a Hugging Face URL")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(f"model {self.id} revision must be a pinned commit")
        if self.upstream_source and not self.upstream_source.startswith("https://huggingface.co/"):
            raise ValueError(f"model {self.id} upstream_source must be a Hugging Face URL")
        if self.upstream_revision and not re.fullmatch(r"[0-9a-f]{40}", self.upstream_revision):
            raise ValueError(f"model {self.id} upstream_revision must be a pinned commit")
        if self.upstream_revision and not self.upstream_source:
            raise ValueError(f"model {self.id} upstream_revision requires upstream_source")
        if not isinstance(self.vision, bool) or not isinstance(self.moe, bool):
            raise ValueError(f"model {self.id} capability flags must be booleans")
        if not self.roles or not set(self.roles) <= {"main", "aux"}:
            raise ValueError(f"model {self.id} has invalid roles")
        if len(self.runtime_features) != len(set(self.runtime_features)):
            raise ValueError(f"model {self.id} repeats a runtime feature")
        if not self.supported_contexts or not set(self.supported_contexts) <= set(CONTEXTS):
            raise ValueError(f"model {self.id} has invalid supported contexts")


@dataclass(frozen=True)
class ModelCatalog:
    models: tuple[ModelVariant, ...]
    contexts: tuple[int, ...]
    auxiliary_options: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ModelCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or set(raw) != {
            "schema", "contexts", "auxiliary_options", "models"
        }:
            raise ValueError("invalid model catalog root")
        if raw["schema"] != CATALOG_SCHEMA:
            raise ValueError("unsupported model catalog schema")
        catalog = cls(
            models=tuple(ModelVariant.from_mapping(item) for item in raw["models"]),
            contexts=tuple(raw["contexts"]),
            auxiliary_options=tuple(raw["auxiliary_options"]),
        )
        catalog.validate()
        return catalog

    def validate(self) -> None:
        ids = [item.id for item in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate model id")
        if self.contexts != CONTEXTS:
            raise ValueError("catalog must expose 64K, 128K, 262K, and 1M contexts")
        by_id = {item.id: item for item in self.models}
        if not self.auxiliary_options or self.auxiliary_options[-1] != "auto":
            raise ValueError("auxiliary options must end with auto")
        for model_id in self.auxiliary_options[:-1]:
            if model_id not in by_id or "aux" not in by_id[model_id].roles:
                raise ValueError(f"invalid auxiliary option: {model_id}")
        if any(not set(item.roles) & {"main", "aux"} for item in self.models):
            raise ValueError("every catalog model must be main-capable or auxiliary-capable")
        if not any("main" in item.roles for item in self.models):
            raise ValueError("catalog must expose at least one main model")

    @property
    def main_models(self) -> tuple[ModelVariant, ...]:
        return tuple(item for item in self.models if "main" in item.roles)


def build_configuration_matrix(catalog: ModelCatalog) -> dict[str, Any]:
    rows = []
    for main in catalog.main_models:
        for aux in catalog.auxiliary_options:
            for context in main.supported_contexts:
                rows.append({
                    "id": f"{main.id}--{aux}--{_context_slug(context)}",
                    "main": main.id,
                    "auxiliary": aux,
                    "context": context,
                    "status": "candidate",
                })
    return {"schema": MATRIX_SCHEMA, "rows": rows}


def validate_configuration_matrix(
    matrix: Mapping[str, Any], catalog: ModelCatalog
) -> None:
    if set(matrix) != {"schema", "rows"} or matrix["schema"] != MATRIX_SCHEMA:
        raise ValueError("invalid configuration matrix root")
    rows = matrix["rows"]
    expected = build_configuration_matrix(catalog)["rows"]
    if rows != expected:
        raise ValueError("configuration matrix is stale or incomplete")


def _context_slug(context: int) -> str:
    return {65_536: "64k", 131_072: "128k", 262_144: "262k", 1_048_576: "1m"}[context]
