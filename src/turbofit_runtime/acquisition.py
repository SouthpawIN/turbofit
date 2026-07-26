"""Pinned, content-verified model acquisition through Turbohaul Manager."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


_SCHEMA = "turbofit.acquisitions/v1"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class AcquisitionError(RuntimeError):
    """A selected runtime artifact could not be acquired or verified safely."""


class AcquisitionClient(Protocol):
    def list_models(self) -> list[dict[str, Any]]: ...
    def pull_hf(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        expected_sha256: str,
    ) -> dict[str, Any]: ...
    def put_manifest(
        self,
        tag: str,
        manifest: dict[str, Any],
        *,
        etag: str | None = None,
    ) -> Any: ...
    def show_model(self, name: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ArtifactSource:
    id: str
    repo_id: str
    filename: str
    revision: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_mapping(cls, artifact_id: str, value: Mapping[str, Any]) -> "ArtifactSource":
        fields = {"repo_id", "filename", "revision", "sha256", "size_bytes"}
        if set(value) != fields:
            raise ValueError(f"artifact {artifact_id} fields do not match acquisition schema")
        item = cls(
            id=artifact_id,
            repo_id=str(value["repo_id"]),
            filename=str(value["filename"]),
            revision=str(value["revision"]),
            sha256=str(value["sha256"]),
            size_bytes=value["size_bytes"],
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not _TAG_RE.fullmatch(self.id):
            raise ValueError(f"invalid artifact id: {self.id}")
        if not _REPO_RE.fullmatch(self.repo_id):
            raise ValueError(f"artifact {self.id} has invalid repo_id")
        if not self.filename or self.filename.startswith("/") or ".." in Path(self.filename).parts:
            raise ValueError(f"artifact {self.id} has unsafe filename")
        if not _REVISION_RE.fullmatch(self.revision):
            raise ValueError(f"artifact {self.id} revision must be a pinned 40-character commit")
        if not _SHA_RE.fullmatch(self.sha256):
            raise ValueError(f"artifact {self.id} sha256 must be 64 lowercase hex characters")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError(f"artifact {self.id} size_bytes must be positive")


@dataclass(frozen=True)
class TagRecipe:
    tag: str
    artifact: str
    context_size: int
    expected_vram_bytes: int
    display_name: str
    description: str
    llama_server_flags: Mapping[str, Any]
    prompt_template: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, tag: str, value: Mapping[str, Any]) -> "TagRecipe":
        fields = {
            "artifact",
            "context_size",
            "expected_vram_bytes",
            "display_name",
            "description",
            "llama_server_flags",
            "prompt_template",
        }
        if set(value) != fields:
            raise ValueError(f"tag {tag} fields do not match acquisition schema")
        flags = value["llama_server_flags"]
        prompt = value["prompt_template"]
        if not isinstance(flags, Mapping) or not isinstance(prompt, Mapping):
            raise ValueError(f"tag {tag} runtime settings must be mappings")
        item = cls(
            tag=tag,
            artifact=str(value["artifact"]),
            context_size=value["context_size"],
            expected_vram_bytes=value["expected_vram_bytes"],
            display_name=str(value["display_name"]),
            description=str(value["description"]),
            llama_server_flags=dict(flags),
            prompt_template=dict(prompt),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not _TAG_RE.fullmatch(self.tag):
            raise ValueError(f"invalid model tag: {self.tag}")
        if not _TAG_RE.fullmatch(self.artifact):
            raise ValueError(f"tag {self.tag} has invalid artifact reference")
        for field in ("context_size", "expected_vram_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"tag {self.tag} {field} must be positive")
        if not self.display_name or not self.description:
            raise ValueError(f"tag {self.tag} requires display_name and description")
        if self.llama_server_flags.get("ctx_size") != self.context_size:
            raise ValueError(f"tag {self.tag} ctx_size must match context_size")
        if set(self.prompt_template) != {"system_default", "stop_tokens"}:
            raise ValueError(f"tag {self.tag} has invalid prompt_template")

    def manifest(self, artifact: ArtifactSource) -> dict[str, Any]:
        return {
            "model_tag": self.tag,
            "display_name": self.display_name,
            "description": self.description,
            "gguf_blob_sha256": artifact.sha256,
            "gguf_size_bytes": artifact.size_bytes,
            "context_size": self.context_size,
            "expected_vram_bytes": self.expected_vram_bytes,
            "revision": 1,
            "llama_server_flags": dict(self.llama_server_flags),
            "prompt_template": dict(self.prompt_template),
        }


@dataclass(frozen=True)
class AcquisitionCatalog:
    artifacts: Mapping[str, ArtifactSource]
    tags: Mapping[str, TagRecipe]

    @classmethod
    def load(cls, path: str | Path) -> "AcquisitionCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AcquisitionCatalog":
        if not isinstance(raw, Mapping) or set(raw) != {"schema", "artifacts", "tags"}:
            raise ValueError("invalid acquisition catalog root")
        if raw["schema"] != _SCHEMA:
            raise ValueError("unsupported acquisition catalog schema")
        artifacts_raw = raw["artifacts"]
        tags_raw = raw["tags"]
        if not isinstance(artifacts_raw, Mapping) or not isinstance(tags_raw, Mapping):
            raise ValueError("acquisition artifacts and tags must be mappings")
        artifacts = {
            str(key): ArtifactSource.from_mapping(str(key), _mapping(value, f"artifact {key}"))
            for key, value in artifacts_raw.items()
        }
        tags = {
            str(key): TagRecipe.from_mapping(str(key), _mapping(value, f"tag {key}"))
            for key, value in tags_raw.items()
        }
        if not artifacts or not tags:
            raise ValueError("acquisition catalog must not be empty")
        unknown = sorted({item.artifact for item in tags.values()} - set(artifacts))
        if unknown:
            raise ValueError(f"unknown acquisition artifacts: {unknown}")
        return cls(artifacts=artifacts, tags=tags)


class ModelAcquirer:
    """Ensure selected Turbohaul tags exist, pulling each missing blob once."""

    def __init__(self, catalog: AcquisitionCatalog, client: AcquisitionClient) -> None:
        self.catalog = catalog
        self.client = client

    def ensure_tags(self, tags: Iterable[str]) -> None:
        requested = tuple(dict.fromkeys(tags))
        unknown = [tag for tag in requested if tag not in self.catalog.tags]
        if unknown:
            raise AcquisitionError(f"no acquisition recipe for selected model tags: {unknown}")

        models = self.client.list_models()
        by_name = {
            str(item.get("name")): item
            for item in models
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        available_hashes = {
            digest.removeprefix("sha256:")
            for item in models
            if isinstance(item, Mapping)
            and isinstance((digest := item.get("digest")), str)
            and _SHA_RE.fullmatch(digest.removeprefix("sha256:"))
        }

        missing: list[TagRecipe] = []
        for tag in requested:
            recipe = self.catalog.tags[tag]
            artifact = self.catalog.artifacts[recipe.artifact]
            existing = by_name.get(tag)
            if existing is None:
                missing.append(recipe)
                continue
            _verify_model_record(existing, recipe, artifact)

        for recipe in missing:
            artifact = self.catalog.artifacts[recipe.artifact]
            if artifact.sha256 not in available_hashes:
                result = self.client.pull_hf(
                    repo_id=artifact.repo_id,
                    filename=artifact.filename,
                    revision=artifact.revision,
                    expected_sha256=artifact.sha256,
                )
                if result.get("status") != "complete" or result.get("sha256") != artifact.sha256:
                    raise AcquisitionError(
                        f"Turbohaul did not verify downloaded artifact {artifact.id}"
                    )
                available_hashes.add(artifact.sha256)
            self.client.put_manifest(recipe.tag, recipe.manifest(artifact))
            detail = self.client.show_model(recipe.tag)
            _verify_model_record(detail, recipe, artifact)


def _verify_model_record(
    model: Mapping[str, Any], recipe: TagRecipe, artifact: ArtifactSource
) -> None:
    digest = model.get("digest")
    if digest != f"sha256:{artifact.sha256}":
        raise AcquisitionError(
            f"selected tag {recipe.tag} points to {digest!r}, expected "
            f"'sha256:{artifact.sha256}'; Turbofit will not overwrite a conflicting tag"
        )
    raw_details = model.get("details")
    details: Mapping[str, Any] = raw_details if isinstance(raw_details, Mapping) else model
    context = details.get("context_length", details.get("context_size"))
    if context != recipe.context_size:
        raise AcquisitionError(f"selected tag {recipe.tag} has wrong context")
    if details.get("expected_vram_bytes") != recipe.expected_vram_bytes:
        raise AcquisitionError(f"selected tag {recipe.tag} has wrong VRAM requirement")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value
