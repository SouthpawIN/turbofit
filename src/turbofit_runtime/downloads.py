"""Pinned multi-file model download catalog."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SCHEMA = "turbofit.downloads/v1"
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class DownloadFile:
    id: str
    repo_id: str
    revision: str
    path: str
    destination: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_mapping(cls, file_id: str, value: Mapping[str, Any]) -> "DownloadFile":
        fields = {"repo_id", "revision", "path", "destination", "sha256", "size_bytes"}
        if set(value) != fields:
            raise ValueError(f"download file {file_id} fields do not match schema")
        item = cls(
            id=file_id,
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            path=str(value["path"]),
            destination=str(value["destination"]),
            sha256=str(value["sha256"]),
            size_bytes=value["size_bytes"],
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not _ID_RE.fullmatch(self.id):
            raise ValueError(f"invalid download file id: {self.id}")
        if not _REPO_RE.fullmatch(self.repo_id):
            raise ValueError(f"download file {self.id} has invalid repo_id")
        if not _REVISION_RE.fullmatch(self.revision):
            raise ValueError(f"download file {self.id} revision must be pinned")
        if not _SHA_RE.fullmatch(self.sha256):
            raise ValueError(f"download file {self.id} sha256 must be lowercase hex")
        for label, raw in (("path", self.path), ("destination", self.destination)):
            path = Path(raw)
            if not raw or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"download file {self.id} has unsafe {label}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError(f"download file {self.id} size_bytes must be positive")

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{self.path}"


@dataclass(frozen=True)
class DownloadCatalog:
    files: Mapping[str, DownloadFile]
    groups: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: str | Path) -> "DownloadCatalog":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DownloadCatalog":
        if not isinstance(raw, Mapping) or set(raw) != {"schema", "files", "groups"}:
            raise ValueError("invalid download catalog root")
        if raw["schema"] != _SCHEMA:
            raise ValueError("unsupported download catalog schema")
        raw_files = raw["files"]
        raw_groups = raw["groups"]
        if not isinstance(raw_files, Mapping) or not isinstance(raw_groups, Mapping):
            raise ValueError("download files and groups must be mappings")
        files = {
            str(file_id): DownloadFile.from_mapping(str(file_id), _mapping(value, f"file {file_id}"))
            for file_id, value in raw_files.items()
        }
        groups: dict[str, tuple[str, ...]] = {}
        for group, members in raw_groups.items():
            group_id = str(group)
            if not _ID_RE.fullmatch(group_id) or not isinstance(members, list) or not members:
                raise ValueError(f"invalid download group: {group_id}")
            normalized = tuple(str(member) for member in members)
            unknown = sorted(set(normalized) - set(files))
            if unknown:
                raise ValueError(f"download group {group_id} references unknown files: {unknown}")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"download group {group_id} repeats files")
            groups[group_id] = normalized
        if not files or not groups:
            raise ValueError("download catalog must not be empty")
        return cls(files=files, groups=groups)

    def files_for_group(self, group: str) -> tuple[DownloadFile, ...]:
        if group not in self.groups:
            raise KeyError(f"unknown download group: {group}")
        return tuple(self.files[file_id] for file_id in self.groups[group])


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value
