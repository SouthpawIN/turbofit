"""Deterministic, candidate-only research queue updates."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "turbofit.research-candidates/v1"
FIELDS = frozenset({"id", "kind", "name", "url", "provenance", "metadata", "status"})
SECRET_KEYS = frozenset({"api_key", "token", "password", "authorization", "credential", "secret"})


@dataclass(frozen=True)
class CandidateDiff:
    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]


def update_candidates(
    path: str | Path,
    incoming: Iterable[Mapping[str, Any]],
    *,
    replace_kind: str | None = None,
) -> CandidateDiff:
    target = Path(path)
    current = _load(target)
    normalized = [_normalize(item) for item in incoming]
    incoming_by_id = {item["id"]: item for item in normalized}
    if len(incoming_by_id) != len(normalized):
        raise ValueError("duplicate candidate id")
    kinds = {replace_kind} if replace_kind else {item["kind"] for item in normalized}
    old_by_id = {item["id"]: item for item in current}
    kept = {
        key: value for key, value in old_by_id.items()
        if value["kind"] not in kinds
    }
    merged = {**kept, **incoming_by_id}
    ordered = [merged[key] for key in sorted(merged)]
    before = {item["id"]: item for item in current}
    after = {item["id"]: item for item in ordered}
    added = tuple(sorted(set(after) - set(before)))
    removed = tuple(sorted(
        key for key in set(before) - set(after) if before[key]["kind"] in kinds
    ))
    updated = tuple(sorted(
        key for key in set(before) & set(after) if before[key] != after[key]
    ))
    payload = {"schema": SCHEMA, "candidates": ordered}
    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        _atomic_write(target, content)
    return CandidateDiff(added, updated, removed)


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema", "candidates"}:
        raise ValueError("invalid candidate queue root")
    if data["schema"] != SCHEMA or not isinstance(data["candidates"], list):
        raise ValueError("unsupported candidate queue")
    return [_normalize(item) for item in data["candidates"]]


def _normalize(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != FIELDS:
        raise ValueError("candidate fields must be exact")
    if item.get("status") != "candidate":
        raise ValueError("research collectors may only write candidate status")
    for key in ("id", "kind", "name", "url"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"candidate {key} must be non-empty")
    provenance = item.get("provenance")
    metadata = item.get("metadata")
    if not isinstance(provenance, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("candidate provenance and metadata must be mappings")
    _reject_secrets(item)
    return {
        "id": item["id"].strip(),
        "kind": item["kind"].strip(),
        "name": item["name"].strip(),
        "url": item["url"].strip(),
        "provenance": _plain(provenance),
        "metadata": _plain(metadata),
        "status": "candidate",
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"non-portable candidate value: {type(value).__name__}")


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in SECRET_KEYS:
                raise ValueError(f"secret field forbidden: {key}")
            _reject_secrets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_secrets(child)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
