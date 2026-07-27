"""Deterministic, source-linked model release intelligence."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

HF_API = "https://huggingface.co/api/models"
DISCOVERY_URL = f"{HF_API}?filter=gguf&sort=lastModified&direction=-1&limit=50&full=true"


def _source(model: Mapping[str, Any]) -> tuple[str, str | None]:
    nested = model.get("source")
    if isinstance(nested, str):
        prefix = "https://huggingface.co/"
        return (nested.removeprefix(prefix).strip("/"), None) if nested.startswith(prefix) else ("", None)
    if isinstance(nested, Mapping):
        return str(nested.get("repo_id") or ""), str(nested.get("revision")) if nested.get("revision") else None
    return str(model.get("source_repo") or ""), str(model.get("artifact_revision")) if model.get("artifact_revision") else None


def _normalized(item: Mapping[str, Any], *, pinned_revision: str | None = None) -> dict[str, Any]:
    live_revision = str(item.get("sha") or "")
    return {
        "repo_id": str(item.get("id") or item.get("modelId") or ""),
        "live_revision": live_revision,
        "pinned_revision": pinned_revision,
        "revision_changed": bool(pinned_revision and live_revision and pinned_revision != live_revision),
        "last_modified": item.get("lastModified"),
        "downloads": int(item.get("downloads") or 0),
        "likes": int(item.get("likes") or 0),
    }


def build_snapshot(
    catalog: Mapping[str, Any], *, fetch_json: Callable[[str], Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch tracked revisions plus newly-updated GGUF releases."""
    tracked_sources: dict[str, str | None] = {}
    for model in catalog.get("models") or []:
        if not isinstance(model, Mapping):
            continue
        repo_id, revision = _source(model)
        if repo_id:
            tracked_sources.setdefault(repo_id, revision)

    tracked = []
    for repo_id, revision in sorted(tracked_sources.items()):
        try:
            raw = fetch_json(f"{HF_API}/{quote(repo_id, safe='/')}")
            if not isinstance(raw, Mapping):
                raise ValueError(f"invalid Hugging Face metadata for {repo_id}")
            tracked.append(_normalized(raw, pinned_revision=revision))
        except Exception as exc:
            tracked.append({
                "repo_id": repo_id, "live_revision": "", "pinned_revision": revision,
                "revision_changed": False, "last_modified": None, "downloads": 0,
                "likes": 0, "error": f"{type(exc).__name__}: {exc}",
            })

    discovered_raw = fetch_json(DISCOVERY_URL)
    if not isinstance(discovered_raw, list):
        raise ValueError("invalid Hugging Face discovery response")
    discoveries = []
    for item in discovered_raw:
        if not isinstance(item, Mapping):
            continue
        normalized = _normalized(item)
        if normalized["repo_id"] and normalized["repo_id"] not in tracked_sources:
            discoveries.append(normalized)

    generated = now or datetime.now(timezone.utc)
    return {
        "schema": "turbofit.model-intel/v1",
        "generated_at": generated.astimezone(timezone.utc).isoformat(),
        "sources": [HF_API, DISCOVERY_URL],
        "tracked": tracked,
        "discoveries": discoveries,
    }
