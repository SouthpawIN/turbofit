"""Connect public discovery candidates to the strict benchmark/onboarding pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse


def _repo_id(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path.strip("/") if parsed.scheme else value.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path


def build_discovery_queue(
    candidates: Mapping[str, Any], catalog: Mapping[str, Any],
) -> dict[str, Any]:
    by_repo: dict[str, list[Mapping[str, Any]]] = {}
    for model in catalog.get("models") or []:
        if not isinstance(model, Mapping):
            continue
        for field in ("source", "upstream_source"):
            source = model.get(field)
            if isinstance(source, str) and source:
                by_repo.setdefault(_repo_id(source), []).append(model)
    entries = []
    for candidate in candidates.get("candidates") or []:
        if not isinstance(candidate, Mapping) or candidate.get("kind") != "huggingface":
            continue
        repo = _repo_id(str(candidate.get("url") or candidate.get("name") or ""))
        tracked = by_repo.get(repo, [])
        live_revision = (candidate.get("provenance") or {}).get("revision") if isinstance(candidate.get("provenance"), Mapping) else None
        catalog_revisions = {
            str(value)
            for model in tracked
            for value in (model.get("revision"), model.get("upstream_revision"))
            if value
        }
        if not tracked:
            status = "needs-onboarding-spec"
        elif live_revision and live_revision not in catalog_revisions:
            status = "needs-repin"
        else:
            status = "ready-for-catalog-campaign"
        entries.append({
            "candidate_id": candidate.get("id"),
            "repo_id": repo,
            "revision": live_revision,
            "status": status,
            "catalog_model_ids": sorted(str(model.get("id")) for model in tracked if model.get("id")),
            "next_action": {
                "needs-onboarding-spec": "author strict released-spec, then run turbofit-model-onboard",
                "needs-repin": "review changed upstream revision and regenerate immutable artifact metadata",
                "ready-for-catalog-campaign": "run current recipe physical campaign",
            }[status],
        })
    entries.sort(key=lambda item: (item["status"], item["candidate_id"] or ""))
    return {
        "schema": "turbofit.discovery-benchmark-queue/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
