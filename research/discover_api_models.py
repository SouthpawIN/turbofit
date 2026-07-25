#!/usr/bin/env python3
"""Collect public OpenAI-compatible model catalogs without credentials."""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.candidate_utils import update_candidates

ROOT = Path(__file__).resolve().parents[1]


def collect_api_models(payload: Any, source_url: str, provider: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("API model payload must contain a data list")
    result = []
    for raw in payload["data"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        model_id = raw["id"].strip()
        if not model_id:
            continue
        created = raw.get("created")
        published = None
        if isinstance(created, (int, float)) and not isinstance(created, bool):
            published = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        result.append({
            "id": f"api:{provider}:{model_id}",
            "kind": "api",
            "name": model_id,
            "url": source_url,
            "provenance": {"source": source_url, "revision": None, "published_at": published},
            "metadata": {"provider": provider, "owned_by": str(raw.get("owned_by") or "")},
            "status": "candidate",
        })
    return sorted(result, key=lambda item: item["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "candidates.json")
    args = parser.parse_args()
    request = urllib.request.Request(args.url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    diff = update_candidates(args.output, collect_api_models(payload, args.url, args.provider), replace_kind="api")
    print(json.dumps(diff.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
