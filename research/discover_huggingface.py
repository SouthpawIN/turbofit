#!/usr/bin/env python3
"""Collect public Hugging Face model metadata into the candidate queue."""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from research.candidate_utils import update_candidates

DEFAULT_URL = "https://huggingface.co/api/models?sort=lastModified&direction=-1&limit=100"
ROOT = Path(__file__).resolve().parents[1]


def collect_huggingface(payload: Any, source_url: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Hugging Face payload must be a list")
    result = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        model_id = raw.get("id") or raw.get("modelId")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        raw_tags = raw.get("tags")
        tags: list[Any] = raw_tags if isinstance(raw_tags, list) else []
        result.append({
            "id": f"huggingface:{model_id}",
            "kind": "huggingface",
            "name": model_id,
            "url": f"https://huggingface.co/{model_id}",
            "provenance": {
                "source": source_url,
                "revision": raw.get("sha"),
                "published_at": raw.get("lastModified"),
            },
            "metadata": {
                "downloads": int(raw.get("downloads") or 0),
                "tags": sorted(str(tag) for tag in tags),
            },
            "status": "candidate",
        })
    return sorted(result, key=lambda item: item["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "candidates.json")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url, timeout=30) as response:
        payload = json.loads(response.read())
    diff = update_candidates(args.output, collect_huggingface(payload, args.url), replace_kind="huggingface")
    print(json.dumps(diff.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
