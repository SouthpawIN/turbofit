#!/usr/bin/env python3
"""Collect RSS/Atom model news into the candidate queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from research.candidate_utils import update_candidates

ROOT = Path(__file__).resolve().parents[1]


def collect_model_news(feed: str, source_url: str) -> list[dict[str, Any]]:
    root = ET.fromstring(feed)
    items = list(root.findall(".//item"))
    if not items:
        items = list(root.findall(".//{*}entry"))
    result = []
    for item in items:
        title = _text(item, "title") or "Untitled model news"
        link = _text(item, "link")
        if not link:
            link_node = item.find("{*}link")
            link = link_node.get("href", "") if link_node is not None else ""
        if not link:
            continue
        guid = _text(item, "guid") or _text(item, "id")
        if not guid:
            guid = hashlib.sha256(link.encode()).hexdigest()
        published = _text(item, "pubDate") or _text(item, "published") or _text(item, "updated")
        result.append({
            "id": f"news:{guid}",
            "kind": "news",
            "name": title,
            "url": link,
            "provenance": {"source": source_url, "revision": guid, "published_at": published},
            "metadata": {},
            "status": "candidate",
        })
    return sorted(result, key=lambda candidate: candidate["id"])


def _text(node: ET.Element, name: str) -> str:
    child = node.find(name)
    if child is None:
        child = node.find(f"{{*}}{name}")
    return (child.text or "").strip() if child is not None else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "candidates.json")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url, timeout=30) as response:
        feed = response.read().decode("utf-8", errors="replace")
    diff = update_candidates(args.output, collect_model_news(feed, args.url), replace_kind="news")
    print(json.dumps(diff.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
