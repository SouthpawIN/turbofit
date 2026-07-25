from __future__ import annotations

import json
from pathlib import Path

from research.candidate_utils import update_candidates
from research.discover_api_models import collect_api_models
from research.discover_huggingface import collect_huggingface
from research.discover_model_news import collect_model_news


def test_huggingface_collector_normalizes_public_model_metadata() -> None:
    payload = [{
        "id": "org/model-a",
        "sha": "abc123",
        "lastModified": "2026-07-24T00:00:00Z",
        "downloads": 42,
        "tags": ["text-generation", "gguf"],
    }]

    candidates = collect_huggingface(payload, "https://huggingface.co/api/models")

    assert candidates == [{
        "id": "huggingface:org/model-a",
        "kind": "huggingface",
        "name": "org/model-a",
        "url": "https://huggingface.co/org/model-a",
        "provenance": {
            "source": "https://huggingface.co/api/models",
            "revision": "abc123",
            "published_at": "2026-07-24T00:00:00Z",
        },
        "metadata": {"downloads": 42, "tags": ["gguf", "text-generation"]},
        "status": "candidate",
    }]


def test_model_news_collector_parses_rss_without_html_or_credentials() -> None:
    feed = """<rss><channel><item><title>Model A released</title><link>https://example.test/a</link><guid>a1</guid><pubDate>Fri, 24 Jul 2026 00:00:00 GMT</pubDate></item></channel></rss>"""

    candidates = collect_model_news(feed, "https://example.test/feed.xml")

    assert candidates[0]["id"] == "news:a1"
    assert candidates[0]["url"] == "https://example.test/a"
    assert candidates[0]["status"] == "candidate"
    assert "api_key" not in json.dumps(candidates).lower()


def test_api_model_collector_never_copies_credentials() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "provider/model-b", "created": 1784851200, "owned_by": "provider", "api_key": "secret"}
        ],
    }

    candidates = collect_api_models(payload, "https://api.example.test/v1/models", "provider")

    text = json.dumps(candidates)
    assert candidates[0]["id"] == "api:provider:provider/model-b"
    assert candidates[0]["status"] == "candidate"
    assert "secret" not in text
    assert "api_key" not in text


def test_candidate_update_is_diff_only_deterministic_and_never_promotes(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    first = [{
        "id": "huggingface:org/model-a",
        "kind": "huggingface",
        "name": "org/model-a",
        "url": "https://huggingface.co/org/model-a",
        "provenance": {"source": "https://huggingface.co/api/models", "revision": "a", "published_at": None},
        "metadata": {},
        "status": "candidate",
    }]

    diff = update_candidates(path, first)
    original = path.read_bytes()
    same = update_candidates(path, list(reversed(first)))

    assert diff.added == ("huggingface:org/model-a",)
    assert same.added == same.updated == same.removed == ()
    assert path.read_bytes() == original
    data = json.loads(path.read_text())
    assert data["schema"] == "turbofit.research-candidates/v1"
    assert all(item["status"] == "candidate" for item in data["candidates"])


def test_updates_preserve_other_collector_kinds_and_report_changes(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    hf = collect_huggingface([{"id": "org/a", "sha": "1"}], "https://hf.test/api")
    api = collect_api_models({"data": [{"id": "model-b"}]}, "https://api.test/models", "p")
    update_candidates(path, hf)

    diff = update_candidates(path, api, replace_kind="api")
    data = json.loads(path.read_text())

    assert diff.added == ("api:p:model-b",)
    assert {item["kind"] for item in data["candidates"]} == {"huggingface", "api"}
