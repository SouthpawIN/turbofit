from __future__ import annotations

from datetime import datetime, timezone

from turbofit_runtime.model_intel import build_snapshot


def test_model_intel_deduplicates_sources_and_separates_catalog_updates_from_discovery() -> None:
    catalog = {
        "models": [
            {"id": "a", "source": {"repo_id": "org/model-a", "revision": "old-a"}},
            {"id": "a-q4", "source": {"repo_id": "org/model-a", "revision": "old-a"}},
            {"id": "b", "source": {"repo_id": "org/model-b", "revision": "live-b"}},
        ]
    }

    def fetch(url: str):
        if "org/model-a" in url or "org%2Fmodel-a" in url:
            return {"id": "org/model-a", "sha": "live-a", "lastModified": "2026-07-25T00:00:00Z", "downloads": 12, "likes": 3}
        if "org/model-b" in url or "org%2Fmodel-b" in url:
            return {"id": "org/model-b", "sha": "live-b", "lastModified": "2026-07-24T00:00:00Z", "downloads": 7, "likes": 1}
        return [
            {"id": "new/release", "sha": "new-sha", "lastModified": "2026-07-26T00:00:00Z", "downloads": 99, "likes": 8},
            {"id": "org/model-a", "sha": "live-a", "lastModified": "2026-07-25T00:00:00Z", "downloads": 12, "likes": 3},
        ]

    snapshot = build_snapshot(catalog, fetch_json=fetch, now=datetime(2026, 7, 26, tzinfo=timezone.utc))

    assert [item["repo_id"] for item in snapshot["tracked"]] == ["org/model-a", "org/model-b"]
    assert snapshot["tracked"][0]["revision_changed"] is True
    assert snapshot["tracked"][1]["revision_changed"] is False
    assert [item["repo_id"] for item in snapshot["discoveries"]] == ["new/release"]
    assert snapshot["generated_at"] == "2026-07-26T00:00:00+00:00"
