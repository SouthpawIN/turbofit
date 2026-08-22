from __future__ import annotations

from pathlib import Path

import pytest

from turbofit_runtime.downloads import DownloadCatalog


ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "runtime-profiles" / "downloads.json"


def test_download_catalog_contains_complete_pinned_unleashed_group() -> None:
    catalog = DownloadCatalog.load(CATALOG)
    files = catalog.files_for_group("qwen3-8-27b-unleashed")

    assert len(files) == 3
    assert {item.revision for item in files} == {"67a999218fd7002f11bf82bc81d6289beea60841"}
    assert sum("UD-Q3_K_XL" in item.path or "UD-IQ3_XXS" in item.path for item in files) == 2
    assert sum("mmproj" in item.path for item in files) == 1
    assert all(len(item.sha256) == 64 and item.size_bytes > 0 for item in files)


def test_download_catalog_rejects_path_traversal() -> None:
    raw = {
        "schema": "turbofit.downloads/v1",
        "files": {
            "bad": {
                "repo_id": "org/repo",
                "revision": "a" * 40,
                "path": "../secret",
                "destination": "model.gguf",
                "sha256": "b" * 64,
                "size_bytes": 1,
            }
        },
        "groups": {"bad": ["bad"]},
    }

    with pytest.raises(ValueError, match="unsafe path"):
        DownloadCatalog.from_mapping(raw)


def test_download_catalog_rejects_unknown_group_file() -> None:
    raw = {
        "schema": "turbofit.downloads/v1",
        "files": {
            "ok": {
                "repo_id": "org/repo",
                "revision": "a" * 40,
                "path": "model.gguf",
                "destination": "model.gguf",
                "sha256": "b" * 64,
                "size_bytes": 1,
            }
        },
        "groups": {"bad": ["missing"]},
    }

    with pytest.raises(ValueError, match="unknown files"):
        DownloadCatalog.from_mapping(raw)
