from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.acquisition import (
    AcquisitionCatalog,
    AcquisitionError,
    ModelAcquirer,
)
from turbofit_runtime.routes import load_runtime_resolutions


ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, models: list[dict] | None = None) -> None:
        self.models = list(models or [])
        self.pulls: list[dict] = []
        self.manifests: list[tuple[str, dict]] = []

    def list_models(self) -> list[dict]:
        return list(self.models)

    def pull_hf(self, **payload):
        self.pulls.append(payload)
        return {
            "status": "complete",
            "sha256": payload["expected_sha256"],
            "bytes_written": 100,
        }

    def put_manifest(self, tag: str, manifest: dict, *, etag=None):
        assert etag is None
        self.manifests.append((tag, manifest))
        self.models.append(
            {
                "name": tag,
                "digest": f"sha256:{manifest['gguf_blob_sha256']}",
                "details": {
                    "context_length": manifest["context_size"],
                    "expected_vram_bytes": manifest["expected_vram_bytes"],
                },
            }
        )
        return type("Response", (), {"payload": {"status": "ok"}, "etag": '"1"'})()

    def show_model(self, name: str) -> dict:
        model = next(item for item in self.models if item["name"] == name)
        return {
            "name": name,
            "digest": model["digest"],
            "context_length": model["details"]["context_length"],
            "expected_vram_bytes": model["details"]["expected_vram_bytes"],
        }


def catalog_mapping() -> dict:
    return {
        "schema": "turbofit.acquisitions/v1",
        "artifacts": {
            "shared": {
                "repo_id": "org/repo",
                "filename": "model.gguf",
                "revision": "a" * 40,
                "sha256": "b" * 64,
                "size_bytes": 100,
            }
        },
        "tags": {
            "main-128k": {
                "artifact": "shared",
                "context_size": 131072,
                "expected_vram_bytes": 1024,
                "display_name": "Main 128K",
                "description": "Measured main",
                "llama_server_flags": {"ctx_size": 131072, "main_gpu": 0},
                "prompt_template": {"system_default": "", "stop_tokens": []},
            },
            "main-64k": {
                "artifact": "shared",
                "context_size": 65536,
                "expected_vram_bytes": 768,
                "display_name": "Main 64K",
                "description": "Measured main",
                "llama_server_flags": {"ctx_size": 65536, "main_gpu": 0},
                "prompt_template": {"system_default": "", "stop_tokens": []},
            },
        },
    }


def test_acquirer_downloads_missing_blob_once_and_registers_every_requested_tag() -> None:
    catalog = AcquisitionCatalog.from_mapping(catalog_mapping())
    client = Client()

    ModelAcquirer(catalog, client).ensure_tags(("main-128k", "main-64k"))

    assert len(client.pulls) == 1
    assert client.pulls[0] == {
        "repo_id": "org/repo",
        "filename": "model.gguf",
        "revision": "a" * 40,
        "expected_sha256": "b" * 64,
    }
    assert [tag for tag, _ in client.manifests] == ["main-128k", "main-64k"]
    assert all(item[1]["gguf_blob_sha256"] == "b" * 64 for item in client.manifests)


def test_acquirer_reuses_existing_blob_without_downloading() -> None:
    catalog = AcquisitionCatalog.from_mapping(catalog_mapping())
    client = Client([
        {
            "name": "other-tag",
            "digest": "sha256:" + "b" * 64,
            "details": {"context_length": 1, "expected_vram_bytes": 1},
        }
    ])

    ModelAcquirer(catalog, client).ensure_tags(("main-128k",))

    assert client.pulls == []
    assert [tag for tag, _ in client.manifests] == ["main-128k"]


def test_acquirer_rejects_existing_tag_with_wrong_blob() -> None:
    catalog = AcquisitionCatalog.from_mapping(catalog_mapping())
    client = Client([
        {
            "name": "main-128k",
            "digest": "sha256:" + "c" * 64,
            "details": {"context_length": 131072, "expected_vram_bytes": 1024},
        }
    ])

    with pytest.raises(AcquisitionError, match="will not overwrite"):
        ModelAcquirer(catalog, client).ensure_tags(("main-128k",))

    assert client.pulls == [] and client.manifests == []


def test_every_selectable_local_runtime_tag_has_a_download_recipe() -> None:
    catalog = AcquisitionCatalog.load(ROOT / "runtime-profiles" / "acquisitions.json")
    resolutions = load_runtime_resolutions(
        ROOT / "runtime-profiles" / "runtime-resolutions.json"
    )
    required_tags = {
        str(role["model_tag"])
        for profile_id, rungs in resolutions.items()
        if profile_id.startswith("hardware-")
        for roles in rungs.values()
        for role in roles.values()
    }

    assert required_tags == set(catalog.tags)


def test_catalog_rejects_unpinned_huggingface_revision() -> None:
    value = catalog_mapping()
    value["artifacts"]["shared"]["revision"] = "main"

    with pytest.raises(ValueError, match="revision"):
        AcquisitionCatalog.from_mapping(value)
