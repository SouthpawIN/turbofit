from turbofit_runtime.discovery_queue import build_discovery_queue


def test_discovery_queue_connects_tracked_huggingface_models_to_campaign() -> None:
    candidates = {
        "candidates": [
            {
                "id": "huggingface:org/model-a",
                "kind": "huggingface",
                "name": "org/model-a",
                "url": "https://huggingface.co/org/model-a",
                "provenance": {"revision": "a" * 40},
                "metadata": {"tags": ["gguf"]},
                "status": "candidate",
            },
            {
                "id": "huggingface:new/model-b",
                "kind": "huggingface",
                "name": "new/model-b",
                "url": "https://huggingface.co/new/model-b",
                "provenance": {"revision": "b" * 40},
                "metadata": {"tags": ["safetensors"]},
                "status": "candidate",
            },
        ]
    }
    catalog = {
        "models": [{
            "id": "model-a-q4",
            "source": "https://huggingface.co/org/model-a",
            "revision": "a" * 40,
        }]
    }

    queue = build_discovery_queue(candidates, catalog)

    tracked = next(item for item in queue["entries"] if item["candidate_id"].endswith("model-a"))
    new = next(item for item in queue["entries"] if item["candidate_id"].endswith("model-b"))
    assert tracked["status"] == "ready-for-catalog-campaign"
    assert tracked["catalog_model_ids"] == ["model-a-q4"]
    assert new["status"] == "needs-onboarding-spec"


def test_discovery_queue_marks_changed_revision_for_repin_not_benchmark() -> None:
    candidates = {"candidates": [{
        "id": "huggingface:org/model-a", "kind": "huggingface", "name": "org/model-a",
        "url": "https://huggingface.co/org/model-a",
        "provenance": {"revision": "b" * 40}, "metadata": {}, "status": "candidate",
    }]}
    catalog = {"models": [{
        "id": "model-a", "source": "https://huggingface.co/org/model-a", "revision": "a" * 40,
    }]}

    queue = build_discovery_queue(candidates, catalog)

    assert queue["entries"][0]["status"] == "needs-repin"
