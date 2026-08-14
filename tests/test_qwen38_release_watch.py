from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).parents[1]


def module() -> dict:
    return runpy.run_path(str(ROOT / "scripts/watch-qwen38-release"), run_name="qwen_watch_test")


def test_release_watcher_rejects_third_party_pre_release_artifacts() -> None:
    result = module()["inspect"]([
        {
            "id": "someone/Qwen3.8-27B-FP8",
            "author": "someone",
            "sha": "a" * 40,
            "private": False,
            "gated": False,
        }
    ])

    assert result["status"] == "waiting-for-official-release"
    assert result["official_models"] == []
    assert result["third_party_models_accepted"] is False


def test_release_watcher_pins_official_revision_and_lists_gguf_files() -> None:
    result = module()["inspect"]([
        {
            "id": "Qwen/Qwen3.8-27B",
            "author": "Qwen",
            "sha": "b" * 40,
            "private": False,
            "gated": False,
            "siblings": [
                {"rfilename": "model.safetensors"},
                {"rfilename": "official-Q8_0.gguf"},
            ],
        }
    ])

    assert result["status"] == "official-release-detected"
    assert result["official_models"][0]["revision"] == "b" * 40
    assert result["official_models"][0]["gguf_files"] == ["official-Q8_0.gguf"]
    should_pause = module()["should_pause_campaign"]
    assert should_pause(result, "") is True
    assert should_pause(result, "onboarded-awaiting-physical-evidence") is False
