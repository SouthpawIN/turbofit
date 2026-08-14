from __future__ import annotations

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "generate-h3-promo-clips"


def load_script():
    loader = SourceFileLoader("generate_h3_promo_clips", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest() -> dict:
    return {
        "schema": "turbofit.h3-promo-prompts/v1",
        "model": "MiniMaxAI/MiniMax-H3",
        "revision": "pinned",
        "workflow": "t2va",
        "width": 640,
        "height": 384,
        "num_frames": 192,
        "num_inference_steps": 20,
        "clips": [{"id": "anime", "label": "ANIME", "seed": 1, "prompt": "anime"}],
    }


def test_h3_prompt_manifest_accepts_pinned_local_generation_plan(tmp_path: Path) -> None:
    module = load_script()
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(manifest()), encoding="utf-8")

    loaded = module.load_manifest(path)

    assert loaded["model"] == "MiniMaxAI/MiniMax-H3"
    assert loaded["clips"][0]["id"] == "anime"


def test_h3_prompt_manifest_rejects_non_aligned_canvas(tmp_path: Path) -> None:
    module = load_script()
    payload = manifest()
    payload["width"] = 641
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="multiples of 32"):
        module.load_manifest(path)
