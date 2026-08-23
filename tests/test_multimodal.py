from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
from turbofit_runtime.multimodal import (
    MultimodalCatalog,
    configure_multimodal,
    recommend_multimodal,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "references" / "multimodal-models.json"


def hardware(*, ram_mb: int, device_mb: int = 0, platform: str = "linux") -> HardwareFingerprint:
    devices = ()
    if device_mb:
        devices = (
            AcceleratorDevice(
                index=0,
                uuid="test-device",
                name="portable accelerator",
                vendor="amd",
                memory_total_mb=device_mb,
                backend="rocm",
                compute_capability="",
                bus_id="0000:01:00.0",
            ),
        )
    return HardwareFingerprint(os=platform, architecture="x86_64", system_ram_mb=ram_mb, devices=devices)


def test_multimodal_catalog_covers_requested_families_with_pinned_sources() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    ids = {item.id for item in catalog.models}

    assert {
        "minimax-h3",
        "acestep-1-5-2b",
        "acestep-1-5-4b",
        "hermes-edge-tts",
        "soprano-tts",
        "darwin-tts-1-7b",
        "hermes-local-stt",
        "parakeet-tdt-0-6b-v3",
        "nemotron-3-5-asr-0-6b",
    } <= ids
    for item in catalog.models:
        if item.source.startswith("https://huggingface.co/"):
            assert len(item.revision) == 40


def test_multimodal_recommendations_use_total_usable_memory() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    result = recommend_multimodal(hardware=hardware(ram_mb=310000, device_mb=24576), catalog=catalog)

    by_id = {item["id"]: item for rows in result["modalities"].values() for item in rows}
    assert result["total_usable_memory_mb"] == (310000 - 8192) + 24576
    assert by_id["minimax-h3"]["fit"] is True
    assert by_id["minimax-h3"]["accelerator_fit"] is True
    assert by_id["minimax-h3"]["action"] == "run-turbofit-local"
    assert by_id["acestep-1-5-4b"]["recommended_fit"] is True


def test_multimodal_recommendations_keep_builtin_options_on_cpu_only() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    result = recommend_multimodal(hardware=hardware(ram_mb=4096), catalog=catalog)

    assert result["total_usable_memory_mb"] == 3072
    assert any(item["id"] == "hermes-image" and item["fit"] for item in result["modalities"]["image"])
    assert any(item["id"] == "hermes-edge-tts" and item["fit"] for item in result["modalities"]["tts"])
    assert not next(item for item in result["modalities"]["video"] if item["id"] == "minimax-h3")["fit"]


def test_configure_multimodal_sets_supported_hermes_voice_providers_and_preserves_config() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    original = {"model": {"provider": "nous"}, "tts": {"edge": {"voice": "en-US-AvaNeural"}}}

    configured = configure_multimodal(
        original,
        selections={"tts": "soprano-tts", "stt": "hermes-local-stt", "music": "acestep-1-5-2b"},
        catalog=catalog,
    )

    assert configured is not original
    assert configured["model"] == original["model"]
    assert "provider" not in configured["tts"]
    assert configured["tts"]["edge"]["voice"] == "en-US-AvaNeural"
    assert configured["stt"]["provider"] == "local"
    assert configured["turbofit"]["multimodal"]["selected"]["music"] == "acestep-1-5-2b"


def test_configure_h3_writes_the_launch_recipe() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    configured = configure_multimodal({}, selections={"video": "minimax-h3"}, catalog=catalog)
    recipe = configured["turbofit"]["multimodal"]["h3"]
    assert recipe["schema"] == "turbofit.h3-launch/v1"
    assert recipe["verify"][0] == "scripts/verify-h3-live"


def test_candidate_soprano_does_not_claim_a_nonexistent_hermes_provider() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    configured = configure_multimodal(
        {"tts": {"provider": "edge"}},
        selections={"tts": "soprano-tts"},
        catalog=catalog,
    )

    assert configured["tts"]["provider"] == "edge"
    assert configured["turbofit"]["multimodal"]["selected"]["tts"] == "soprano-tts"


def test_configure_multimodal_rejects_wrong_modality() -> None:
    catalog = MultimodalCatalog.load(CATALOG)
    with pytest.raises(ValueError, match="does not support"):
        configure_multimodal({}, selections={"image": "acestep-1-5-2b"}, catalog=catalog)
