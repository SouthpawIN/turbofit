from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.model_catalog import (
    CONTEXTS,
    ModelCatalog,
    build_configuration_matrix,
    validate_configuration_matrix,
)

ROOT = Path(__file__).parents[1]
CATALOG_PATH = ROOT / "references" / "model-catalog.json"
MATRIX_PATH = ROOT / "references" / "configuration-matrix.json"


def test_catalog_has_every_requested_main_and_auxiliary_variant() -> None:
    catalog = ModelCatalog.load(CATALOG_PATH)
    assert len(catalog.main_models) == 42
    unleashed = next(item for item in catalog.main_models if item.id == "qwen3-8-27b-unleashed-ud-q3-k-xl")
    assert unleashed.source == "https://huggingface.co/outsourc-e/Qwen3.8-27B-Unleashed-GGUF"
    assert unleashed.upstream_source == "https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored"
    assert unleashed.quantization == "UD-Q3_K_XL"
    assert catalog.contexts == CONTEXTS
    assert catalog.auxiliary_options == (
        "ornith-1-5-35a3b",
        "carwin-nano",
        "auto",
    )
    assert all(item.source.startswith("https://huggingface.co/") for item in catalog.models)


def test_complete_matrix_is_generated_not_hand_maintained() -> None:
    catalog = ModelCatalog.load(CATALOG_PATH)
    matrix = json.loads(MATRIX_PATH.read_text())
    validate_configuration_matrix(matrix, catalog)
    expected = len(catalog.main_models) * len(catalog.auxiliary_options) * len(CONTEXTS)
    assert expected == 504
    assert len(matrix["rows"]) == expected
    assert len({item["id"] for item in matrix["rows"]}) == expected
    assert {item["context"] for item in matrix["rows"]} == set(CONTEXTS)


def test_matrix_rejects_missing_configuration() -> None:
    catalog = ModelCatalog.load(CATALOG_PATH)
    matrix = build_configuration_matrix(catalog)
    matrix["rows"].pop()
    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_configuration_matrix(matrix, catalog)
