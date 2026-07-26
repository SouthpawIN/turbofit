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
    assert len(catalog.main_models) == 12
    assert catalog.contexts == CONTEXTS
    assert catalog.auxiliary_options == (
        "carwin-nano", "ternary-bonsai-27b", "bonsai-27b", "auto"
    )
    assert all(item.source.startswith("https://huggingface.co/") for item in catalog.models)


def test_complete_matrix_is_generated_not_hand_maintained() -> None:
    catalog = ModelCatalog.load(CATALOG_PATH)
    matrix = json.loads(MATRIX_PATH.read_text())
    validate_configuration_matrix(matrix, catalog)
    assert len(matrix["rows"]) == 12 * 4 * 4
    assert len({item["id"] for item in matrix["rows"]}) == 192
    assert {item["context"] for item in matrix["rows"]} == set(CONTEXTS)


def test_matrix_rejects_missing_configuration() -> None:
    catalog = ModelCatalog.load(CATALOG_PATH)
    matrix = build_configuration_matrix(catalog)
    matrix["rows"].pop()
    with pytest.raises(ValueError, match="stale or incomplete"):
        validate_configuration_matrix(matrix, catalog)
