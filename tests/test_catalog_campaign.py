from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.catalog_campaign import (
    build_campaign_matrix,
    CatalogExecutorAdapter,
    CatalogRegistryAdapter,
)
from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.schema import load_matrix


ROOT = Path(__file__).parents[1]


def test_catalog_campaign_materializes_all_canonical_configurations(tmp_path: Path) -> None:
    catalog = ModelCatalog.load(ROOT / "references" / "model-catalog.json")
    configurations = json.loads((ROOT / "references" / "configuration-matrix.json").read_text())
    output = tmp_path / "campaign.json"

    build_campaign_matrix(configurations, catalog, output)
    matrix = load_matrix(output)

    assert len(matrix.rows) == 1620
    assert len({row.id for row in matrix.rows}) == 1620
    assert {row.context for row in matrix.rows} == set(catalog.contexts)
    assert any("DeepSeek V4 Flash" in row.main for row in matrix.rows)
    assert any("Qwen 3.8 27B Q8" in row.main for row in matrix.rows)
    assert any("Qwen 3.8 27B BF16" in row.main for row in matrix.rows)


def test_catalog_executor_uses_original_configuration_payload() -> None:
    calls = []

    class Executor:
        def execute_catalog(self, payload):
            calls.append(payload)
            return "result"

        def catalog_recipe_sha256(self, payload):
            return "sha256:recipe"

    payload = {"id": "row-a", "main_model": "model-a", "auxiliary": "auto", "context": 65536}
    adapter = CatalogExecutorAdapter(Executor(), {"row-a": payload})

    class Row:
        id = "row-a"

    assert adapter.execute(Row()) == "result"
    assert adapter.recipe_sha256(Row()) == "sha256:recipe"
    assert calls == [payload]


def test_catalog_registry_resolves_the_variant_recipe() -> None:
    calls = []

    class Recipes:
        def resolve_catalog_configuration(self, payload):
            return {"recipe_for": payload["main"]}

    class Registry:
        def register(self, item, result, evidence, *, recipe=None):
            calls.append((item.id, result, evidence, recipe))

    payload = {"id": "row-a", "main": "variant-a", "auxiliary": "auto", "context": 65536}
    adapter = CatalogRegistryAdapter(Registry(), Recipes(), {"row-a": payload})

    class Row:
        id = "row-a"

    adapter.register(Row(), "result", Path("evidence.json"))
    assert calls == [("row-a", "result", Path("evidence.json"), {"recipe_for": "variant-a"})]


def test_catalog_registry_does_not_promote_archived_main() -> None:
    calls = []

    class Recipes:
        def resolve_catalog_configuration(self, payload):
            calls.append(("resolve", payload))

    class Registry:
        def register(self, *args, **kwargs):
            calls.append(("register", args, kwargs))

    payload = {"id": "row-a", "main": "archive-a", "auxiliary": "auto", "context": 65536}
    adapter = CatalogRegistryAdapter(
        Registry(), Recipes(), {"row-a": payload}, excluded_main_ids={"archive-a"}
    )

    class Row:
        id = "row-a"

    adapter.register(Row(), "result", Path("evidence.json"))
    assert calls == []
