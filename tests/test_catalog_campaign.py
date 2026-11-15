from __future__ import annotations

import json
from pathlib import Path

from turbofit_runtime.catalog_campaign import (
    build_campaign_matrix,
    quant_policy_rows,
    CatalogExecutorAdapter,
    CatalogRegistryAdapter,
)
from turbofit_runtime.model_catalog import ModelCatalog
from turbofit_runtime.schema import load_matrix


ROOT = Path(__file__).parents[1]


def test_catalog_campaign_materializes_canonical_rows_under_the_quant_ceiling(
    tmp_path: Path,
) -> None:
    catalog = ModelCatalog.load(ROOT / "references" / "model-catalog.json")
    configurations = json.loads((ROOT / "references" / "configuration-matrix.json").read_text())
    output = tmp_path / "campaign.json"
    excluded, exceptions = quant_policy_rows(configurations, catalog)

    build_campaign_matrix(configurations, catalog, output)
    matrix = load_matrix(output)

    # 290 canonical rows minus the rows whose main is above the Q4 ceiling
    # without a recorded justification; the catalog rows themselves stay put.
    assert len(configurations["rows"]) == 290
    assert len(matrix.rows) == 290 - len(excluded) == 272
    assert len({row.id for row in matrix.rows}) == len(matrix.rows)
    assert {row.context for row in matrix.rows} == set(catalog.contexts)
    assert any("GSQ-RCO" in row.main for row in matrix.rows)
    assert any("Qwen 3.8 27B" in row.main for row in matrix.rows)
    # Q5/Q6 rows are not campaign rows by default and are reported with a reason.
    assert excluded, "the policy must exclude at least the Q5 builds"
    assert all(record["reason"] for record in excluded.values())
    assert not any("Q5" in row.main or "Q6" in row.main for row in matrix.rows)
    # Q8/full-precision rows remain measurable, with the exception reason recorded.
    assert any(record["reason"].startswith("accuracy-tier") for record in exceptions.values())


def test_catalog_campaign_can_still_materialize_every_row_explicitly(tmp_path: Path) -> None:
    catalog = ModelCatalog.load(ROOT / "references" / "model-catalog.json")
    configurations = json.loads((ROOT / "references" / "configuration-matrix.json").read_text())
    output = tmp_path / "campaign-all.json"

    build_campaign_matrix(configurations, catalog, output, enforce_quant_policy=False)

    assert len(load_matrix(output).rows) == 290


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
