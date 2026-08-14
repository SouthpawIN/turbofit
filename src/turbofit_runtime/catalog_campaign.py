"""Adapter that executes the canonical catalog through CampaignRunner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .model_catalog import ModelCatalog, validate_configuration_matrix
from .schema import MatrixRow


class CatalogExecutorAdapter:
    def __init__(self, executor: Any, configurations: Mapping[str, dict[str, Any]]) -> None:
        self.executor = executor
        self.configurations = dict(configurations)

    def execute(self, item: Any):
        return self.executor.execute_catalog(self.configurations[item.id])

    def recipe_sha256(self, item: Any) -> str:
        return self.executor.catalog_recipe_sha256(self.configurations[item.id])

    def evidence_is_current(self, record: dict) -> bool:
        return self.executor.evidence_is_current(record)

    def current_physical_fingerprint(self) -> str:
        return self.executor.current_physical_fingerprint()

    def prepare(self, item: Any) -> None:
        self.executor.prepare(item)

    def finish(self, item: Any) -> None:
        self.executor.finish(item)

    def record_campaign_failure(
        self, item: Any, error: str, raw_result_path: str | None,
        before: Any, after: Any,
    ) -> str:
        return self.executor.record_campaign_failure(
            item, error, raw_result_path, before, after,
        )


class CatalogRegistryAdapter:
    def __init__(
        self,
        registry: Any,
        recipes: Any,
        configurations: Mapping[str, dict[str, Any]],
        *,
        excluded_main_ids: set[str] | None = None,
    ) -> None:
        self.registry = registry
        self.recipes = recipes
        self.configurations = dict(configurations)
        self.excluded_main_ids = frozenset(excluded_main_ids or ())

    def register(self, item: Any, result: Any, evidence_path: Path) -> None:
        configuration = self.configurations[item.id]
        if configuration["main"] in self.excluded_main_ids:
            return
        recipe = self.recipes.resolve_catalog_configuration(configuration)
        self.registry.register(item, result, evidence_path, recipe=recipe)


def build_campaign_matrix(
    configurations: Mapping[str, Any], catalog: ModelCatalog, output: Path
) -> dict[str, Any]:
    return build_selected_campaign_matrix(configurations, catalog, output)


def build_selected_campaign_matrix(
    configurations: Mapping[str, Any],
    catalog: ModelCatalog,
    output: Path,
    configuration_ids: set[str] | None = None,
) -> dict[str, Any]:
    validate_configuration_matrix(configurations, catalog)
    by_id = {item.id: item for item in catalog.models}
    rows = []
    for item in configurations["rows"]:
        if configuration_ids is not None and item["id"] not in configuration_ids:
            continue
        main = by_id[item["main"]]
        aux_id = item["auxiliary"]
        aux_name = "auto" if aux_id == "auto" else by_id[aux_id].name
        priorities = list(main.runtime_features) or ["standard"]
        row_id = MatrixRow.make_id(main.name, aux_name, int(item["context"]))
        rows.append({
            "id": row_id,
            "main": main.name,
            "aux": aux_name,
            "context": item["context"],
            "status": "pending",
            "method_priority": priorities,
        })
    payload = {"schema_version": 1, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_configuration_index(
    configurations: Mapping[str, Any], catalog: ModelCatalog
) -> dict[str, dict[str, Any]]:
    validate_configuration_matrix(configurations, catalog)
    by_id = {item.id: item for item in catalog.models}
    result = {}
    for item in configurations["rows"]:
        main = by_id[item["main"]]
        aux = item["auxiliary"]
        aux_name = "auto" if aux == "auto" else by_id[aux].name
        row_id = MatrixRow.make_id(main.name, aux_name, int(item["context"]))
        payload = dict(item)
        payload["id"] = row_id
        result[row_id] = payload
    return result
