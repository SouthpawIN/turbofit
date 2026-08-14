"""Complete, evidence-labeled view of Turbofit's eight hardware tiers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .executor import production_recipe_sha256
from .hardware import HardwareFingerprint
from .intelligence import canonical_intelligence_recipe
from .recipes import RecipeBook
from .schema import MatrixRow

TIERS = (8, 16, 24, 48, 64, 96, 200, 300)


def _artifact_bytes(manifest: dict[str, Any], model_ids: tuple[str, ...]) -> int:
    selected = {}
    wanted = set(model_ids) - {"auto"}
    for artifact in manifest.get("artifacts") or []:
        if wanted.intersection(artifact.get("families") or []):
            selected[artifact["destination"]] = int(artifact["size_bytes"])
    return sum(selected.values())


def _intelligence_index(
    root: Path, valid_recipe_sha256: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    path = root / "references/intelligence-scores.json"
    if not path.is_file():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8")).get("records") or []
    except ValueError:
        return {}
    level_rank = {"screening": 0, "promotion": 1, "release": 2}
    result = {}
    for record in records:
        identifier = str(record.get("configuration_id", ""))
        level = str(record.get("benchmark_level", ""))
        if not identifier or level not in level_rank:
            continue
        if valid_recipe_sha256 is not None and record.get("production_recipe_sha256") != valid_recipe_sha256.get(identifier):
            continue
        current = result.get(identifier)
        if current is None or level_rank[level] > level_rank.get(str(current.get("benchmark_level")), -1):
            result[identifier] = record
    return result


def _native_tier(hardware: HardwareFingerprint) -> int:
    capacity = hardware.total_usable_memory_mb if hardware.shared_memory_pool else hardware.total_vram_mb
    gb = capacity / 1024
    return max(tier for tier in TIERS if tier <= max(8, gb))


def _exact_physical_topology(hardware: HardwareFingerprint, constraint: dict[str, Any]) -> bool:
    total_mb = (
        hardware.total_usable_memory_mb if hardware.shared_memory_pool
        else hardware.total_vram_mb
    )
    required_total = int(constraint["total_vram_gb"])
    actual_total = round(total_mb / 1024)
    is_open_ended = required_total == 300
    if actual_total < required_total or (not is_open_ended and actual_total != required_total):
        return False
    required_devices = int(constraint["min_devices"])
    if hardware.shared_memory_pool:
        return required_devices == 1
    if len(hardware.devices) < required_devices or (
        not is_open_ended and len(hardware.devices) != required_devices
    ):
        return False
    minimum_mb = int(constraint["per_device_min_gb"]) * 1024
    return sum(device.memory_total_mb >= minimum_mb for device in hardware.devices) >= required_devices


def build_tier_report(root: str | Path, hardware: HardwareFingerprint) -> dict[str, Any]:
    root = Path(root)
    configurations = json.loads((root / "references/configuration-matrix.json").read_text(encoding="utf-8"))
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    tournaments = json.loads((root / "references/hardware-tier-tournaments.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "references/artifact-manifest.json").read_text(encoding="utf-8"))
    evidence_index = json.loads((root / "runtime-profiles/class-evidence-index.json").read_text(encoding="utf-8"))
    campaign_path = root / "references/catalog-campaign-state.json"
    campaign_rows = (json.loads(campaign_path.read_text(encoding="utf-8")).get("rows") or {}) if campaign_path.is_file() else {}
    recipes = RecipeBook.load(root / "references/model-recipes.json")
    valid_intelligence_recipes = {}
    for configuration in configurations["rows"]:
        recipe = recipes.resolve_catalog_configuration(configuration)
        valid_intelligence_recipes[configuration["id"]] = canonical_intelligence_recipe(
            configuration, recipe,
        )[0]
    intelligence = _intelligence_index(root, valid_intelligence_recipes)
    configurations_by_id = {item["id"]: item for item in configurations["rows"]}
    models = {item["id"]: item for item in catalog["models"]}
    tournament_by_gb = {item["vram_gb"]: item for item in tournaments["tiers"]}
    tiers = []
    for capacity in TIERS:
        profile = yaml.safe_load((root / "runtime-profiles" / f"{capacity}gb.yaml").read_text(encoding="utf-8"))
        constraint = profile["hardware"]
        tournament = tournament_by_gb[capacity]
        winner = tournament["winner"]
        winner_metrics: dict[str, Any] = {}
        if winner:
            source = (evidence_index.get(winner["evidence"]) or {}).get("source")
            if source and (root / source).is_file():
                try:
                    winner_metrics = json.loads((root / source).read_text(encoding="utf-8")).get("winner_metrics") or {}
                except ValueError:
                    winner_metrics = {}
        candidate_rows = []
        for identifier in tournament["candidates"]:
            configuration = configurations_by_id[identifier]
            main = models[configuration["main"]]
            auxiliary = None if configuration["auxiliary"] == "auto" else models[configuration["auxiliary"]]
            score = intelligence.get(identifier)
            auxiliary_name = "auto" if auxiliary is None else auxiliary["name"]
            display_id = MatrixRow.make_id(main["name"], auxiliary_name, int(configuration["context"]))
            catalog_configuration = dict(configuration)
            catalog_configuration["id"] = display_id
            expected_recipe = production_recipe_sha256(
                recipes.resolve_catalog_configuration(catalog_configuration), catalog_configuration,
            )
            campaign_record = campaign_rows.get(display_id) or {}
            current_physical = (
                campaign_record.get("status") == "success"
                and campaign_record.get("recipe_sha256") == expected_recipe
            )
            physical = bool(
                winner and winner["configuration"] == identifier and current_physical
                and _exact_physical_topology(hardware, constraint)
            )
            artifact_bytes = _artifact_bytes(manifest, (main["id"], configuration["auxiliary"]))
            artifact_gib = artifact_bytes / (1024 ** 3)
            inferred_host_floor = math.ceil(artifact_gib + 8)
            inferred_host_recommended = math.ceil(artifact_gib * 1.25 + 16)
            measured_host_peak_mb = score.get("host_process_rss_peak_mb") if score else None
            measured_host_peak_gb = round(float(measured_host_peak_mb) / 1024, 2) if isinstance(measured_host_peak_mb, (int, float)) else None
            intelligence_value = score.get("intelligence_score") if score else None
            throughput_value = score.get("throughput_tps") if score else None
            if throughput_value is None and physical:
                throughput_value = winner_metrics.get("quality_decode_tps_median")
                if throughput_value is None:
                    pair_tps = winner_metrics.get("campaign_decode_tps") or {}
                    measured = [float(value) for value in pair_tps.values() if isinstance(value, (int, float)) and value > 0]
                    throughput_value = min(measured) if measured else None
            balanced_value = None
            if intelligence_value is not None and throughput_value is not None:
                intelligence_fraction = float(intelligence_value) / 100.0
                speed_fraction = min(float(throughput_value) / 50.0, 1.0)
                if intelligence_fraction > 0 and speed_fraction > 0:
                    balanced_value = 200.0 * intelligence_fraction * speed_fraction / (intelligence_fraction + speed_fraction)
            candidate_rows.append({
                "configuration_id": identifier,
                "main": {
                    "id": main["id"], "name": main["name"],
                    "quantization": main["quantization"],
                    "runtime_features": main["runtime_features"],
                },
                "auxiliary": {
                    "id": "auto", "name": "Route auxiliary tasks to main",
                    "quantization": "route-to-main", "runtime_features": [],
                } if auxiliary is None else {
                    "id": auxiliary["id"], "name": auxiliary["name"],
                    "quantization": auxiliary["quantization"],
                    "runtime_features": auxiliary["runtime_features"],
                },
                "context": int(configuration["context"]),
                "artifact_storage_bytes": artifact_bytes,
                "host_memory_requirement": {
                    "declared_minimum_gb": constraint.get("system_ram_gb"),
                    "inferred_artifact_load_floor_gb": inferred_host_floor,
                    "inferred_recommended_gb": inferred_host_recommended,
                    "physically_measured_peak_gb": measured_host_peak_gb,
                    "status": "measured-runtime-rss" if measured_host_peak_gb is not None else ("declared" if constraint.get("system_ram_gb") is not None else "inferred-until-rss-telemetry"),
                    "formula": "artifact GiB + 8 GiB floor; artifact GiB × 1.25 + 16 GiB recommended",
                },
                "accelerator_requirement": {
                    "aggregate_gb": constraint["total_vram_gb"],
                    "device_count": constraint["min_devices"],
                    "per_device_min_gb": constraint["per_device_min_gb"],
                    "topology": constraint["topology"],
                    "backend": constraint["accelerator"],
                },
                "offload_quantization_mode": {
                    "main_quantization": main["quantization"],
                    "auxiliary_quantization": "route-to-main" if auxiliary is None else auxiliary["quantization"],
                    "features": sorted(set(main["runtime_features"] + ([] if auxiliary is None else auxiliary["runtime_features"]))),
                },
                "fit": {
                    "inferred": True,
                    "physically_demonstrated": physical,
                    "evidence": winner["evidence"] if physical else None,
                    "hardware_fingerprint": winner["hardware_fingerprint"] if physical else None,
                },
                "intelligence_score": intelligence_value,
                "intelligence_level": score.get("benchmark_level") if score else None,
                "measured_tps": throughput_value,
                "balanced_score": balanced_value,
            })
        measured = next((item for item in candidate_rows if item["fit"]["physically_demonstrated"]), None)
        smart = sorted(
            (item for item in candidate_rows if item["intelligence_score"] is not None),
            key=lambda item: (item["intelligence_score"], item["measured_tps"] or 0), reverse=True,
        )
        fast = sorted(
            (item for item in candidate_rows if item["measured_tps"] is not None),
            key=lambda item: (item["measured_tps"], item["intelligence_score"] or 0), reverse=True,
        )
        balanced = sorted(
            (item for item in candidate_rows if item["balanced_score"] is not None),
            key=lambda item: (item["balanced_score"], item["intelligence_score"], item["measured_tps"]), reverse=True,
        )
        tiers.append({
            "id": f"hardware-{capacity}gb",
            "capacity_gb": capacity,
            "topology": constraint,
            "status": "physically-validated" if measured else "catalog-candidates-only",
            "recommendations": {
                "measured_winner": measured,
                "smartest": smart[0] if smart else None,
                "fastest": fast[0] if fast else None,
                "balanced": balanced[0] if balanced else None,
            },
            "candidates": candidate_rows,
        })
    return {
        "schema": "turbofit.hardware-tier-report/v1",
        "current_hardware": {
            "os": hardware.os,
            "architecture": hardware.architecture,
            "system_ram_mb": hardware.system_ram_mb,
            "accelerator_memory_mb": hardware.total_vram_mb,
            "usable_memory_mb": hardware.total_usable_memory_mb,
            "shared_memory_pool": hardware.shared_memory_pool,
            "device_memory_mb": [device.memory_total_mb for device in hardware.devices],
            "native_tier_gb": _native_tier(hardware),
        },
        "tiers": tiers,
    }
