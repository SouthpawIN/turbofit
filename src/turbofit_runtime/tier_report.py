"""Complete, evidence-labeled view of Turbofit's eight hardware tiers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .executor import production_recipe_sha256
from .experts import available_ram_bytes
from .hardware import HardwareFingerprint
from .intelligence import canonical_intelligence_recipe
from .quant_policy import classify_entry, default_admits
from .recommend import vram_reporting_tolerance_mb
from .recipes import RecipeBook
from .schema import MatrixRow
from .tier_fit import (
    DEFAULT_RAM_GIB,
    RAM_GIB_ENV,
    RAM_RESERVE_BYTES,
    default_ram_gib,
    ram_budget_bytes,
)

TIERS = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384)


def _ram_budget(hardware: HardwareFingerprint, ram_bytes: int | None) -> tuple[int, str]:
    """The RAM budget this report places offload options against.

    Host first (``MemAvailable`` — what an engine can actually take), then the
    fingerprint's installed figure, then the documented default. An explicit
    ``ram_bytes`` wins so the committed artefact can state its assumption.
    """
    if ram_bytes is not None:
        if ram_bytes <= 0:
            raise ValueError("ram_bytes must be positive")
        return int(ram_bytes), "explicit"
    host = available_ram_bytes()
    if host:
        return host, "host-memavailable"
    installed = int(getattr(hardware, "system_ram_mb", 0) or 0)
    if installed > 0:
        return installed * 1024, "host-installed"
    return ram_budget_bytes(default_ram_gib()), "documented-default"


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


def _score_for_tier(
    scores: dict[str, dict[str, Any]], identifier: str, capacity_gb: int,
) -> dict[str, Any] | None:
    score = scores.get(identifier)
    if score is None or score.get("hardware_tier_gb") != capacity_gb:
        return None
    return score


def _durable_winner_is_current(
    root: Path,
    winner: dict[str, Any],
    evidence_index: dict[str, Any],
    expected_recipe: str,
) -> bool:
    entry = evidence_index.get(str(winner.get("evidence"))) or {}
    if not str(entry.get("status", "")).startswith("validated"):
        return False
    source = entry.get("source")
    if not source or not (root / source).is_file():
        return False
    try:
        payload = json.loads((root / source).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("recipe_sha256") == expected_recipe
        and payload.get("hardware_fingerprint") == winner.get("hardware_fingerprint")
        and payload.get("physical_evidence_sha256") == winner.get("evidence")
    )


def native_tier(hardware: HardwareFingerprint) -> int:
    """The published capacity band this host belongs to.

    Public because onboarding needs the same band the report uses: two answers
    to "which tier is this machine?" would put the first-run download on one
    tier and the recommendation card on another.
    """
    capacity = hardware.total_usable_memory_mb if hardware.shared_memory_pool else hardware.total_vram_mb
    gb = capacity / 1024
    return max(tier for tier in TIERS if tier <= max(8, gb))


#: Backwards-compatible alias: the report and older callers used the private name.
_native_tier = native_tier


def _exact_physical_topology(hardware: HardwareFingerprint, constraint: dict[str, Any]) -> bool:
    total_mb = (
        hardware.total_usable_memory_mb if hardware.shared_memory_pool
        else hardware.total_vram_mb
    )
    required_total = int(constraint["total_vram_gb"])
    actual_total = round(total_mb / 1024)
    is_open_ended = required_total == 384
    if actual_total < required_total or (not is_open_ended and actual_total != required_total):
        return False
    required_devices = int(constraint["min_devices"])
    if hardware.shared_memory_pool:
        return required_devices == 1
    if len(hardware.devices) < required_devices or (
        not is_open_ended and len(hardware.devices) != required_devices
    ):
        return False
    minimum_mb = int(constraint["per_device_min_gb"]) * 1024 - vram_reporting_tolerance_mb(
        int(constraint["per_device_min_gb"])
    )
    return sum(device.memory_total_mb >= minimum_mb for device in hardware.devices) >= required_devices


def _offload_row(
    entry: dict[str, Any],
    *,
    configuration: dict[str, Any],
    main: dict[str, Any],
    auxiliary: dict[str, Any] | None,
    constraint: dict[str, Any],
    artifact_bytes: int,
) -> dict[str, Any]:
    """One row for a model that only fits this tier with experts in RAM."""
    artifact_gib = artifact_bytes / (1024 ** 3)
    return {
        "configuration_id": entry["configuration_id"],
        "placement": "offload",
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
        "accelerator_requirement": {
            "aggregate_gb": constraint["total_vram_gb"],
            "device_count": constraint["min_devices"],
            "per_device_min_gb": constraint["per_device_min_gb"],
            "topology": constraint["topology"],
            "backend": constraint["accelerator"],
        },
        "host_memory_requirement": {
            "declared_minimum_gb": constraint.get("system_ram_gb"),
            "inferred_artifact_load_floor_gb": math.ceil(artifact_gib + 8),
            "inferred_recommended_gb": math.ceil(artifact_gib * 1.25 + 16),
            "physically_measured_peak_gb": None,
            "status": "declared" if constraint.get("system_ram_gb") is not None else "inferred-until-rss-telemetry",
            "formula": "artifact GiB + 8 GiB floor; artifact GiB × 1.25 + 16 GiB recommended",
        },
        "expert_offload": {
            "n_cpu_moe": int(entry["n_cpu_moe"]),
            "expert_layers": int(entry["expert_layers"]),
            "ram_gib": float(entry["ram_gib"]),
            "vram_gib": float(entry["vram_gib"]),
            "total_gib": float(entry["total_gib"]),
            "evidence": entry["evidence"],
            "reason": entry["reason"],
            "engine_flag": "--n-cpu-moe",
        },
        "fit": {
            "inferred": True,
            "physically_demonstrated": False,
            "placement": "offload",
            "evidence": None,
            "hardware_fingerprint": None,
        },
        "intelligence_score": None,
        "intelligence_level": None,
        "measured_tps": None,
        "balanced_score": None,
    }


def build_tier_report(
    root: str | Path,
    hardware: HardwareFingerprint,
    ram_bytes: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    ram_bytes, ram_source = _ram_budget(hardware, ram_bytes)
    configurations = json.loads((root / "references/configuration-matrix.json").read_text(encoding="utf-8"))
    catalog = json.loads((root / "references/model-catalog.json").read_text(encoding="utf-8"))
    tournaments = json.loads((root / "references/hardware-tier-tournaments.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "references/artifact-manifest.json").read_text(encoding="utf-8"))
    evidence_index = json.loads((root / "runtime-profiles/class-evidence-index.json").read_text(encoding="utf-8"))
    campaign_path = root / "references/catalog-campaign-state.json"
    campaign_rows = (json.loads(campaign_path.read_text(encoding="utf-8")).get("rows") or {}) if campaign_path.is_file() else {}
    recipes = RecipeBook.load(root / "references/model-recipes.json", hardware=hardware)
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
            score = _score_for_tier(intelligence, identifier, capacity)
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
            if winner and winner["configuration"] == identifier:
                current_physical = current_physical or _durable_winner_is_current(
                    root, winner, evidence_index, expected_recipe,
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
            quant = classify_entry(main)
            _admitted, quant_reason = default_admits(main, fits_fully_in_vram=True)
            candidate_rows.append({
                "configuration_id": identifier,
                "quant_class": quant.quant_class,
                "quant_reason": None if quant.is_allowed else quant_reason,
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
                    "placement": "gpu",
                    "evidence": winner["evidence"] if physical else None,
                    "hardware_fingerprint": winner["hardware_fingerprint"] if physical else None,
                },
                "intelligence_score": intelligence_value,
                "intelligence_level": score.get("benchmark_level") if score else None,
                "measured_tps": throughput_value,
                "balanced_score": balanced_value,
            })
        measured = next((item for item in candidate_rows if item["fit"]["physically_demonstrated"]), None)
        offload_rows = []
        for entry in tournament.get("offload_candidates") or []:
            configuration = configurations_by_id[entry["configuration_id"]]
            main = models[configuration["main"]]
            auxiliary = None if configuration["auxiliary"] == "auto" else models[configuration["auxiliary"]]
            offload_rows.append(_offload_row(
                entry,
                configuration=configuration,
                main=main,
                auxiliary=auxiliary,
                constraint=constraint,
                artifact_bytes=_artifact_bytes(
                    manifest, (main["id"], configuration["auxiliary"]),
                ),
            ))
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
            "offload_candidates": offload_rows,
        })
    return {
        "schema": "turbofit.hardware-tier-report/v1",
        "ram_budget": {
            "gib": round(ram_bytes / 1024 ** 3, 2),
            "source": ram_source,
            "reserve_gib": round(RAM_RESERVE_BYTES / 1024 ** 3, 2),
            "default_gib": DEFAULT_RAM_GIB,
            "override": f"--ram-gb on this script, or {RAM_GIB_ENV} for every caller",
        },
        "current_hardware": {
            "os": hardware.os,
            "architecture": hardware.architecture,
            "system_ram_mb": hardware.system_ram_mb,
            "host_usable_memory_mb": hardware.host_usable_memory_mb,
            "accelerator_memory_mb": hardware.total_vram_mb,
            "usable_memory_mb": hardware.total_usable_memory_mb,
            "memory_pool_kind": hardware.memory_pool_kind,
            "shared_memory_pool": hardware.shared_memory_pool,
            "device_memory_mb": [device.memory_total_mb for device in hardware.devices],
            "native_tier_gb": _native_tier(hardware),
        },
        "tiers": tiers,
    }
