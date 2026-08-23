"""Measured intelligence scores for exact production model configurations.

No catalog tier, download count, parameter count, or estimated capability is an
intelligence score. A score exists only when every required benchmark has a
hash-bound raw result for the exact main/aux/quant/context recipe.
"""
from __future__ import annotations

import math
import re
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_BENCHMARKS = ("deep-swe", "agentic-pair")
INTELLIGENCE_RECIPE_PROTOCOL = "turbofit.intelligence-recipe/v2"
DEEPSWE_RUNNER_PROTOCOL = "turbofit.deepswe-pier/v7"
AGENTIC_PAIR_PROTOCOL = "turbofit.agentic-production-pair/v1"


def canonical_intelligence_recipe(
    configuration: dict[str, Any], recipe: Any,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "intelligence_recipe_protocol": INTELLIGENCE_RECIPE_PROTOCOL,
        "benchmark_protocols": {
            "deep-swe": DEEPSWE_RUNNER_PROTOCOL,
            "agentic-pair": AGENTIC_PAIR_PROTOCOL,
        },
        "configuration": configuration,
        "profile_name": recipe.profile_name,
        "main_alias": recipe.main_alias,
        "aux_alias": recipe.aux_alias,
        "aux_mode": recipe.aux_mode,
        "components": [
            {
                "role": component.role,
                "family": component.family,
                "alias": component.alias,
                "method": component.method,
                "gpu": component.gpu,
                "port": component.port,
                "command": list(component.command),
                "model_path": component.model_path,
                "projector_path": component.projector_path,
            }
            for component in recipe.components
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest(), payload


@dataclass(frozen=True)
class BenchmarkMeasurement:
    name: str
    revision: str
    score: float
    tasks_total: int
    tasks_passed: int
    raw_result: str
    raw_result_sha256: str


@dataclass(frozen=True)
class ConfigurationIntelligence:
    configuration_id: str
    hardware_tier_gb: int
    main: str
    auxiliary: str
    context: int
    quantizations: tuple[str, ...]
    production_recipe_sha256: str
    throughput_tps: float
    measurements: tuple[BenchmarkMeasurement, ...]

    @property
    def coverage(self) -> str:
        names = {item.name for item in self.measurements}
        return "complete" if set(REQUIRED_BENCHMARKS) <= names else "partial"

    @property
    def intelligence(self) -> float:
        return intelligence_score(self)

    @property
    def balanced(self) -> float:
        intelligence_fraction = self.intelligence / 100.0
        speed_fraction = min(self.throughput_tps / 50.0, 1.0)
        if intelligence_fraction <= 0 or speed_fraction <= 0:
            return 0.0
        return round(200.0 * intelligence_fraction * speed_fraction / (intelligence_fraction + speed_fraction), 6)


def validate_measurement(item: BenchmarkMeasurement) -> None:
    if item.name not in REQUIRED_BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {item.name}")
    if not item.revision.strip():
        raise ValueError("benchmark revision is required")
    if isinstance(item.score, bool) or not isinstance(item.score, (int, float)) or not math.isfinite(item.score):
        raise ValueError("benchmark score must be finite")
    if not 0 <= float(item.score) <= 1:
        raise ValueError("benchmark score must be between 0 and 1")
    if isinstance(item.tasks_total, bool) or not isinstance(item.tasks_total, int) or item.tasks_total <= 0:
        raise ValueError("tasks_total must be a positive integer")
    if isinstance(item.tasks_passed, bool) or not isinstance(item.tasks_passed, int):
        raise ValueError("tasks_passed must be an integer")
    if not 0 <= item.tasks_passed <= item.tasks_total:
        raise ValueError("tasks_passed must be within tasks_total")
    observed = item.tasks_passed / item.tasks_total
    if not math.isclose(observed, float(item.score), rel_tol=0, abs_tol=1e-9):
        raise ValueError("benchmark score does not match passed/total evidence")
    if not item.raw_result.strip():
        raise ValueError("raw benchmark result path is required")
    if _HASH.fullmatch(item.raw_result_sha256) is None:
        raise ValueError("raw benchmark result must be SHA-256 bound")


def validate_configuration(item: ConfigurationIntelligence) -> None:
    if not item.configuration_id.strip() or not item.main.strip() or not item.auxiliary.strip():
        raise ValueError("configuration identity, main, and auxiliary are required")
    if isinstance(item.hardware_tier_gb, bool) or item.hardware_tier_gb not in (8, 16, 24, 48, 64, 96, 200, 300):
        raise ValueError("hardware tier must be one of 8/16/24/48/64/96/200/300 GB")
    if item.context not in (65_536, 131_072, 262_144, 1_048_576):
        raise ValueError("unsupported production context")
    if not item.quantizations or any(not value.strip() for value in item.quantizations):
        raise ValueError("exact production quantization labels are required")
    if _HASH.fullmatch(item.production_recipe_sha256) is None:
        raise ValueError("production recipe must be SHA-256 bound")
    if isinstance(item.throughput_tps, bool) or not isinstance(item.throughput_tps, (int, float)):
        raise ValueError("throughput must be numeric")
    if not math.isfinite(item.throughput_tps) or item.throughput_tps <= 0:
        raise ValueError("throughput must be positive and finite")
    names = [measurement.name for measurement in item.measurements]
    if len(names) != len(set(names)):
        raise ValueError("duplicate benchmark measurement")
    for measurement in item.measurements:
        validate_measurement(measurement)


def intelligence_score(item: ConfigurationIntelligence) -> float:
    validate_configuration(item)
    measurements = {measurement.name: measurement for measurement in item.measurements}
    missing = [name for name in REQUIRED_BENCHMARKS if name not in measurements]
    if missing:
        raise ValueError("missing required benchmark measurements: " + ", ".join(missing))
    # Equal-weight arithmetic mean preserves each real suite's contribution.
    # A genuine 0/N result remains visible in its raw measurement, but it no
    # longer erases non-zero measured capability from every other suite. Tier
    # promotion can still enforce per-suite floors separately.
    values = [float(measurements[name].score) for name in REQUIRED_BENCHMARKS]
    return round(100.0 * sum(values) / len(values), 6)


def rank_configurations(
    configurations: Sequence[ConfigurationIntelligence] | Iterable[ConfigurationIntelligence],
) -> dict[str, list[ConfigurationIntelligence]]:
    items = list(configurations)
    for item in items:
        intelligence_score(item)
    stable = lambda item: item.configuration_id
    return {
        "intelligence": sorted(items, key=lambda item: (item.intelligence, item.throughput_tps, stable(item)), reverse=True),
        "balanced": sorted(items, key=lambda item: (item.balanced, item.intelligence, item.throughput_tps, stable(item)), reverse=True),
        "speed": sorted(items, key=lambda item: (item.throughput_tps, item.intelligence, stable(item)), reverse=True),
    }


def refresh_score_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Recompute derived composite fields from hash-bound suite measurements."""
    item = from_mapping(value)
    return {
        **dict(value),
        "intelligence_score": item.intelligence,
        "balanced_score": item.balanced,
    }


def from_mapping(value: Mapping[str, object]) -> ConfigurationIntelligence:
    measurements = tuple(BenchmarkMeasurement(**dict(item)) for item in value.get("measurements", ()) if isinstance(item, Mapping))
    result = ConfigurationIntelligence(
        configuration_id=str(value.get("configuration_id", "")),
        hardware_tier_gb=value.get("hardware_tier_gb"),  # type: ignore[arg-type]
        main=str(value.get("main", "")),
        auxiliary=str(value.get("auxiliary", "")),
        context=value.get("context"),  # type: ignore[arg-type]
        quantizations=tuple(str(item) for item in value.get("quantizations", ())),  # type: ignore[union-attr]
        production_recipe_sha256=str(value.get("production_recipe_sha256", "")),
        throughput_tps=value.get("throughput_tps"),  # type: ignore[arg-type]
        measurements=measurements,
    )
    validate_configuration(result)
    return result
