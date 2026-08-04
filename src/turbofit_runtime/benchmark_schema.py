"""Immutable benchmark evidence schema and promotion gates."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_STAGE_IDS = (
    "artifact",
    "runtime",
    "performance",
    "quality",
    "pressure-self-heal",
)
RECORD_FIELDS = frozenset({
    "candidate_id",
    "artifact_hashes",
    "host_fingerprint",
    "observed_context",
    "throughput_tps",
    "ttft_ms",
    "per_card_vram_mb",
    "power_w_by_card",
    "quality_score",
    "raw_result_identity",
    "stages",
})


@dataclass(frozen=True)
class StageSpec:
    id: str
    required: bool
    required_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.id not in REQUIRED_STAGE_IDS:
            raise ValueError(f"unknown benchmark stage: {self.id}")
        if not self.required_fields:
            raise ValueError(f"stage {self.id} has no required_fields")
        unknown = set(self.required_fields) - RECORD_FIELDS
        if unknown:
            raise ValueError(f"stage {self.id} has unknown required_fields: {sorted(unknown)}")


@dataclass(frozen=True)
class BenchmarkSuite:
    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        ids = tuple(stage.id for stage in self.stages)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate benchmark stage")
        if ids != REQUIRED_STAGE_IDS:
            raise ValueError(
                f"incomplete or unordered benchmark stages: expected {REQUIRED_STAGE_IDS}"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BenchmarkSuite":
        _exact(mapping, {"schema", "stages"}, "suite")
        if mapping.get("schema") != "turbofit.benchmark-suite/v1":
            raise ValueError("unsupported benchmark suite schema")
        raw_stages = mapping.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("suite.stages must be a list")
        stages: list[StageSpec] = []
        for index, raw in enumerate(raw_stages):
            if not isinstance(raw, Mapping):
                raise ValueError(f"suite.stages[{index}] must be a mapping")
            _exact(raw, {"id", "required", "required_fields"}, f"suite.stages[{index}]")
            fields = raw.get("required_fields")
            if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
                raise ValueError(f"suite.stages[{index}].required_fields must be strings")
            required = raw.get("required")
            if not isinstance(required, bool):
                raise ValueError(f"suite.stages[{index}].required must be boolean")
            stages.append(StageSpec(str(raw.get("id")), required, tuple(fields)))
        return cls(tuple(stages))


@dataclass(frozen=True)
class StageResult:
    stage: str
    passed: bool
    evidence_identity: str

    def __post_init__(self) -> None:
        if self.stage not in REQUIRED_STAGE_IDS:
            raise ValueError(f"unknown stage result: {self.stage}")
        if not isinstance(self.passed, bool):
            raise ValueError("stage passed must be boolean")
        _hash(self.evidence_identity, "stage evidence_identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "evidence_identity": self.evidence_identity,
        }


@dataclass(frozen=True)
class BenchmarkRecord:
    candidate_id: str
    artifact_hashes: tuple[str, ...]
    host_fingerprint: str
    observed_context: int
    throughput_tps: float
    ttft_ms: float
    per_card_vram_mb: tuple[int, ...]
    power_w_by_card: tuple[float, ...]
    quality_score: float
    raw_result_identity: str
    stages: tuple[StageResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.artifact_hashes:
            raise ValueError("artifact_hashes must be non-empty")
        for item in self.artifact_hashes:
            _hash(item, "artifact_hashes")
        _hash(self.host_fingerprint, "host_fingerprint")
        _positive_int(self.observed_context, "observed_context")
        _positive_number(self.throughput_tps, "throughput_tps")
        _nonnegative_number(self.ttft_ms, "ttft_ms")
        if not self.per_card_vram_mb:
            raise ValueError("per_card_vram_mb must be non-empty")
        for value in self.per_card_vram_mb:
            _positive_int(value, "per_card_vram_mb")
        for value in self.power_w_by_card:
            _positive_number(value, "power_w_by_card")
        if self.power_w_by_card and len(self.power_w_by_card) != len(self.per_card_vram_mb):
            raise ValueError("power_w_by_card must align with per_card_vram_mb")
        if isinstance(self.quality_score, bool) or not isinstance(self.quality_score, (int, float)):
            raise ValueError("quality_score must be numeric")
        if not 0 <= float(self.quality_score) <= 1:
            raise ValueError("quality_score must be between 0 and 1")
        _hash(self.raw_result_identity, "raw_result_identity")
        ids = tuple(result.stage for result in self.stages)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate stage result")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BenchmarkRecord":
        _exact(mapping, set(RECORD_FIELDS), "record")
        raw_stages = mapping.get("stages")
        if not isinstance(raw_stages, list):
            raise ValueError("record.stages must be a list")
        stages: list[StageResult] = []
        for index, raw in enumerate(raw_stages):
            if not isinstance(raw, Mapping):
                raise ValueError(f"record.stages[{index}] must be a mapping")
            _exact(raw, {"stage", "passed", "evidence_identity"}, f"record.stages[{index}]")
            stages.append(StageResult(
                stage=str(raw.get("stage")),
                passed=raw.get("passed"),
                evidence_identity=str(raw.get("evidence_identity")),
            ))
        return cls(
            candidate_id=str(mapping.get("candidate_id")),
            artifact_hashes=tuple(mapping.get("artifact_hashes") or ()),
            host_fingerprint=str(mapping.get("host_fingerprint")),
            observed_context=mapping.get("observed_context"),
            throughput_tps=mapping.get("throughput_tps"),
            ttft_ms=mapping.get("ttft_ms"),
            per_card_vram_mb=tuple(mapping.get("per_card_vram_mb") or ()),
            power_w_by_card=tuple(mapping.get("power_w_by_card") or ()),
            quality_score=mapping.get("quality_score"),
            raw_result_identity=str(mapping.get("raw_result_identity")),
            stages=tuple(stages),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "artifact_hashes": list(self.artifact_hashes),
            "host_fingerprint": self.host_fingerprint,
            "observed_context": self.observed_context,
            "throughput_tps": self.throughput_tps,
            "ttft_ms": self.ttft_ms,
            "per_card_vram_mb": list(self.per_card_vram_mb),
            "power_w_by_card": list(self.power_w_by_card),
            "quality_score": self.quality_score,
            "raw_result_identity": self.raw_result_identity,
            "stages": [result.to_mapping() for result in self.stages],
        }


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    failures: tuple[str, ...]


class PromotionRejected(ValueError):
    pass


def require_promotion(record: BenchmarkRecord, suite: BenchmarkSuite) -> PromotionDecision:
    by_stage = {result.stage: result for result in record.stages}
    failures: list[str] = []
    mapping = record.to_mapping()
    for stage in suite.stages:
        if not stage.required:
            continue
        result = by_stage.get(stage.id)
        if result is None:
            failures.append(f"missing stage: {stage.id}")
            continue
        if not result.passed:
            failures.append(f"failed stage: {stage.id}")
        for field in stage.required_fields:
            value = mapping[field]
            if value is None or value == "" or value == []:
                failures.append(f"{stage.id} missing field: {field}")
    if failures:
        raise PromotionRejected("; ".join(failures))
    return PromotionDecision(True, ())


def load_suite(path: str | Path) -> BenchmarkSuite:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        mapping = json.loads(content)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load benchmark suite YAML") from exc
        mapping = yaml.safe_load(content)
    if not isinstance(mapping, Mapping):
        raise ValueError("benchmark suite root must be a mapping")
    return BenchmarkSuite.from_mapping(mapping)


def load_record(path: str | Path) -> BenchmarkRecord:
    import json

    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(mapping, Mapping):
        raise ValueError("benchmark record root must be a mapping")
    return BenchmarkRecord.from_mapping(mapping)


def _exact(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    missing = allowed - set(mapping)
    if unknown:
        raise ValueError(f"{name} unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} missing fields: {sorted(missing)}")


def _hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase sha256 identity")


def _positive_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _positive_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be positive")


def _nonnegative_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field} must be nonnegative")
