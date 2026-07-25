"""Portable, immutable Turbofit runtime-profile schema."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

SCHEMA = "turbofit.runtime/v1"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "token",
        "access_token",
        "bearer_token",
        "credential",
        "credentials",
        "private_key",
        "secret",
    }
)
_PLACEMENT_KEYS = frozenset(
    {"gpu", "gpu_index", "main_gpu", "device", "cuda_visible_devices", "tensor_split"}
)


class AuxMode(str, Enum):
    DEDICATED = "dedicated"
    SHARED_MAIN = "shared-main"
    API = "api"


@dataclass(frozen=True)
class HardwareConstraint:
    class_vram_gb: float
    min_devices: int
    total_vram_gb: float
    per_device_min_gb: float
    accelerator: str
    compute_capability_min: str | None = None
    system_ram_gb: float | None = None
    topology: str = "any"

    def validate(self) -> None:
        _positive_number(self.class_vram_gb, "class_vram_gb")
        _positive_int(self.min_devices, "min_devices")
        _positive_number(self.total_vram_gb, "total_vram_gb")
        _positive_number(self.per_device_min_gb, "per_device_min_gb")
        _nonempty(self.accelerator, "accelerator")
        _nonempty(self.topology, "topology")
        if self.compute_capability_min is not None:
            _nonempty(self.compute_capability_min, "compute_capability_min")
        if self.system_ram_gb is not None:
            _positive_number(self.system_ram_gb, "system_ram_gb")
        required = self.min_devices * self.per_device_min_gb
        if self.total_vram_gb < required:
            raise ValueError(
                "total_vram_gb must be >= min_devices * per_device_min_gb"
            )


@dataclass(frozen=True)
class RuntimePolicy:
    recommendation: str
    external_gpu_priority: str
    contraction_dwell_s: float
    expansion_dwell_s: float
    expansion_margin_gb_per_card: float
    cooldown_s: float
    flap_failure_limit: int | None = None
    flap_window_s: float | None = None

    def validate(self) -> None:
        _nonempty(self.recommendation, "recommendation")
        if self.external_gpu_priority != "absolute":
            raise ValueError("external_gpu_priority must equal 'absolute'")
        for name in (
            "contraction_dwell_s",
            "expansion_dwell_s",
            "expansion_margin_gb_per_card",
            "cooldown_s",
        ):
            _nonnegative_number(getattr(self, name), name)
        if self.flap_failure_limit is not None:
            _positive_int(self.flap_failure_limit, "flap_failure_limit")
        if self.flap_window_s is not None:
            _nonnegative_number(self.flap_window_s, "flap_window_s")


@dataclass(frozen=True)
class RoleRoutes:
    main: str
    auxiliary: str
    fallback: str

    def validate(self) -> None:
        _nonempty(self.main, "roles.main")
        _nonempty(self.auxiliary, "roles.auxiliary")
        _nonempty(self.fallback, "roles.fallback")


@dataclass(frozen=True)
class RuntimeRung:
    id: str
    context: int
    aux_mode: AuxMode
    evidence: str
    main_manifest: str | None = None
    aux_manifest: str | None = None
    main_api_policy: str | None = None
    aux_api_policy: str | None = None

    def validate(self, *, terminal: bool) -> None:
        _slug(self.id, "rung id")
        _positive_int(self.context, "context")
        _nonempty(self.evidence, "evidence")
        if terminal:
            if self.aux_mode is not AuxMode.API:
                raise ValueError("terminal rung must have aux_mode=api")
            _nonempty(self.main_api_policy, "main_api_policy")
            _nonempty(self.aux_api_policy, "aux_api_policy")
            if self.main_manifest is not None or self.aux_manifest is not None:
                raise ValueError("terminal rung cannot contain local manifests")
            return
        if self.aux_mode is AuxMode.API:
            raise ValueError("only terminal rung may have aux_mode=api")
        _manifest(self.main_manifest, "main_manifest")
        if self.aux_mode is AuxMode.DEDICATED:
            _manifest(self.aux_manifest, "aux_manifest")
        elif self.aux_manifest is not None:
            raise ValueError("shared-main rung cannot contain aux_manifest")
        if self.main_api_policy is not None or self.aux_api_policy is not None:
            raise ValueError("local rung cannot contain API policies")


@dataclass(frozen=True)
class Turbofile:
    schema: str
    id: str
    revision: int
    hardware: HardwareConstraint
    policy: RuntimePolicy
    roles: RoleRoutes
    rungs: tuple[RuntimeRung, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rungs", tuple(self.rungs))

    def validate(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"schema must equal {SCHEMA!r}")
        _slug(self.id, "id")
        _positive_int(self.revision, "revision")
        self.hardware.validate()
        self.policy.validate()
        self.roles.validate()
        if not self.rungs:
            raise ValueError("rungs must not be empty")
        ids = [rung.id for rung in self.rungs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rung id")
        for index, rung in enumerate(self.rungs):
            rung.validate(terminal=index == len(self.rungs) - 1)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Turbofile":
        root = _mapping(mapping, "root")
        _portable_scan(root)
        _exact_fields(
            root,
            required={"schema", "id", "revision", "hardware", "policy", "roles", "rungs"},
            optional=set(),
            where="root",
        )
        hardware_data = _mapping(root["hardware"], "hardware")
        _exact_fields(
            hardware_data,
            required={
                "class_vram_gb",
                "min_devices",
                "total_vram_gb",
                "per_device_min_gb",
                "accelerator",
            },
            optional={"compute_capability_min", "system_ram_gb", "topology"},
            where="hardware",
        )
        policy_data = _mapping(root["policy"], "policy")
        _exact_fields(
            policy_data,
            required={
                "recommendation",
                "external_gpu_priority",
                "contraction_dwell_s",
                "expansion_dwell_s",
                "expansion_margin_gb_per_card",
                "cooldown_s",
            },
            optional={"flap_failure_limit", "flap_window_s"},
            where="policy",
        )
        roles_data = _mapping(root["roles"], "roles")
        _exact_fields(
            roles_data,
            required={"main", "auxiliary", "fallback"},
            optional=set(),
            where="roles",
        )
        rung_values = root["rungs"]
        if not isinstance(rung_values, list):
            raise ValueError("rungs must be a list")
        rungs: list[RuntimeRung] = []
        for index, value in enumerate(rung_values):
            data = _mapping(value, f"rungs[{index}]")
            _exact_fields(
                data,
                required={"id", "context", "aux_mode", "evidence"},
                optional={
                    "main_manifest",
                    "aux_manifest",
                    "main_api_policy",
                    "aux_api_policy",
                },
                where=f"rungs[{index}]",
            )
            try:
                aux_mode = AuxMode(data["aux_mode"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"rungs[{index}].aux_mode is invalid") from exc
            rungs.append(
                RuntimeRung(
                    id=data["id"],
                    context=data["context"],
                    aux_mode=aux_mode,
                    evidence=data["evidence"],
                    main_manifest=data.get("main_manifest"),
                    aux_manifest=data.get("aux_manifest"),
                    main_api_policy=data.get("main_api_policy"),
                    aux_api_policy=data.get("aux_api_policy"),
                )
            )
        profile = cls(
            schema=root["schema"],
            id=root["id"],
            revision=root["revision"],
            hardware=HardwareConstraint(
                class_vram_gb=hardware_data["class_vram_gb"],
                min_devices=hardware_data["min_devices"],
                total_vram_gb=hardware_data["total_vram_gb"],
                per_device_min_gb=hardware_data["per_device_min_gb"],
                accelerator=hardware_data["accelerator"],
                compute_capability_min=hardware_data.get("compute_capability_min"),
                system_ram_gb=hardware_data.get("system_ram_gb"),
                topology=hardware_data.get("topology", "any"),
            ),
            policy=RuntimePolicy(
                recommendation=policy_data["recommendation"],
                external_gpu_priority=policy_data["external_gpu_priority"],
                contraction_dwell_s=policy_data["contraction_dwell_s"],
                expansion_dwell_s=policy_data["expansion_dwell_s"],
                expansion_margin_gb_per_card=policy_data["expansion_margin_gb_per_card"],
                cooldown_s=policy_data["cooldown_s"],
                flap_failure_limit=policy_data.get("flap_failure_limit"),
                flap_window_s=policy_data.get("flap_window_s"),
            ),
            roles=RoleRoutes(
                main=roles_data["main"],
                auxiliary=roles_data["auxiliary"],
                fallback=roles_data["fallback"],
            ),
            rungs=tuple(rungs),
        )
        profile.validate()
        return profile

    def to_mapping(self) -> dict[str, Any]:
        hardware: dict[str, Any] = {
            "class_vram_gb": self.hardware.class_vram_gb,
            "min_devices": self.hardware.min_devices,
            "total_vram_gb": self.hardware.total_vram_gb,
            "per_device_min_gb": self.hardware.per_device_min_gb,
            "accelerator": self.hardware.accelerator,
            "topology": self.hardware.topology,
        }
        _optional(hardware, "compute_capability_min", self.hardware.compute_capability_min)
        _optional(hardware, "system_ram_gb", self.hardware.system_ram_gb)
        policy: dict[str, Any] = {
            "recommendation": self.policy.recommendation,
            "external_gpu_priority": self.policy.external_gpu_priority,
            "contraction_dwell_s": self.policy.contraction_dwell_s,
            "expansion_dwell_s": self.policy.expansion_dwell_s,
            "expansion_margin_gb_per_card": self.policy.expansion_margin_gb_per_card,
            "cooldown_s": self.policy.cooldown_s,
        }
        _optional(policy, "flap_failure_limit", self.policy.flap_failure_limit)
        _optional(policy, "flap_window_s", self.policy.flap_window_s)
        rungs: list[dict[str, Any]] = []
        for rung in self.rungs:
            item: dict[str, Any] = {
                "id": rung.id,
                "context": rung.context,
                "aux_mode": rung.aux_mode.value,
                "evidence": rung.evidence,
            }
            for key in (
                "main_manifest",
                "aux_manifest",
                "main_api_policy",
                "aux_api_policy",
            ):
                _optional(item, key, getattr(rung, key))
            rungs.append(item)
        return {
            "schema": self.schema,
            "id": self.id,
            "revision": self.revision,
            "hardware": hardware,
            "policy": policy,
            "roles": {
                "main": self.roles.main,
                "auxiliary": self.roles.auxiliary,
                "fallback": self.roles.fallback,
            },
            "rungs": rungs,
        }


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    return value


def _exact_fields(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], where: str
) -> None:
    unknown = set(value) - required - optional
    if unknown:
        raise ValueError(f"unknown field in {where}: {sorted(unknown)[0]}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"missing field in {where}: {sorted(missing)[0]}")


def _portable_scan(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            lowered = key.lower()
            if lowered in _SECRET_KEYS:
                raise ValueError(f"secret field is not portable: {path}.{key}")
            if lowered in _PLACEMENT_KEYS:
                raise ValueError(f"placement field is not portable: {path}.{key}")
            _portable_scan(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _portable_scan(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if value.startswith("/") or value.startswith("~/"):
            raise ValueError(f"local path is not portable: {path}")
        if lowered.startswith("bearer ") or lowered.startswith("sk-"):
            raise ValueError(f"secret value is not portable: {path}")


def _slug(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ValueError(f"{name} must be a portable lowercase slug")


def _nonempty(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")


def _nonnegative_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
    ):
        raise ValueError(f"{name} must be non-negative")


def _manifest(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a sha256: content address")


def _optional(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value
