"""Hardware-aware multimodal model catalog and Hermes configuration bridge."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .hardware import HardwareFingerprint


SCHEMA = "turbofit.multimodal-catalog/v1"
MODALITIES = ("image", "video", "music", "tts", "stt")
INTEGRATIONS = {"builtin", "hermes-tool", "turbofit-local", "turbofit-local-int8-offload", "candidate"}
ADAPTER_ACTIONS = "install-adapter"


@dataclass(frozen=True)
class MultimodalModel:
    id: str
    name: str
    modalities: tuple[str, ...]
    source: str
    revision: str
    runtime: str
    minimum_memory_mb: int
    recommended_memory_mb: int
    minimum_accelerator_memory_mb: int
    platforms: tuple[str, ...]
    integration: str
    components: tuple[tuple[str, str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MultimodalModel":
        required = {
            "id", "name", "modalities", "source", "revision", "runtime",
            "minimum_memory_mb", "recommended_memory_mb", "platforms", "integration",
        }
        optional = {"minimum_accelerator_memory_mb", "components"}
        if not required <= set(value) or not set(value) <= required | optional:
            raise ValueError("multimodal model fields do not match schema")
        components = tuple(
            (str(item["role"]), str(item["source"]), str(item["revision"]))
            for item in value.get("components", ())
        )
        item = cls(
            id=str(value["id"]),
            name=str(value["name"]),
            modalities=tuple(value["modalities"]),
            source=str(value["source"]),
            revision=str(value["revision"]),
            runtime=str(value["runtime"]),
            minimum_memory_mb=value["minimum_memory_mb"],
            recommended_memory_mb=value["recommended_memory_mb"],
            minimum_accelerator_memory_mb=value.get("minimum_accelerator_memory_mb", 0),
            platforms=tuple(value["platforms"]),
            integration=str(value["integration"]),
            components=components,
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.id):
            raise ValueError(f"invalid multimodal model id: {self.id}")
        if not self.name or not self.runtime:
            raise ValueError(f"multimodal model {self.id} has empty metadata")
        if not self.modalities or not set(self.modalities) <= set(MODALITIES):
            raise ValueError(f"multimodal model {self.id} has invalid modalities")
        if self.source == "builtin:hermes":
            if self.revision != "builtin":
                raise ValueError(f"builtin model {self.id} must use builtin revision")
        elif not self.source.startswith("https://huggingface.co/") or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(f"model {self.id} must use a pinned Hugging Face source")
        for name in ("minimum_memory_mb", "recommended_memory_mb", "minimum_accelerator_memory_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"model {self.id} has invalid {name}")
        if self.recommended_memory_mb < self.minimum_memory_mb:
            raise ValueError(f"model {self.id} recommended memory is below minimum")
        if not self.platforms or not set(self.platforms) <= {"linux", "windows", "darwin"}:
            raise ValueError(f"model {self.id} has invalid platform support")
        if self.integration not in INTEGRATIONS:
            raise ValueError(f"model {self.id} has invalid integration")
        seen_roles: set[str] = set()
        for role, source, revision in self.components:
            if not role or role in seen_roles:
                raise ValueError(f"model {self.id} has an empty or duplicate component role")
            seen_roles.add(role)
            if not (source.startswith("https://huggingface.co/") and re.fullmatch(r"[0-9a-f]{40}", revision)):
                raise ValueError(f"model {self.id} component {role} must be a pinned Hugging Face source")


@dataclass(frozen=True)
class MultimodalCatalog:
    models: tuple[MultimodalModel, ...]

    @classmethod
    def load(cls, path: str | Path) -> "MultimodalCatalog":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or set(raw) != {"schema", "models"} or raw["schema"] != SCHEMA:
            raise ValueError("invalid multimodal catalog root")
        catalog = cls(models=tuple(MultimodalModel.from_mapping(item) for item in raw["models"]))
        ids = [item.id for item in catalog.models]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate multimodal model id")
        for modality in MODALITIES:
            if not any(modality in item.modalities for item in catalog.models):
                raise ValueError(f"multimodal catalog has no {modality} option")
        return catalog

    def by_id(self, model_id: str) -> MultimodalModel:
        for item in self.models:
            if item.id == model_id:
                return item
        raise ValueError(f"unknown multimodal model: {model_id}")


def _platform_name(hardware: HardwareFingerprint) -> str:
    name = hardware.os.lower()
    if name.startswith("win"):
        return "windows"
    if name in {"macos", "mac", "osx"}:
        return "darwin"
    return "linux" if "linux" in name else name


def _as_mapping(state: Any) -> Mapping[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state
    as_dict = getattr(state, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    raise TypeError(f"runtime state must be a mapping or expose as_dict(), got {type(state).__name__}")


def _action(
    item: MultimodalModel,
    *,
    fit: bool,
    platform_supported: bool,
    accelerator_fit: bool,
    runtime_state: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Return (action, blocked_by).

    Memory fit alone never makes a local runtime usable: when the caller probes
    the declared runtime and it is not ready, the honest answer is "install the
    adapter", not "run it now".
    """
    if not fit:
        if not platform_supported:
            return "unsupported-platform", "platform"
        if not accelerator_fit:
            return "insufficient-accelerator-memory", "accelerator"
        return "insufficient-memory", "memory"

    if runtime_state is not None:
        state = str(runtime_state.get("status", ""))
        if state not in {"ready", "delegated"}:
            return ADAPTER_ACTIONS, state or "runtime"
        if item.integration == "builtin" and state != "delegated" and item.runtime != "hermes":
            return ADAPTER_ACTIONS, state

    if item.integration == "builtin":
        return "use-now", ""
    if item.integration == "hermes-tool":
        return "enable-hermes-tool", ""
    if item.integration in {"turbofit-local", "turbofit-local-int8-offload"}:
        return "run-turbofit-local", ""
    return ADAPTER_ACTIONS, "adapter"


def recommend_multimodal(
    *,
    hardware: HardwareFingerprint,
    catalog: MultimodalCatalog,
    runtimes: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank every modality against physical memory, OS support and runtime readiness.

    ``runtimes`` maps a runtime name to a probe result (see
    :func:`turbofit_runtime.media_pipeline.probe_runtime`).  When supplied
    entries whose engine or weights are missing are reported as
    ``install-adapter`` with a ``blocked_by`` reason and the fix command.
    """
    usable = hardware.total_usable_memory_mb
    platform = _platform_name(hardware)
    modalities: dict[str, list[dict[str, Any]]] = {name: [] for name in MODALITIES}
    for item in catalog.models:
        platform_supported = platform in item.platforms
        accelerator_fit = hardware.total_vram_mb >= item.minimum_accelerator_memory_mb
        fit = platform_supported and usable >= item.minimum_memory_mb and accelerator_fit
        recommended_fit = fit and usable >= item.recommended_memory_mb
        runtime_state = None
        if runtimes is not None:
            runtime_state = runtimes.get(item.runtime)
        state_map = _as_mapping(runtime_state)
        action, blocked_by = _action(
            item,
            fit=fit,
            platform_supported=platform_supported,
            accelerator_fit=accelerator_fit,
            runtime_state=state_map,
        )
        status = None if state_map is None else str(state_map.get("status", ""))
        command = "" if state_map is None else str(state_map.get("command", ""))
        payload = {
            "id": item.id,
            "name": item.name,
            "source": item.source,
            "revision": item.revision,
            "runtime": item.runtime,
            "integration": item.integration,
            "components": [
                {"role": role, "source": source, "revision": revision}
                for role, source, revision in item.components
            ],
            "minimum_memory_mb": item.minimum_memory_mb,
            "recommended_memory_mb": item.recommended_memory_mb,
            "minimum_accelerator_memory_mb": item.minimum_accelerator_memory_mb,
            "accelerator_fit": accelerator_fit,
            "platform_supported": platform_supported,
            "fit": fit,
            "recommended_fit": recommended_fit,
            "action": action,
            "blocked_by": blocked_by,
            "runtime_status": status,
            "runtime_ready": None if state_map is None else bool(state_map.get("ready")),
            "fix_command": command,
        }
        for modality in item.modalities:
            modalities[modality].append(dict(payload))
    for rows in modalities.values():
        rows.sort(
            key=lambda item: (
                not item["fit"],
                not item["recommended_fit"],
                item["integration"] == "candidate",
                -item["recommended_memory_mb"],
                item["name"],
            )
        )
    return {
        "schema": SCHEMA,
        "platform": platform,
        "topology_key": hardware.topology_key,
        "system_ram_mb": hardware.system_ram_mb,
        "accelerator_memory_mb": hardware.total_vram_mb,
        "total_usable_memory_mb": usable,
        "modalities": modalities,
    }


def configure_multimodal(
    config: Mapping[str, Any],
    *,
    selections: Mapping[str, str],
    catalog: MultimodalCatalog,
) -> dict[str, Any]:
    """Store modality choices and apply selections Hermes supports directly."""
    updated = copy.deepcopy(dict(config))
    normalized: dict[str, str] = {}
    for modality, model_id in selections.items():
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality: {modality}")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"selection for {modality} must be a model id")
        item = catalog.by_id(model_id)
        if modality not in item.modalities:
            raise ValueError(f"model {model_id} does not support {modality}")
        normalized[modality] = model_id

    turbofit = updated.get("turbofit")
    if not isinstance(turbofit, Mapping):
        turbofit = {}
    turbofit = copy.deepcopy(dict(turbofit))
    multimodal = turbofit.get("multimodal")
    if not isinstance(multimodal, Mapping):
        multimodal = {}
    multimodal = copy.deepcopy(dict(multimodal))
    current = multimodal.get("selected")
    current = copy.deepcopy(dict(current)) if isinstance(current, Mapping) else {}
    current.update(normalized)
    multimodal["selected"] = current
    turbofit["multimodal"] = multimodal
    updated["turbofit"] = turbofit

    tts_id = normalized.get("tts")
    if tts_id == "hermes-edge-tts":
        tts = updated.get("tts")
        tts = copy.deepcopy(dict(tts)) if isinstance(tts, Mapping) else {}
        tts["provider"] = "edge"
        updated["tts"] = tts
    if normalized.get("stt") == "hermes-local-stt":
        stt = updated.get("stt")
        stt = copy.deepcopy(dict(stt)) if isinstance(stt, Mapping) else {}
        stt["provider"] = "local"
        updated["stt"] = stt
    if normalized.get("image") == "hermes-image":
        image_gen = updated.get("image_gen")
        if not isinstance(image_gen, Mapping):
            updated["image_gen"] = {}
    if normalized.get("video") == "hermes-video":
        video_gen = updated.get("video_gen")
        if not isinstance(video_gen, Mapping):
            updated["video_gen"] = {}
    return updated
