"""Load and canonically serialize portable Turbofile profiles."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .runtime_profile import Turbofile


class ProfileFormatError(ValueError):
    """A profile document is ambiguous or uses an unsupported format."""


def load_profile(path: str | Path) -> Turbofile:
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json":
        return load_json_profile(target)
    if suffix in {".yaml", ".yml"}:
        return load_yaml_profile(target)
    raise ProfileFormatError(f"unsupported profile format: {suffix or '<none>'}")


def load_json_profile(path: str | Path) -> Turbofile:
    text = Path(path).read_text(encoding="utf-8")
    mapping = json.loads(text, object_pairs_hook=_unique_json_object)
    return Turbofile.from_mapping(_root_mapping(mapping))


def load_yaml_profile(path: str | Path) -> Turbofile:
    try:
        yaml = _load_yaml_module()
    except ModuleNotFoundError as exc:
        raise ImportError(
            "PyYAML is required to load YAML Turbofiles; use JSON for the "
            "dependency-free path or install PyYAML"
        ) from exc
    loader = type("UniqueKeySafeLoader", (yaml.SafeLoader,), {})

    def construct_mapping(loader_instance: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        loader_instance.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader_instance.construct_object(key_node, deep=deep)
            if key in result:
                raise ProfileFormatError(f"duplicate key in YAML profile: {key}")
            result[key] = loader_instance.construct_object(value_node, deep=deep)
        return result

    loader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    mapping = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=loader)
    return Turbofile.from_mapping(_root_mapping(mapping))


def canonical_json(value: Turbofile | Mapping[str, Any]) -> str:
    plain: Any = value.to_mapping() if isinstance(value, Turbofile) else value
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def profile_digest(value: Turbofile | Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileFormatError(f"duplicate key in JSON profile: {key}")
        result[key] = value
    return result


def _root_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileFormatError("profile document root must be a mapping")
    return value


def _load_yaml_module() -> Any:
    return importlib.import_module("yaml")
