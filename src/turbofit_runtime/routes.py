"""Resolve portable rungs to Turbohaul tags and atomically publish gateway routes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .runtime_profile import AuxMode, Turbofile

RuntimeResolutions = dict[str, dict[str, dict[str, dict[str, int | str]]]]


def load_runtime_resolutions(path: str | Path) -> RuntimeResolutions:
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"schema", "profiles"}:
        raise ValueError("invalid runtime resolutions root")
    if raw["schema"] != "turbofit.runtime-resolutions/v1":
        raise ValueError("unsupported runtime resolutions schema")
    profiles = raw["profiles"]
    if not isinstance(profiles, Mapping):
        raise ValueError("runtime resolution profiles must be a mapping")
    result: RuntimeResolutions = {}
    for profile_id, rungs in profiles.items():
        if not isinstance(profile_id, str) or not profile_id or not isinstance(rungs, Mapping):
            raise ValueError("invalid runtime resolution profile")
        result[profile_id] = {}
        for rung_id, roles in rungs.items():
            if not isinstance(rung_id, str) or not rung_id or not isinstance(roles, Mapping):
                raise ValueError("invalid runtime resolution rung")
            if not roles or not set(roles) <= {"main", "aux"} or "main" not in roles:
                raise ValueError("local resolution requires main and optional aux roles")
            parsed_roles: dict[str, dict[str, int | str]] = {}
            for role, value in roles.items():
                if (
                    not isinstance(value, Mapping)
                    or not {"model_tag", "expected_vram_mb"} <= set(value)
                    or not set(value) <= {"model_tag", "expected_vram_mb", "split_mode"}
                ):
                    raise ValueError("invalid runtime resolution role")
                tag = value["model_tag"]
                expected = value["expected_vram_mb"]
                split_mode = value.get("split_mode", "none")
                if not isinstance(tag, str) or not tag or "/" in tag or "\\" in tag:
                    raise ValueError("runtime model_tag must be a portable tag")
                if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
                    raise ValueError("expected_vram_mb must be a positive integer")
                if split_mode not in {"none", "layer", "row"}:
                    raise ValueError("split_mode must be none, layer, or row")
                parsed_roles[role] = {
                    "model_tag": tag,
                    "expected_vram_mb": expected,
                    "split_mode": split_mode,
                }
            result[profile_id][rung_id] = parsed_roles
    return result


def build_route_state(
    profile: Turbofile,
    rung_index: int,
    resolutions: RuntimeResolutions,
    *,
    manager_port: int,
) -> dict[str, Any]:
    if isinstance(rung_index, bool) or not isinstance(rung_index, int):
        raise ValueError("rung_index must be an integer")
    if not 0 <= rung_index < len(profile.rungs):
        raise ValueError("rung_index is outside profile")
    if isinstance(manager_port, bool) or not isinstance(manager_port, int) or not 1 <= manager_port <= 65535:
        raise ValueError("manager_port must be in 1..65535")
    rung = profile.rungs[rung_index]
    if rung.aux_mode is AuxMode.API:
        assert rung.main_api_policy is not None and rung.aux_api_policy is not None
        routes = {
            "main": {"kind": "api-policy", "policy": rung.main_api_policy},
            "aux": {"kind": "api-policy", "policy": rung.aux_api_policy},
        }
    else:
        try:
            roles = resolutions[profile.id][rung.id]
        except KeyError as exc:
            raise ValueError(f"missing runtime resolution for {profile.id}/{rung.id}") from exc
        main = roles["main"]
        routes = {
            "main": {
                "kind": "local",
                "alias": main["model_tag"],
                "port": manager_port,
            }
        }
        if rung.aux_mode is AuxMode.SHARED_MAIN:
            routes["aux"] = {"kind": "shared-main"}
        else:
            aux = roles.get("aux")
            if aux is None:
                raise ValueError(f"dedicated rung {profile.id}/{rung.id} lacks aux resolution")
            routes["aux"] = {
                "kind": "local",
                "alias": aux["model_tag"],
                "port": manager_port,
                "mode": "dedicated",
            }
    return {
        "schema": "turbofit.runtime-routes/v1",
        "active": profile.id,
        "rung_id": rung.id,
        "rung_index": rung_index,
        "routes": routes,
    }


def publish_route_state(path: str | Path, state: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(state), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
