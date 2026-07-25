"""User-facing manual/auto runtime selection without direct process authority."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TextIO

from .hardware import HardwareFingerprint
from .recommend import hardware_satisfies
from .runtime_profile import AuxMode, Turbofile
from .selection import ProfileCatalog, load_selection, save_selection


def _has_local(profile: Turbofile) -> bool:
    return any(rung.aux_mode is not AuxMode.API for rung in profile.rungs)


def _manual_compatible(profile: Turbofile, hardware: HardwareFingerprint) -> bool:
    return not _has_local(profile) or hardware_satisfies(hardware, profile.hardware)


def run(
    argv: Sequence[str],
    *,
    catalog: ProfileCatalog,
    hardware: HardwareFingerprint,
    selection_path: str | Path,
    output: TextIO,
) -> int:
    command = argv[0] if argv else "list"
    if command == "list":
        rows = [
            {
                "id": profile.id,
                "class_vram_gb": profile.hardware.class_vram_gb,
                "topology": profile.hardware.topology,
                "local": _has_local(profile),
                "manual_compatible": _manual_compatible(profile, hardware),
                "rungs": [rung.id for rung in profile.rungs],
            }
            for profile in catalog.profiles
        ]
        output.write(json.dumps(rows, indent=2) + "\n")
        return 0

    if command in {"set", "use"} and len(argv) == 2:
        requested = argv[1]
        try:
            choice = catalog.select(hardware, requested=requested)
            save_selection(selection_path, choice)
        except (OSError, ValueError) as exc:
            output.write(json.dumps({"error": str(exc)}, indent=2) + "\n")
            return 2
        rung = choice.profile.rungs[choice.initial_rung_index]
        payload = {
            "configured": True,
            "mode": choice.mode.value,
            "requested": requested,
            "profile_id": choice.profile.id,
            "effective_rung": rung.id,
            "controller_pending": True,
            "selection_path": str(selection_path),
        }
        output.write(json.dumps(payload, indent=2) + "\n")
        return 0

    if command == "status":
        try:
            selection = load_selection(selection_path)
        except FileNotFoundError:
            output.write(
                json.dumps(
                    {"configured": False, "selection_path": str(selection_path)},
                    indent=2,
                )
                + "\n"
            )
            return 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            output.write(json.dumps({"configured": False, "error": str(exc)}, indent=2) + "\n")
            return 2
        output.write(json.dumps({"configured": True, "selection": selection}, indent=2) + "\n")
        return 0

    output.write(
        json.dumps(
            {"error": "usage: turbofit-runtime [list|set auto|set PROFILE|use PROFILE|status]"},
            indent=2,
        )
        + "\n"
    )
    return 2
