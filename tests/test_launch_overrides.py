"""TF-1/TF-2 lane tests: typed launch overrides and independent role contexts."""
from __future__ import annotations

import pytest

from turbofit_runtime.recipes import RecipeBook
from turbofit_runtime.runtime_profile import Turbofile


def book(*, flavor: str = "mainline") -> RecipeBook:
    data = {
        "schema_version": 1,
        "atomic_binary": "bin/llama-server",
        "models": {
            "test-main": {
                "alias": "test-main",
                "kind": "process",
                "model": "${TURBOFIT_MODEL_ROOT}/main.gguf",
                "runtime_flavor": flavor,
                "methods": {"204800": "mtp", "262144": "baseline"},
                "launch_overrides": {
                    "threads": 24,
                    "checkpoints": 0,
                    "mtp_n_max": 3,
                },
            },
            "test-aux": {
                "alias": "test-aux",
                "kind": "process",
                "model": "${TURBOFIT_MODEL_ROOT}/aux.gguf",
                "runtime_flavor": flavor,
                "methods": {"204800": "baseline", "262144": "baseline"},
            },
        },
    }
    return RecipeBook(data, platform_name="linux", backend_name="cuda")


def flag_value(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


# --- TF-1: exact typed launch overrides ------------------------------------


def test_typed_launch_overrides_emit_exact_flags() -> None:
    component = book().resolve_named("t", "test-main", "test-aux", 204_800).components[1]

    assert component.role == "main"
    assert flag_value(component.command, "--threads") == "24"
    assert flag_value(component.command, "--ctx-checkpoints") == "0"
    assert "--no-context-shift" not in component.command


def test_mtp_n_max_override_composes_ik_spec_type() -> None:
    component = book(flavor="ik").resolve_named("t", "test-main", "test-aux", 204_800).components[1]

    assert flag_value(component.command, "--spec-type") == "mtp:n_max=3,p_min=0.5"


def test_mainline_mtp_override_uses_typed_draft_n_max_flag() -> None:
    component = book().resolve_named("t", "test-main", "test-aux", 204_800).components[1]

    assert flag_value(component.command, "--spec-draft-n-max") == "3"


def test_context_override_launch_overrides_win_over_spec_level() -> None:
    data = book().data
    data["models"]["test-main"]["context_overrides"] = {
        "204800": {"launch_overrides": {"threads": 12}}
    }
    component = RecipeBook(data, platform_name="linux", backend_name="cuda").resolve_named(
        "t", "test-main", "test-aux", 204_800
    ).components[1]

    assert flag_value(component.command, "--threads") == "12"


def test_unknown_launch_override_key_is_rejected() -> None:
    data = book().data
    data["models"]["test-main"]["launch_overrides"] = {
        "extra_args": ["sh", "-c", "rm", "-rf", "/"]
    }
    with pytest.raises(ValueError, match="unknown launch override"):
        RecipeBook(data, platform_name="linux", backend_name="cuda").resolve_named(
            "t", "test-main", "test-aux", 204_800
        )


def test_wrong_typed_launch_override_value_is_rejected() -> None:
    for bad in ("24", -1, True):
        data = book().data
        data["models"]["test-main"]["launch_overrides"] = {"threads": bad}
        with pytest.raises(ValueError):
            RecipeBook(data, platform_name="linux", backend_name="cuda").resolve_named(
                "t", "test-main", "test-aux", 204_800
            )


def test_cache_type_override_applies_to_ik_flavor() -> None:
    data = book(flavor="ik").data
    data["models"]["test-main"]["launch_overrides"] = {
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
    }
    component = RecipeBook(data, platform_name="linux", backend_name="cuda").resolve_named(
        "t", "test-main", "test-aux", 262_144
    ).components[1]

    assert flag_value(component.command, "--cache-type-k") == "q4_0"
    assert flag_value(component.command, "--cache-type-v") == "q4_0"


def test_slot_similarity_and_parallel_overrides_are_typed() -> None:
    data = book().data
    data["models"]["test-aux"]["launch_overrides"] = {
        "slot_similarity": 1.0,
        "parallel": 1,
        "checkpoints": 32,
    }
    aux = RecipeBook(data, platform_name="linux", backend_name="cuda").resolve_named(
        "t", "test-main", "test-aux", 262_144
    ).components[0]

    assert aux.role == "aux"
    assert flag_value(aux.command, "--slot-prompt-similarity") == "1.0"
    assert flag_value(aux.command, "--parallel") == "1"
    assert flag_value(aux.command, "--ctx-checkpoints") == "32"


# --- TF-2: independent role contexts ----------------------------------------


def test_dedicated_roles_resolve_distinct_contexts() -> None:
    resolved = book().resolve_named("t", "test-main", "test-aux", 204_800, aux_context=262_144)

    main, aux = resolved.components[1], resolved.components[0]
    assert flag_value(main.command, "-c") == "204800"
    assert flag_value(aux.command, "-c") == "262144"
    assert resolved.main_context == 204_800
    assert resolved.aux_context == 262_144


def test_unknown_context_fails_clearly() -> None:
    with pytest.raises(ValueError, match="no method recipe for context 4096"):
        book().resolve_named("t", "test-main", "test-aux", 4_096)


def test_shared_main_rejects_mismatched_aux_context() -> None:
    with pytest.raises(ValueError, match="shared-main"):
        book().resolve_named("t", "test-main", "auto", 204_800, aux_context=262_144)


def test_equal_context_resolution_stays_backward_compatible() -> None:
    resolved = book().resolve_named("t", "test-main", "test-aux", 262_144)

    assert resolved.main_context == 262_144
    assert resolved.aux_context == 262_144
    for component in resolved.components:
        assert flag_value(component.command, "-c") == "262144"


def test_catalog_configuration_accepts_auxiliary_context() -> None:
    resolved = book().resolve_catalog_configuration({
        "id": "t", "main": "test-main", "auxiliary": "test-aux",
        "context": 204_800, "auxiliary_context": 262_144, "status": "candidate",
    })

    assert resolved.aux_context == 262_144


# --- TF-2: profile representation carries per-role context ------------------


def _profile_with_distinct_contexts():
    from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
    from turbofit_runtime.manual_profiles import build_manual_profile_payload
    from turbofit_runtime.recipes import ResolvedComponent, ResolvedRecipe

    recipe = ResolvedRecipe(
        row_id="t", profile_name="t", main_alias="test-main", aux_alias="test-aux",
        aux_mode="dedicated",
        components=(
            ResolvedComponent("aux", "test-aux", "test-aux", "process", "baseline", "1", 11610, ("/bin/true",)),
            ResolvedComponent("main", "test-main", "test-main", "process", "mtp", "0", 11605, ("/bin/true",)),
        ),
        main_context=204_800,
        aux_context=262_144,
    )
    # components need their resolved contexts for the sidecar handoff
    recipe = ResolvedRecipe(
        row_id=recipe.row_id,
        profile_name=recipe.profile_name,
        main_alias=recipe.main_alias,
        aux_alias=recipe.aux_alias,
        aux_mode=recipe.aux_mode,
        components=(
            ResolvedComponent("aux", "test-aux", "test-aux", "process", "baseline", "1", 11610, ("/bin/true",), context=262_144),
            ResolvedComponent("main", "test-main", "test-main", "process", "mtp", "0", 11605, ("/bin/true",), context=204_800),
        ),
        main_context=204_800,
        aux_context=262_144,
    )
    hardware = HardwareFingerprint(
        "linux", "x86_64", 262_144,
        devices=(AcceleratorDevice(0, "g0", "RTX", "nvidia", "cuda", 24_576, "8.6", "01"),),
    )
    entry = {
        "context": 204_800,
        "production_recipe_sha256": "sha256:" + "a" * 64,
        "metrics": {"gpu_peak_mb": {"0": 12_000}},
    }
    return build_manual_profile_payload(
        profile_id="t", profile_entry=entry, recipe=recipe, hardware=hardware
    )


def test_manual_profile_records_distinct_role_contexts() -> None:
    profile, resolutions, _ = _profile_with_distinct_contexts()

    rung = profile["rungs"][0]
    assert rung["context"] == 204_800
    assert rung["main_context"] == 204_800
    assert rung["auxiliary_context"] == 262_144
    roles = resolutions["profiles"]["t"]["manual-exact"]
    assert roles["main"]["context"] == 204_800
    assert roles["aux"]["context"] == 262_144


def test_manual_profile_with_distinct_contexts_loads_through_turbofile() -> None:
    profile, _, _ = _profile_with_distinct_contexts()

    parsed = Turbofile.from_mapping(profile)
    rung = parsed.rungs[0]
    assert rung.main_context == 204_800
    assert rung.auxiliary_context == 262_144


def test_equal_context_manual_profile_stays_byte_compatible() -> None:
    from turbofit_runtime.hardware import AcceleratorDevice, HardwareFingerprint
    from turbofit_runtime.manual_profiles import build_manual_profile_payload
    from turbofit_runtime.recipes import ResolvedComponent, ResolvedRecipe

    recipe = ResolvedRecipe(
        row_id="t", profile_name="t", main_alias="m", aux_alias="a",
        aux_mode="dedicated",
        components=(
            ResolvedComponent("aux", "a", "a", "process", "baseline", "1", 11610, ("/bin/true",)),
            ResolvedComponent("main", "m", "m", "process", "baseline", "0", 11605, ("/bin/true",)),
        ),
        main_context=65_536,
        aux_context=65_536,
    )
    hardware = HardwareFingerprint(
        "linux", "x86_64", 262_144,
        devices=(AcceleratorDevice(0, "g0", "RTX", "nvidia", "cuda", 24_576, "8.6", "01"),),
    )
    entry = {
        "context": 65_536,
        "production_recipe_sha256": "sha256:" + "a" * 64,
        "metrics": {"gpu_peak_mb": {"0": 12_000}},
    }
    profile, resolutions, _ = build_manual_profile_payload(
        profile_id="t", profile_entry=entry, recipe=recipe, hardware=hardware
    )

    rung = profile["rungs"][0]
    assert "main_context" not in rung and "auxiliary_context" not in rung