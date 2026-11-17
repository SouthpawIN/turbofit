"""Strata tier placement, preference and capability honesty.

The rules under test are the operator's, not the implementation's:

* a Strata row is admitted at a lower tier only when the CPU can decode it
  (AVX2 per Strata's docs) and the tier holds its required context;
* at equal tier and equal context an admitted Strata MoE outranks the dense
  27B, and the reason says which engine and which context;
* a host without AVX2 is told *which* flag is missing, never ranked away;
* the 64K floor is a hard gate: no 32K substitute, and a tile with nothing at
  64K says so.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from turbofit_runtime import strata_tiers
from turbofit_runtime.allowed_lineup import check_local_options, strata_options
from turbofit_runtime.context_floor import MIN_CONTEXT_TOKENS
from turbofit_runtime.cpu_features import STRATA_REQUIRED

GIB = 1024 ** 3
AVX2 = frozenset({"fpu", "avx", "avx2", "sse4_2"})
AVX512 = frozenset({*STRATA_REQUIRED, "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"})


def rows():
    return strata_tiers.load_rows()


def test_floor_is_64k() -> None:
    assert MIN_CONTEXT_TOKENS == 65_536


def test_strata_row_is_admitted_at_a_lower_tier_when_capable() -> None:
    """Flash Next holds 128K on an 8 GB card with 64 GiB RAM (AVX2 host)."""
    view = strata_tiers.tier_view(tier_gb=8, ram_bytes=64 * GIB, cpu_flags=AVX2)
    assert view.capability.supported is True
    assert view.admitted, view.reason
    top = view.admitted[0]
    assert top.engine == "strata"
    assert top.model_id.startswith("qwen3-8-flash-next")
    assert top.context == 131_072
    assert top.caveats and "runs slowly" in top.caveats[0]
    assert top.reason.startswith("strata:")
    assert "128K" in top.reason


def test_strata_row_is_admitted_at_24gb_at_256k() -> None:
    view = strata_tiers.tier_view(tier_gb=24, ram_bytes=64 * GIB, cpu_flags=AVX2)
    top = view.admitted[0]
    assert top.context == 262_144
    assert view.preferred_over_llamacpp is True


def test_a_row_that_cannot_hold_its_required_context_is_not_admitted() -> None:
    """A row whose documented context is 32K is refused at every tier."""
    row = replace(rows()[0], context_curve=((8, 32_768),))
    result = strata_tiers.place_row(row, tier_gb=24, ram_bytes=128 * GIB)
    assert isinstance(result, strata_tiers.StrataRefusal)
    assert "below the 65,536 token (64K) floor" in result.reason


def test_row_below_its_vram_floor_is_refused_with_the_documented_reason() -> None:
    view = strata_tiers.tier_view(tier_gb=4, ram_bytes=64 * GIB, cpu_flags=AVX2)
    assert view.admitted == ()
    assert "smallest documented floor is 8 GB" in view.reason
    assert "runs slowly" in view.reason


def test_avx2_absent_excludes_with_the_missing_flag_named() -> None:
    state = strata_tiers.capability(frozenset({"fpu", "avx"}))
    assert state.supported is False
    assert state.missing_flags == ("avx2",)
    assert "lacks avx2" in state.reason
    view = strata_tiers.tier_view(tier_gb=24, ram_bytes=64 * GIB, cpu_flags=frozenset({"fpu"}))
    assert view.admitted == ()
    assert "lacks avx2" in view.reason


def test_avx512_only_annotates_never_excludes() -> None:
    plain = strata_tiers.capability(AVX2)
    fast = strata_tiers.capability(AVX512)
    assert plain.supported is True and fast.supported is True
    assert plain.accelerated is False and fast.accelerated is True
    assert "AVX-512 absent" in plain.accelerator_note
    assert "faster prefill" in fast.accelerator_note
    assert strata_tiers.tier_view(
        tier_gb=24, ram_bytes=64 * GIB, cpu_flags=AVX2,
    ).admitted


def test_capability_unknown_is_not_a_refusal() -> None:
    state = strata_tiers.capability(None)
    assert state.supported is None
    view = strata_tiers.tier_view(tier_gb=24, ram_bytes=64 * GIB, cpu_flags=None)
    assert view.admitted == ()
    assert "not probed" in view.reason


def test_strata_moe_outranks_the_dense_27b_at_the_same_tier() -> None:
    options = check_local_options(vram_gb=48, host_ram_gb=64, cpu_flags=AVX2)
    mains = [item for item in options if item.get("role") == "main"]
    assert mains[0]["engine"] == "strata"
    assert str(mains[0]["alias"]).startswith("qwen3-8-flash-next")
    assert int(mains[0]["context"]) >= MIN_CONTEXT_TOKENS
    dense = [item for item in mains if item["engine"] == "llama.cpp"]
    assert dense, "the llama.cpp options must stay listed behind the Strata pick"


def test_recommendation_reason_names_the_engine_and_the_context() -> None:
    preferred, view = strata_options(vram_gb=24, host_ram_gb=64, cpu_flags=AVX2)
    assert view is not None and view.preferred_over_llamacpp
    why = str(preferred[0]["why"])
    assert why.startswith("strata:")
    assert "256K" in why
    assert "llama.cpp" in why
    assert "Qwen 3.8 Flash Next" in why


def test_strata_does_not_lead_when_llamacpp_holds_more_context() -> None:
    """At 8 GB a 35B-A3B on llama.cpp holds 256K; Strata's 128K is not preferred."""
    options = check_local_options(vram_gb=8, host_ram_gb=64, cpu_flags=AVX2)
    mains = [item for item in options if item.get("role") == "main"]
    assert mains[0]["engine"] == "llama.cpp"
    _preferred, view = strata_options(vram_gb=8, host_ram_gb=64, cpu_flags=AVX2)
    assert view is not None
    assert view.preferred_over_llamacpp is False
    assert "NOT preferred" in view.reason


def test_every_strata_family_is_covered_by_a_rule() -> None:
    """A Strata recipe without a placement rule fails loudly, not silently."""
    recipes = strata_tiers._strata_recipes(strata_tiers.default_root())
    covered = {rule.recipe_alias for rule in strata_tiers.FAMILY_RULES}
    assert set(recipes) <= covered, sorted(set(recipes) - covered)


def test_placement_table_leaves_unsupported_tiers_visible() -> None:
    """A tier with nothing at 64K stays in the table, with its reason."""
    table = {view.tier_gb: view for view in strata_tiers.placement_table(ram_gib=16)}
    assert 8 in table and 96 in table
    # 16 GiB RAM cannot hold the 64 GiB documented floor for Flash Next.
    assert table[8].admitted == ()
    assert "RAM" in table[8].reason or "documented" in table[8].reason
