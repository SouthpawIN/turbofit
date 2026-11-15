"""Lock the two-list Fit List invariants.

The user's requirement: the GPU (Dedicated) list and the unified (CPU) list are
two SEPARATE ladders, kept clean by memory model — dense stays on Dedicated
(real GPU can run dense), the Unified list uses MoE stand-ins (dense 27B
crawls on CPU). Both converge on Qwen 3.8 Flash Next at 64+ GB.

This test is the regression guard against the "mixed list" bug where dense and
MoE rungs were interleaved in one ladder.
"""
from __future__ import annotations

from turbofit_runtime import fit_check
from turbofit_runtime.fit_ladders import dedicated_ladder, unified_ladder

FLASH_NEXT = "Qwen 3.8 Flash Next"


def _models(ladder) -> list:
    return [r.model for r in ladder]


def test_dedicated_list_is_dense_until_flash_next_convergence() -> None:
    d = dedicated_ladder()
    # every rung except the final one is a dense model
    assert all(r.class_ == "dense" for r in d[:-1])
    # and it converges on Flash Next (MoE) at the top
    assert d[-1].model == FLASH_NEXT
    assert d[-1].class_ == "moe"
    # dense order: Bonsai 2 -> Underdog -> GSQ-RCO 27B
    assert _models(d)[:4] == ["Bonsai 2", "Bonsai 2", "Underdog 27B", "Qwen 3.8 GSQ-RCO 27B"]


def test_unified_list_is_moe_only_never_dense() -> None:
    u = unified_ladder()
    # no dense model anywhere on the CPU list
    assert all(r.class_ == "moe" for r in u), f"dense leaked onto unified: {_models(u)}"
    # MoE stand-ins: Maple -> Cyber-Tiel (x4 quant rungs) -> Flash Next
    assert u[0].model == "Maple Preview"
    assert all(r.model == "Cyber-Tiel Coder" for r in u[1:5])
    assert u[-1].model == FLASH_NEXT


def test_both_lists_converge_on_flash_next_at_64plus() -> None:
    d = dedicated_ladder()
    u = unified_ladder()
    assert d[-1].band.startswith("64")
    assert u[-1].band.startswith("64")
    assert d[-1].model == u[-1].model == FLASH_NEXT
    assert d[-1].engine_line() == u[-1].engine_line()  # both -> strata


def test_every_rung_carries_a_per_model_engine_priority() -> None:
    for r in dedicated_ladder() + unified_ladder():
        assert r.engines, f"{r.band} {r.model} has no engine priority"
        # priority is an ordered list of engine ids
        assert all(isinstance(e, str) and e for e in r.engines)


def test_unified_stand_in_correspondence() -> None:
    u = unified_ladder()
    assert "Bonsai 2" in u[0].note and "Underdog" in u[0].note   # Maple stands in for low dense
    assert "GSQ-RCO" in u[1].note                                # Cyber-Tiel stands in for mid dense


def test_dedicated_dense_rung_engine_priorities() -> None:
    d = dedicated_ladder()
    # ternary-dense rungs (Bonsai 2 x2, Underdog) are served by prism-llama.cpp
    for r in d[:3]:
        assert r.engines == ["prism-llama.cpp"], f"{r.band} {r.model}: {r.engines}"
    # the dense GSQ-RCO 27B rungs are turbohaul-manager-led (CUDA), with CUDA engines
    for r in d[3:5]:
        assert r.engines[0] == "turbohaul-manager"
        assert all(e in ("llama.cpp", "vllm", "sglang") for e in r.engines[1:])


def test_candidate_engines_are_carried_on_rungs_separately() -> None:
    # low-VRAM dense candidates come from the matrix, never mixed into priority
    for r in dedicated_ladder():
        is_27b_dense = r.model in ("Bonsai 2", "Underdog 27B", "Qwen 3.8 GSQ-RCO 27B")
        expected = ["salt-core", "saliency"] if is_27b_dense else []
        assert r.low_vram_candidates == expected, f"{r.band} {r.model}: {r.low_vram_candidates}"
        assert not (set(r.low_vram_candidates) & set(r.engines))
    for r in unified_ladder():
        assert r.low_vram_candidates == [], f"{r.band} {r.model}: {r.low_vram_candidates}"


def test_source_tiers_have_separate_dense_and_moe_lists() -> None:
    # the raw tier data itself is already split (this is what the publisher reads)
    assert all(t.model_tokens[0] in ("bonsai-2", "underdog", "gsq-rco", "strata")
               for t in fit_check.DEDICATED_TIERS)
    assert all(t.model_tokens[0] in ("maple", "cyber-tiel", "strata")
               for t in fit_check.SHARED_TIERS)
