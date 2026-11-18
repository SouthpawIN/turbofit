"""Tests for the machine-agnostic Strata split optimiser."""

from __future__ import annotations

import pytest

from turbofit_runtime.strata_planner import (
    GIB,
    FAMILY_ROWS,
    MachineBudget,
    ModelRow,
    Placement,
    Refusal,
    best_placement,
    enumerate_placements,
    recommend_for_machine,
)

#: Every flag the engine's own ``strata --help`` lists.  A planner that emits
#: anything outside this set is launching a command the engine will reject.
VALID_STRATA_FLAGS = frozenset(
    {
        "--max-context",
        "--expert-cache",
        "--resident-cpu-experts",
        "--resident-experts",
        "--mmap-experts",
        "--kv",
        "--kv-resident",
        "--ple-io",
        "--prefill",
        "--spec",
    }
)
VALID_KV_MODES = {"fp16", "int8", "q4_0", "k8v4"}
VALID_PLE_IO = {"direct", "mmap", "ram"}

FLASH_NEXT = next(r for r in FAMILY_ROWS if r.model_id == "qwen3-8-flash-next-iq3xxs")
MAYA_L = next(r for r in FAMILY_ROWS if r.model_id == "glm-5-3-flash-maya-l")
DARWIN = next(r for r in FAMILY_ROWS if r.model_id == "darwin-180b-rsi-strata")


def _assert_flags_valid(placement: Placement) -> None:
    flags = placement.flags
    assert flags, "a placement must carry the engine flags that realise it"
    unknown = {flag for flag in flags if flag.startswith("--")} - VALID_STRATA_FLAGS
    assert not unknown, f"invented strata flags: {sorted(unknown)}"
    # Parse: value flags consume the next token, boolean switches do not.
    switches = {"--resident-cpu-experts", "--resident-experts", "--mmap-experts"}
    values: dict[str, str] = {}
    index = 0
    while index < len(flags):
        flag = flags[index]
        if flag in switches:
            index += 1
            continue
        assert index + 1 < len(flags), f"{flag} has no value in {flags}"
        values[flag] = flags[index + 1]
        index += 2

    assert values["--kv"] in VALID_KV_MODES
    assert values["--ple-io"] in VALID_PLE_IO
    assert int(values["--expert-cache"]) >= 0
    assert int(values["--max-context"]) == placement.context
    if "--kv-resident" in values:
        assert 0 <= int(values["--kv-resident"]) <= placement.context
    # Realise-the-placement contract: a cache tier and a stream tier are each
    # reflected in the flags the planner emits.
    assert (placement.expert_cache_slots > 0) == (int(values["--expert-cache"]) > 0)
    assert (placement.ssd_bytes > 0) == ("--mmap-experts" in flags)
    assert (placement.ram_bytes > 0) or ("--resident-cpu-experts" not in flags)


def test_twelve_gigabyte_machine_runs_flash_next_at_128k_by_streaming():
    """Strata's documented floor: 12 GB + 64 GiB holds Flash Next at 128K.

    It cannot all be resident, so the placement must stream part of the pack and
    must still be admitted (docs: "8 GB runs, slowly", 12 GB recommended).
    """
    machine = MachineBudget(vram_bytes=12 * 1024 ** 3, ram_bytes=64 * GIB)
    placements, blocking = enumerate_placements(FLASH_NEXT, machine, 131_072)
    assert placements, blocking
    at_128k = [p for p in placements if p.context == 131_072]
    assert at_128k, "no 128K placement on a 12 GB / 64 GiB host"

    best = best_placement(FLASH_NEXT, machine, 131_072)
    assert isinstance(best, Placement)
    assert best.context == 131_072
    # Dense + KV on the card, experts split between VRAM cache, RAM and disk.
    assert best.vram_share_gib["dense"] > 0
    assert best.expert_cache_slots > 0
    assert best.ram_bytes > 0
    # 42.9 GiB of experts cannot fit 12 GB of VRAM + 64 GiB RAM alongside a
    # 29.5 GiB embedding table, so disk streaming must appear.
    assert best.ssd_bytes > 0
    assert "--mmap-experts" in best.flags
    _assert_flags_valid(best)


def test_forty_eight_gigabyte_machine_holds_every_262k_family_at_262k():
    """48 GB VRAM + 377 GiB RAM: VRAM takes dense + KV + a large expert cache."""
    machine = MachineBudget(vram_bytes=48 * 1024 ** 3, ram_bytes=377 * GIB)
    two_sixty_two_k = [row for row in FAMILY_ROWS if 262_144 in row.contexts()]
    assert two_sixty_two_k, "expected 262K-capable families in the catalog"

    for row in two_sixty_two_k:
        best = best_placement(row, machine, 262_144)
        assert isinstance(best, Placement), f"{row.model_id} refused: {best.reason}"
        assert best.context == 262_144, f"{row.model_id} fell back to {best.context}"
        # Dense weights live on the card, and so does a large expert cache.
        assert best.vram_share_gib["dense"] == pytest.approx(row.dense_bytes / GIB)
        assert best.expert_cache_slots > 1000, row.model_id
        cached = best.expert_cache_slots * row.slot_bytes
        assert cached > 8 * GIB, f"{row.model_id} only cached {cached / GIB:.1f} GiB"
        # 377 GiB is enough to pin every uncached expert, so nothing streams.
        assert best.ssd_bytes == 0, f"{row.model_id} still streams from disk"
        assert "--resident-cpu-experts" in best.flags
        assert best.vram_bytes <= machine.usable_vram_bytes
        assert best.ram_bytes <= machine.usable_ram_bytes
        _assert_flags_valid(best)


def test_model_limited_family_is_capped_by_its_own_context_not_the_machine():
    """GLM 5.3 Flash tops out at 128K in the catalog; 48 GB cannot change that."""
    machine = MachineBudget(vram_bytes=48 * 1024 ** 3, ram_bytes=377 * GIB)
    best = best_placement(MAYA_L, machine, 262_144)
    assert isinstance(best, Placement)
    assert best.context == 131_072
    assert best.fits_preferred_context is False


def test_insufficient_ram_is_refused_and_ram_is_named():
    """A 125 GiB pack with 8 GiB of RAM and no SSD room cannot be placed."""
    machine = MachineBudget(
        vram_bytes=24 * 1024 ** 3,
        ram_bytes=8 * GIB,
        ssd_free_bytes=4 * GIB,
    )
    placements, blocking = enumerate_placements(DARWIN, machine)
    assert placements == ()
    assert blocking
    assert any("RAM" in reason for reason in blocking)

    refused = best_placement(DARWIN, machine)
    assert isinstance(refused, Refusal)
    assert "RAM" in refused.reason
    assert "RAM" in refused.binding


def test_ranking_prefers_more_expert_bytes_in_vram():
    """Two placements of the same row: the deeper VRAM cache must rank first."""
    machine = MachineBudget(vram_bytes=48 * 1024 ** 3, ram_bytes=377 * GIB)
    placements, _ = enumerate_placements(FLASH_NEXT, machine, 262_144)
    assert len(placements) > 1
    top = placements[0]
    same_context = [p for p in placements if p.context == top.context]
    for other in same_context:
        assert top.predicted_speed >= other.predicted_speed
        assert top.expert_cache_slots >= other.expert_cache_slots or (
            top.predicted_speed > other.predicted_speed
        )
    assert top.predicted_speed > 1.0, "a cached placement must beat all-SSD streaming"
    # And the ordering is a real total order on speed within a context.
    speeds = [p.predicted_speed for p in same_context]
    assert speeds == sorted(speeds, reverse=True)


def test_recommend_for_machine_returns_every_family():
    chosen = recommend_for_machine(
        48 * 1024 ** 3, 377 * GIB, None, ["avx2", "avx512f"]
    )
    families = {row.family for row in FAMILY_ROWS}
    assert families <= set(chosen)
    for key, item in chosen.items():
        if isinstance(item, Placement):
            _assert_flags_valid(item)


def test_small_machine_refuses_rather_than_inventing_a_placement():
    """4 GB VRAM cannot hold Flash Next's 5.6 GiB dense part at any context."""
    machine = MachineBudget(vram_bytes=4 * 1024 ** 3, ram_bytes=16 * GIB)
    placements, blocking = enumerate_placements(FLASH_NEXT, machine)
    assert placements == ()
    assert any("VRAM" in reason for reason in blocking)


def test_row_rejects_non_positive_pack():
    with pytest.raises(ValueError):
        ModelRow(
            model_id="x", family="x", quant="q", pack_bytes=0,
            expert_bytes=0, expert_layers=1, dense_bytes=0,
        )
