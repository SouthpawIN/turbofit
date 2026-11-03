"""The fallback chain and the three dials it drives."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbofit_runtime.chain import (
    CHAIN_SCHEMA,
    build_dials,
    context_steps,
    dials_for_selection,
    load_chain,
)

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "references" / "fallback-chain.json"


def usage(*free_mb: int, total: int = 24576) -> dict:
    return {
        "gpus": [
            {"gpu": index, "total_mb": total, "used_mb": total - free, "free_mb": free}
            for index, free in enumerate(free_mb)
        ],
        "host": {"system_ram_mb": 64048, "topology_key": f"{len(free_mb)}x{total}mb"},
        "tps": None,
    }


def test_the_shipped_chain_loads_and_sheds_context_before_model_class() -> None:
    chain = load_chain(CHAIN)

    dedicated = chain.levels("dedicated")

    assert chain.schema == CHAIN_SCHEMA
    assert chain.context_floor == 32768
    kinds = [level.kind for level in dedicated]
    assert kinds[:3] == ["model", "context", "context"], "context must be shed before a smaller model"
    assert kinds[-1] == "yield", "releasing residency is the last resort"
    assert [level.context for level in dedicated[:3]] == [262144, 131072, 65536]


def test_context_steps_halve_to_the_floor() -> None:
    assert context_steps(262144) == [262144, 131072, 65536, 32768]
    assert context_steps(32768) == [32768]
    assert context_steps(40000) == [40000, 32768]


def test_a_structural_fault_in_the_chain_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(CHAIN.read_text())
    payload["lanes"]["dedicated"]["levels"][0]["surprise"] = True
    broken = tmp_path / "chain.json"
    broken.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown fields"):
        load_chain(broken)


def test_full_pressure_reaches_a_smaller_level_while_the_top_does_not_fit() -> None:
    chain = load_chain(CHAIN)

    dials = build_dials(chain=chain, usage=usage(6000, 6000), lane="dedicated")

    levels = {entry["id"]: entry for entry in dials["chain"]["levels"]}
    assert levels["d0"]["fits_now"] is False
    assert levels["d1"]["fits_now"] is False, "6 GB/card cannot hold 19 GB either"
    assert dials["memory_pressure"]["state"] == "elevated"
    assert dials["chain"]["active_level"] != "d0"
    assert levels[dials["chain"]["active_level"]]["fits_now"] is True


def test_free_memory_keeps_the_top_level_and_lists_its_options() -> None:
    chain = load_chain(CHAIN)

    dials = build_dials(chain=chain, usage=usage(24000, 24000), lane="dedicated")

    assert dials["chain"]["active_level"] == "d0"
    assert dials["memory_pressure"]["state"] == "nominal"
    ids = {option["model"] for option in dials["options"]}
    assert "deepseek-v4-1-flash" in ids
    assert "muse-glimmer-30b-hermes-agentic" in ids
    for option in dials["options"]:
        assert "path" not in option, "a candidate option must not invent artifact paths"
        assert option["evidence"] in {"measured", "portable-fit", "candidate"}


def test_every_option_is_marked_unproven_until_it_has_on_box_evidence() -> None:
    chain = load_chain(CHAIN)
    candidates = [
        option
        for lane in chain.lane_ids()
        for level in chain.levels(lane)
        for option in level.options
        if option.get("evidence") == "candidate"
    ]

    models = {option["model"] for option in candidates}
    assert {"lfm-24a2b", "groot", "muse-glimmer-30b-hermes-agentic"} <= models
    assert all(option.get("note") for option in candidates), "a candidate must say what it is"


def test_dials_default_to_the_unified_lane_on_shared_memory() -> None:
    dials = dials_for_selection(
        usage={"gpus": [], "host": {"topology_key": "uma", "host_usable_memory_mb": 65536}},
        selection={"profile": "apple-24gb"},
    )

    assert dials["chain"]["lane"] == "unified"


def test_status_snapshot_exposes_the_dials(monkeypatch) -> None:
    import plugin_tools

    snap = plugin_tools.status_snapshot({"providers": {}}, probe=False)

    assert set(snap["dials"]) >= {"memory_pressure", "context", "chain", "options"}
    assert snap["dials"]["chain"]["levels"], "the chain must be visible even with no probe"


def test_an_unprobed_host_reports_unknown_instead_of_nothing_fits() -> None:
    """No measurement must not look like a full card with no room."""
    chain = load_chain(CHAIN)

    dials = build_dials(chain=chain, usage={"gpus": [], "host": {}}, lane="dedicated")

    assert dials["memory_pressure"]["state"] == "unknown"
    assert dials["memory_pressure"]["measured"] is False
    assert dials["memory_pressure"]["headroom_mb"] is None
    assert dials["chain"]["active_level"] is None
    assert all(entry["fits_now"] is None for entry in dials["chain"]["levels"])
    assert dials["context"]["steps"], "the planned context steps are still worth showing"
