from turbofit_runtime.intel_leaderboard import build_leaderboard, latest_per_configuration, render_markdown


def _rec(cid, tier, intel, tps, ts="2026-11-14T00:00:00Z", main=None):
    return {"configuration_id": cid, "hardware_tier_gb": tier, "intelligence_score": intel,
            "throughput_tps": tps, "recorded_at": ts, "main": main or cid, "context": 65536,
            "benchmark_level": "screening", "auxiliary": "maple"}


def test_ranks_intelligence_first_then_speed():
    board = build_leaderboard([_rec("slow-smart", 24, 60.0, 5.0), _rec("fast-dumb", 24, 40.0, 80.0)])
    tier = board["tiers"][0]
    assert tier["recommended"]["configuration_id"] == "slow-smart"
    assert [o["configuration_id"] for o in tier["options"]] == ["fast-dumb"]


def test_speed_breaks_ties():
    board = build_leaderboard([_rec("a", 24, 50.0, 10.0), _rec("b", 24, 50.0, 20.0)])
    assert board["tiers"][0]["recommended"]["configuration_id"] == "b"


def test_newest_measurement_wins():
    recs = latest_per_configuration([_rec("x", 24, 10.0, 1.0, "2026-01-01T00:00:00Z"),
                                    _rec("x", 24, 70.0, 9.0, "2026-02-01T00:00:00Z")])
    assert len(recs) == 1 and recs[0]["intelligence_score"] == 70.0


def test_one_winner_per_tier_and_empty_is_safe():
    board = build_leaderboard([_rec("lo", 16, 30.0, 4.0), _rec("hi", 48, 55.0, 7.0)])
    assert [t["hardware_tier_gb"] for t in board["tiers"]] == [16, 48]
    assert build_leaderboard([])["tiers"] == []
    assert "No measured configurations" in render_markdown(build_leaderboard([]))


CATALOG = {
    "models": [
        {"id": "q5-main", "quantization": "Q5_K_M"},
        {"id": "q8-main", "quantization": "Q8_0"},
        {"id": "iq3-main", "quantization": "GSQ-RCO-IQ3_S"},
    ]
}


def test_q5_measurement_is_never_the_recommendation():
    board = build_leaderboard(
        [
            _rec("q5-row", 48, 70.0, 40.0, main="q5-main"),
            _rec("iq3-row", 48, 55.0, 38.0, main="iq3-main"),
        ],
        CATALOG,
    )
    tier = board["tiers"][0]

    assert tier["recommended"]["configuration_id"] == "iq3-row"
    assert tier["recommended"]["quant_class"] == "allowed"
    assert tier["quant_skipped"][0]["configuration_id"] == "q5-row"
    assert "Q5_K_M" in tier["quant_skipped"][0]["reason"]
    # The measurement is still visible, flagged as not-default.
    assert any(option.get("not_default") for option in tier["options"])
    assert "not default" in render_markdown(board)


def test_q8_measurement_is_recommended_with_a_visible_reason():
    board = build_leaderboard(
        [
            _rec("q8-row", 96, 70.0, 20.0, main="q8-main"),
            _rec("iq3-row", 96, 55.0, 38.0, main="iq3-main"),
        ],
        CATALOG,
    )
    recommended = board["tiers"][0]["recommended"]

    assert recommended["configuration_id"] == "q8-row"
    assert recommended["quant_class"] == "allowed-exception"
    assert recommended["quant_exception_reason"].startswith("fits-fully-in-vram")


def test_tier_with_only_excluded_measurements_has_no_recommendation():
    board = build_leaderboard([_rec("q5-row", 48, 70.0, 40.0, main="q5-main")], CATALOG)
    tier = board["tiers"][0]

    assert tier["recommended"] is None
    assert "quantization policy" in tier["reason"]
    assert "No recommendation" in render_markdown(board)
