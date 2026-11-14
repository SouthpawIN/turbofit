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
