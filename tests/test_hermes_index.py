from turbofit_runtime.hermes_index import build_table, index_row, render_markdown


def rec(cid, intel, tps, suites=("deep-swe", "agentic-pair"), ts="2026-11-14T00:00:00Z"):
    return {"configuration_id": cid, "main": cid.split("--")[0], "hardware_tier_gb": 48,
            "context": 65536, "auxiliary": "maple-preview-tq2", "intelligence_score": intel,
            "throughput_tps": tps, "recorded_at": ts, "benchmark_level": "screening", "suites": list(suites)}


def test_seconds_per_task_replaces_dollars():
    row = index_row(rec("a", 46.9, 39.39))
    assert row["seconds_per_task_at_1k"] == round(1 / 39.39, 4)
    assert index_row(rec("b", 10.0, 0.0))["seconds_per_task_at_1k"] is None


def test_suite_coverage_is_reported_not_assumed():
    row = index_row(rec("a", 46.9, 20.0))
    assert row["suites_measured"] == 2 and row["suites_total"] == 6


def test_table_sorts_by_index_and_renders():
    table = build_table([rec("slow", 30.0, 5.0), rec("fast", 70.0, 40.0)])
    assert [r["configuration"] for r in table["rows"]] == ["fast", "slow"]
    md = render_markdown(table)
    assert "| fast | 48 GB | 65536 |" in md and "2/6" in md


def test_empty_is_safe():
    assert "No measured configurations" in render_markdown(build_table([]))


def test_newest_record_wins():
    table = build_table([rec("x", 10.0, 1.0, ts="2026-01-01T00:00:00Z"),
                         rec("x", 80.0, 9.0, ts="2026-02-01T00:00:00Z")])
    assert len(table["rows"]) == 1 and table["rows"][0]["index"] == 80.0
