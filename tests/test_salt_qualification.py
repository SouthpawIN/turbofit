"""Tests for the SALT-core qualification gate and generated integration spec.

Determinism contract:
* gate criteria live only in references/engine-serve-matrix.json (gate block);
* host work (make, live HTTP) is injected via runner/probe callables in tests;
* the generated doc is a pure function of (matrix, committed gate report);
* the committed doc equals the render of the committed references.
"""
from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turbofit_runtime.engine_check import set_engine_status
from turbofit_runtime.leaderboard import load_engine_matrix
from turbofit_runtime.salt_qualification import (
    ENGINE,
    GENERATED_BEGIN,
    GENERATED_END,
    GATE_REPORT_SCHEMA,
    load_gate,
    check_source_provenance,
    check_host_build,
    check_smoke_suite,
    check_server_contract,
    check_inference,
    run_salt_gate,
    render_gate_report,
    render_integration_doc,
    publish_integration_doc,
)

MATRIX = ROOT / "references" / "engine-serve-matrix.json"
REPORT = ROOT / "references" / "salt-core-gate-report.json"
DOC = ROOT / "references" / "salt-core-integration.md"


def _matrix() -> dict:
    return load_engine_matrix(MATRIX)


# ---------------------------------------------------------------------------
# Gate contract (matrix is the single source of truth)
# ---------------------------------------------------------------------------


def test_gate_loads_machine_readable_contract():
    gate = load_gate(_matrix())
    assert gate["kind"] == "inference-qualification"
    assert gate["repo"]["url"] == "https://github.com/SouthpawIN/SALT-core"
    assert len(gate["repo"]["pinned_commit"]) == 40
    assert gate["repo"]["manifest"] == "CORE-FILES.sha256"
    assert [c["id"] for c in gate["checks"]] == ["G1", "G2", "G3", "G4", "G5"]
    crit = gate["pass_criteria"]
    assert crit["sustained_tps_min"] == 4
    assert crit["expected_band_tps"] == [4, 8]
    assert crit["context_tokens"] == 262144
    assert crit["hallucination_delta_max_pct"] == 2
    assert crit["launch"] == "single-process"


def test_gate_criteria_match_the_entry_note():
    """The machine-readable criteria must not drift from the human note."""
    entry = _matrix()["engines"][ENGINE]
    note = entry["note"]
    crit = entry["gate"]["pass_criteria"]
    assert f"{crit['sustained_tps_min']}-8 tok/s sustained" in note
    assert "262K context" in note
    assert f"<{crit['hallucination_delta_max_pct']}% hallucination delta" in note
    assert f"{crit['launch']} launch" in note


def test_committed_entry_stays_candidate_until_gate_passes():
    entry = _matrix()["engines"][ENGINE]
    assert entry["status"] == "candidate"
    assert entry["upstream"]["pinned_commit"] == entry["gate"]["repo"]["pinned_commit"]


def test_load_gate_rejects_malformed_blocks():
    matrix = _matrix()
    entry = matrix["engines"][ENGINE]
    saved = entry["gate"]
    try:
        entry["gate"] = {}
        try:
            load_gate(matrix)
            assert False, "empty gate block must raise"
        except ValueError:
            pass
        crit = dict(saved["pass_criteria"])
        del crit["context_tokens"]
        entry["gate"] = {**saved, "pass_criteria": crit}
        try:
            load_gate(matrix)
            assert False, "missing criterion must raise"
        except ValueError:
            pass
    finally:
        entry["gate"] = saved


# ---------------------------------------------------------------------------
# Fixture repo
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, tamper: str | None = None, extra_manifest_line: str | None = None):
    repo = tmp_path / "SALT-core"
    (repo / "server").mkdir(parents=True)
    orig_a, orig_b = b"alpha", b"beta"
    a = b"tampered" if tamper == "a.txt" else orig_a
    btxt = orig_b
    (repo / "a.txt").write_bytes(a)
    (repo / "b.txt").write_bytes(btxt)
    lines = [
        f"{sha256(orig_a).hexdigest()}  a.txt",
        f"{sha256(orig_b).hexdigest()}  b.txt",
    ]
    if extra_manifest_line:
        lines.append(extra_manifest_line)
    (repo / "CORE-FILES.sha256").write_text("\n".join(lines) + "\n")
    (repo / "server" / "serve.py").write_text(
        'if path == "/v1/models":\n    pass\n'
        'if self.path == "/v1/chat/completions":\n    pass\n'
    )
    (repo / "server" / "serve-launch.sh").write_text("#!/bin/sh\n")
    (repo / "salt").write_bytes(b"binary")
    return repo


def _gate():
    return load_gate(_matrix())


# ---------------------------------------------------------------------------
# G1 source provenance
# ---------------------------------------------------------------------------


def test_g1_passes_on_verifiable_tree(tmp_path):
    repo = _make_repo(tmp_path)
    check = check_source_provenance(repo, _gate())
    assert check.status == "pass"
    ev = check.evidence
    assert ev["verified"] == 2
    assert ev["mismatched"] == 0
    assert ev["missing_private_only"] == 0
    assert ev["pinned_commit_present"] is None  # non-git tree: pin not checked


def test_g1_reports_private_only_paths(tmp_path):
    repo = _make_repo(
        tmp_path,
        extra_manifest_line=f"{sha256(b'x').hexdigest()}  private-only/path.c",
    )
    check = check_source_provenance(repo, _gate())
    assert check.status == "pass"
    assert check.evidence["missing_private_only"] == 1
    assert check.evidence["verified"] == 2


def test_g1_detects_tampering(tmp_path):
    repo = _make_repo(tmp_path, tamper="a.txt")
    check = check_source_provenance(repo, _gate())
    assert check.status == "pass"  # partial verification still verifies
    assert check.evidence["mismatched"] == 1


def test_g1_fails_when_nothing_verifies(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "a.txt").write_bytes(b"tampered")
    (repo / "b.txt").write_bytes(b"tampered")
    check = check_source_provenance(repo, _gate())
    assert check.status == "fail"
    assert check.evidence["mismatched"] == 2


def test_g1_fails_without_manifest(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "CORE-FILES.sha256").unlink()
    check = check_source_provenance(repo, _gate())
    assert check.status == "fail"


def test_g1_fails_on_missing_repo():
    check = check_source_provenance(Path("/nonexistent/salt-core"), _gate())
    assert check.status == "fail"


def test_g1_git_pin_present_and_absent(tmp_path):
    repo = _make_repo(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pin"],
        check=True,
    )
    ok = check_source_provenance(repo, _gate(), git_probe=lambda r, c: True)
    assert ok.status == "pass"
    assert ok.evidence["pinned_commit_present"] is True
    bad = check_source_provenance(repo, _gate(), git_probe=lambda r, c: False)
    assert bad.status == "fail"
    assert bad.evidence["pinned_commit_present"] is False


# ---------------------------------------------------------------------------
# G2 / G3 host checks (injected runner — no make in tests)
# ---------------------------------------------------------------------------


def test_g2_g3_pass_with_clean_runner(tmp_path):
    repo = _make_repo(tmp_path)
    runner = lambda cmd, cwd: (0, "ok\n")
    build = check_host_build(repo, runner)
    assert build.status == "pass"
    assert build.evidence["binary_present"] is True
    smoke = check_smoke_suite(repo, runner)
    assert smoke.status == "pass"


def test_g2_fails_on_build_error(tmp_path):
    repo = _make_repo(tmp_path)
    build = check_host_build(repo, lambda cmd, cwd: (2, "make: *** Error"))
    assert build.status == "fail"
    assert build.evidence["exit_code"] == 2


def test_g2_fails_when_binary_missing(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "salt").unlink()
    build = check_host_build(repo, lambda cmd, cwd: (0, "ok\n"))
    assert build.status == "fail"


def test_g3_fails_on_test_failure(tmp_path):
    repo = _make_repo(tmp_path)
    smoke = check_smoke_suite(repo, lambda cmd, cwd: (1, "FAIL\n"))
    assert smoke.status == "fail"


# ---------------------------------------------------------------------------
# G4 server contract
# ---------------------------------------------------------------------------


def test_g4_static_pass_without_live(tmp_path):
    repo = _make_repo(tmp_path)
    check = check_server_contract(repo)
    assert check.status == "pass"
    assert check.evidence["routes_found"] == ["/v1/models", "/v1/chat/completions"]
    assert check.evidence["live"] is None


def test_g4_fails_when_routes_missing(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "server" / "serve.py").write_text("no routes here\n")
    check = check_server_contract(repo)
    assert check.status == "fail"
    assert check.evidence["routes_missing"] == ["/v1/models", "/v1/chat/completions"]


def test_g4_fails_without_serve_py(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "server" / "serve.py").unlink()
    check = check_server_contract(repo)
    assert check.status == "fail"


def test_g4_live_probe_pass_and_block(tmp_path):
    repo = _make_repo(tmp_path)
    good = check_server_contract(
        repo, endpoint="http://127.0.0.1:8084",
        live_probe=lambda e: {"ok": True, "status": 200, "models": 1},
    )
    assert good.status == "pass"
    assert good.evidence["live"]["ok"] is True
    bad = check_server_contract(
        repo, endpoint="http://127.0.0.1:8084",
        live_probe=lambda e: {"ok": False, "error": "connection refused"},
    )
    assert bad.status == "blocked"
    assert bad.evidence["live"]["error"] == "connection refused"


# ---------------------------------------------------------------------------
# G5 inference qualification
# ---------------------------------------------------------------------------


def test_g5_blocked_without_live_report():
    check = check_inference(_gate(), None)
    assert check.status == "blocked"
    assert check.evidence["criteria"]["context_tokens"] == 262144


def test_g5_pass_on_qualifying_run():
    check = check_inference(
        _gate(),
        {
            "sustained_tps": 5.2,
            "context_tokens": 262144,
            "hallucination_delta_pct": 1.5,
            "launch": "single-process",
        },
    )
    assert check.status == "pass"
    assert check.evidence["failures"] == []


def test_g5_fails_below_tps_floor():
    check = check_inference(
        _gate(),
        {
            "sustained_tps": 3.9,
            "context_tokens": 262144,
            "hallucination_delta_pct": 1.5,
            "launch": "single-process",
        },
    )
    assert check.status == "fail"
    assert any("sustained_tps" in f for f in check.evidence["failures"])


def test_g5_fails_on_hallucination_delta():
    check = check_inference(
        _gate(),
        {
            "sustained_tps": 5.0,
            "context_tokens": 262144,
            "hallucination_delta_pct": 2.4,
            "launch": "single-process",
        },
    )
    assert check.status == "fail"
    assert any("hallucination" in f for f in check.evidence["failures"])


def test_g5_fails_on_wrong_launch():
    check = check_inference(
        _gate(),
        {"sustained_tps": 5.0, "context_tokens": 262144, "launch": "multi-process"},
    )
    assert check.status == "fail"
    assert any("launch" in f for f in check.evidence["failures"])


# ---------------------------------------------------------------------------
# Full gate run + verdict
# ---------------------------------------------------------------------------


def test_verdict_candidate_until_inference_proven(tmp_path):
    repo = _make_repo(tmp_path)
    report = run_salt_gate(
        repo,
        _matrix(),
        runner=lambda cmd, cwd: (0, "ok\n"),
    )
    assert report["schema"] == GATE_REPORT_SCHEMA
    assert report["engine"] == ENGINE
    assert report["verdict"] == "candidate"
    assert report["unmet"] == ["G5"]
    by_id = {c["id"]: c for c in report["checks"]}
    assert by_id["G1"]["status"] == "pass"
    assert by_id["G2"]["status"] == "pass"
    assert by_id["G3"]["status"] == "pass"
    assert by_id["G4"]["status"] == "pass"
    assert by_id["G5"]["status"] == "blocked"


def test_verdict_pass_when_all_checks_green(tmp_path):
    repo = _make_repo(tmp_path)
    report = run_salt_gate(
        repo,
        _matrix(),
        runner=lambda cmd, cwd: (0, "ok\n"),
        live_report={
            "sustained_tps": 5.2,
            "context_tokens": 262144,
            "hallucination_delta_pct": 1.5,
            "launch": "single-process",
        },
    )
    assert report["verdict"] == "pass"
    assert report["unmet"] == []


def test_host_checks_can_be_skipped(tmp_path):
    repo = _make_repo(tmp_path)
    report = run_salt_gate(repo, _matrix(), run_host_checks=False)
    by_id = {c["id"]: c for c in report["checks"]}
    assert by_id["G2"]["status"] == "skipped"
    assert by_id["G3"]["status"] == "skipped"
    assert report["verdict"] == "candidate"
    assert "G2" in report["unmet"] and "G3" in report["unmet"]


def test_report_render_is_deterministic(tmp_path):
    repo = _make_repo(tmp_path)
    a = run_salt_gate(repo, _matrix(), runner=lambda cmd, cwd: (0, "ok\n"))
    b = run_salt_gate(repo, _matrix(), runner=lambda cmd, cwd: (0, "ok\n"))
    a.pop("repo")
    b.pop("repo")
    assert render_gate_report(a) == render_gate_report(b)


def test_promotion_is_data_only_and_preserves_gate(tmp_path):
    target = tmp_path / "matrix.json"
    target.write_text(MATRIX.read_text(encoding="utf-8"))
    set_engine_status(target, ENGINE, "released")
    reloaded = load_engine_matrix(target)
    entry = reloaded["engines"][ENGINE]
    assert entry["status"] == "released"
    assert entry["gate"]["pass_criteria"]["context_tokens"] == 262144
    assert entry["upstream"]["pinned_commit"] == entry["gate"]["repo"]["pinned_commit"]


# ---------------------------------------------------------------------------
# Generated integration spec
# ---------------------------------------------------------------------------


def test_doc_is_deterministic_and_marked():
    md = render_integration_doc(_matrix())
    assert md.startswith(GENERATED_BEGIN)
    assert md.rstrip().endswith(GENERATED_END)
    assert md == render_integration_doc(_matrix())


def test_doc_includes_gate_contract_from_matrix():
    md = render_integration_doc(_matrix())
    assert "4 GB VRAM" in md
    assert "| G5 | inference-qualification | live |" in md
    assert "262144-token context" in md
    assert "zenodo.org/records/23111764" in md
    assert "ce0363260a488c96b951b4d826a188e1014575d4" in md


def test_doc_renders_committed_report_section():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    md = render_integration_doc(_matrix(), report)
    assert report["verdict"] in md
    assert "Latest gate report" in md
    assert "Promotion path" in md


def test_committed_doc_matches_committed_references():
    text = DOC.read_text(encoding="utf-8")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert text == render_integration_doc(_matrix(), report)


def test_committed_report_is_a_valid_gate_run():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema"] == GATE_REPORT_SCHEMA
    assert report["engine"] == ENGINE
    by_id = {c["id"]: c for c in report["checks"]}
    assert set(by_id) == {"G1", "G2", "G3", "G4", "G5"}
    assert by_id["G1"]["status"] == "pass"
    assert by_id["G2"]["status"] == "pass"
    assert by_id["G3"]["status"] == "pass"
    assert by_id["G4"]["status"] == "pass"
    assert by_id["G5"]["status"] == "blocked"
    assert report["verdict"] == "candidate"
    assert report["unmet"] == ["G5"]
    # committed gate criteria equal the committed matrix criteria
    assert report["gate"] == _matrix()["engines"][ENGINE]["gate"]


def test_publish_is_idempotent(tmp_path):
    target1 = tmp_path / "a.md"
    target2 = tmp_path / "b.md"
    publish_integration_doc(MATRIX, REPORT, target1)
    publish_integration_doc(MATRIX, REPORT, target2)
    assert target1.read_text() == target2.read_text()
    assert target1.read_text().startswith(GENERATED_BEGIN)


def test_publish_without_report(tmp_path):
    target = tmp_path / "c.md"
    publish_integration_doc(MATRIX, None, target)
    text = target.read_text()
    assert "No gate run recorded in this tree" in text
    assert text.startswith(GENERATED_BEGIN)
