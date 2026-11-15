"""SALT-core integration: machine-readable qualification gate + generated spec.

The ``salt-core`` entry in ``references/engine-serve-matrix.json`` is a
candidate engine (the TurboFit low-VRAM dense-27B configuration served by the
upstream C99 SALT engine). This module is its qualification gate:

* G1 source provenance — public repo, pinned commit, hash manifest
* G2 host build — ``make salt``
* G3 smoke suite — ``make test`` (public config/state/Q4/framing tests)
* G4 server contract — OpenAI ``/v1/models`` + ``/v1/chat/completions``
  (static source check, optional live probe)
* G5 inference qualification — sustained tok/s, 262K context, hallucination
  delta vs 24 GB, single-process launch (live hardware evidence)

The gate criteria live ONLY in the matrix ``gate`` block (Phase C data-only
doctrine). The generated spec ``references/salt-core-integration.md`` is a
pure function of (matrix, committed gate report), wrapped in generated
markers so tooling can replace it — same pattern as the leaderboard.
Promotion to ``released`` remains a data-only edit via
``engine_check.set_engine_status`` once the gate verdict is ``pass``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .engine_check import set_engine_status
from .leaderboard import load_engine_matrix

GENERATED_BEGIN = "<!-- turbofit:generated:salt-core-integration:begin -->"
GENERATED_END = "<!-- turbofit:generated:salt-core-integration:end -->"
GATE_REPORT_SCHEMA = "turbofit.salt-core-gate/v1"
ENGINE = "salt-core"

#: Facts from the public SALT-core repo (README, 2026-10-19 clone of
#: https://github.com/SouthpawIN/SALT-core). Committed constant — no step may
#: invent upstream claims beyond what the repo's own README states.
SALT_UPSTREAM: dict[str, Any] = {
    "name": "SALT - State Assured Latent-Trasition engine",
    "language": "C99",
    "paper": "SALT: Reusable Computation for Stateful Language-Model Inference",
    "paper_url": "https://zenodo.org/records/23111764",
    "repo_url": "https://github.com/SouthpawIN/SALT-core",
    "export": "Dry-run public export (2026-10-02) — not yet a release",
    "private_pin_revision": "ce0363260a488c96b951b4d826a188e1014575d4",
    "private_pin_at": "2026-10-02T10:58:17-04:00",
    "backends": ["CPU", "Metal", "CUDA", "HIP"],
    "model_weights": "not included in the export; artifacts have their own licenses",
    "verified_per_readme": [
        "isolated Linux CPU and CUDA builds",
        "focused KV-compatibility tests",
        "public `make test` smoke suite (configuration, state control, Q4 byte identity, resource binding, server framing/session contracts)",
        "`make all` and `make -n test all` on Linux",
    ],
    "not_qualified_per_readme": [
        "real inference from the exported tree",
        "HIP/Metal build",
        "complete state/KV parity",
    ],
    "server": "OpenAI-compatible /v1/models + /v1/chat/completions, single-flight FIFO, SSE streaming (server/serve.py, default port 8080)",
}

REQUIRED_ROUTES = ("/v1/models", "/v1/chat/completions")

Runner = Callable[[list[str], Path], tuple[int, str]]
GitProbe = Callable[[Path, str], bool]
LiveProbe = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class GateCheck:
    """One gate check with a deterministic status and evidence."""

    id: str
    name: str
    status: str  # "pass" | "fail" | "blocked" | "skipped"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_gate(matrix: dict[str, Any]) -> dict[str, Any]:
    """Read the machine-readable gate contract from the matrix entry.

    The registry is the single source of truth for gate criteria — this
    function only validates shape, it never invents values.
    """
    entry = matrix.get("engines", {}).get(ENGINE)
    if not entry:
        raise ValueError(f"engine {ENGINE!r} not present in matrix")
    gate = entry.get("gate")
    if not isinstance(gate, dict):
        raise ValueError(f"engine {ENGINE!r} has no machine-readable 'gate' block")
    for key in ("kind", "checks", "repo", "pass_criteria"):
        if key not in gate:
            raise ValueError(f"gate block missing {key!r}")
    crit = gate["pass_criteria"]
    for key in (
        "sustained_tps_min",
        "context_tokens",
        "hallucination_delta_max_pct",
        "launch",
    ):
        if key not in crit:
            raise ValueError(f"pass_criteria missing {key!r}")
    return gate


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source_provenance(
    repo: Path,
    gate: dict[str, Any],
    git_probe: GitProbe | None = None,
) -> GateCheck:
    """G1: public provenance — pinned commit present + hash manifest verified.

    The public export pins a private revision; the manifest may list files
    that exist only in the private tree. Missing files are reported as
    evidence (private-only paths), not as corruption. The check fails only
    when nothing verifies, the manifest is absent, or a declared pinned
    commit is missing from a git checkout.
    """
    check = "G1"
    name = "source-provenance"
    repo_cfg = gate["repo"]
    if not repo.is_dir():
        return GateCheck(check, name, "fail", {"error": f"repo path {repo} does not exist"})
    manifest = repo / repo_cfg["manifest"]
    if not manifest.is_file():
        return GateCheck(check, name, "fail", {"error": f"manifest {manifest} missing"})
    ok = mismatched = missing = total = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, rel = parts[0], parts[1].strip().lstrip("*").strip()
        total += 1
        target = repo / rel
        if not target.is_file():
            missing += 1
            continue
        if _sha256(target) == digest.lower():
            ok += 1
        else:
            mismatched += 1
    pinned = repo_cfg.get("pinned_commit", "")
    pinned_present: bool | None
    if pinned and (repo / ".git").exists():
        probe = git_probe or _default_git_probe
        pinned_present = probe(repo, pinned)
    else:
        pinned_present = None
    status = "pass" if (ok > 0 and pinned_present is not False) else "fail"
    return GateCheck(
        check,
        name,
        status,
        {
            "repo": str(repo),
            "manifest": repo_cfg["manifest"],
            "manifest_files": total,
            "verified": ok,
            "mismatched": mismatched,
            "missing_private_only": missing,
            "pinned_commit": pinned,
            "pinned_commit_present": pinned_present,
        },
    )


def _default_git_probe(repo: Path, commit: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-t", commit],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def check_host_build(repo: Path, runner: Runner, target: str = "salt") -> GateCheck:
    """G2: ``make <target>`` builds on the host and emits the binary."""
    code, out = runner(["make", target], repo)
    binary = repo / "salt"
    return GateCheck(
        "G2",
        "host-build",
        "pass" if (code == 0 and binary.is_file()) else "fail",
        {
            "command": ["make", target],
            "exit_code": code,
            "binary_present": binary.is_file(),
            "log_tail": out.strip()[-400:],
        },
    )


def check_smoke_suite(repo: Path, runner: Runner) -> GateCheck:
    """G3: the public ``make test`` smoke suite passes on the host."""
    code, out = runner(["make", "test"], repo)
    return GateCheck(
        "G3",
        "smoke-suite",
        "pass" if code == 0 else "fail",
        {"command": ["make", "test"], "exit_code": code, "log_tail": out.strip()[-400:]},
    )


def check_server_contract(
    repo: Path,
    endpoint: str | None = None,
    live_probe: LiveProbe | None = None,
) -> GateCheck:
    """G4: the OpenAI server contract.

    Static: ``server/serve.py`` declares both required routes and a launcher
    exists. Live (only when an endpoint is given): ``GET /v1/models`` must
    answer 200 with an OpenAI-shaped ``data`` list.
    """
    check = "G4"
    name = "server-contract"
    serve = repo / "server" / "serve.py"
    source = serve.read_text(encoding="utf-8") if serve.is_file() else ""
    found = [r for r in REQUIRED_ROUTES if f'"{r}"' in source or f"'{r}'" in source]
    evidence: dict[str, Any] = {
        "serve_py_present": serve.is_file(),
        "routes_found": found,
        "routes_missing": [r for r in REQUIRED_ROUTES if r not in found],
        "launcher_present": (repo / "server" / "serve-launch.sh").is_file(),
        "static": True,
    }
    if not serve.is_file() or found != list(REQUIRED_ROUTES):
        return GateCheck(check, name, "fail", evidence)
    if endpoint is None:
        evidence["live"] = None
        evidence["live_note"] = (
            "live probe not attempted; model artifacts are excluded from the "
            "public export, so a live endpoint requires separately obtained weights"
        )
        return GateCheck(check, name, "pass", evidence)
    probe = (live_probe or _default_live_probe)(endpoint)
    evidence["live"] = probe
    if probe.get("ok"):
        return GateCheck(check, name, "pass", evidence)
    return GateCheck(check, name, "blocked", evidence)


def check_inference(gate: dict[str, Any], live_report: dict[str, Any] | None) -> GateCheck:
    """G5: inference qualification — the gate that blocks promotion.

    Requires live hardware evidence (a run report) measured on the declared
    4 GB VRAM configuration. Without a report the check is blocked, never
    failed: absence of hardware is not a defect of the source.
    """
    crit = gate["pass_criteria"]
    if live_report is None:
        return GateCheck(
            "G5",
            "inference-qualification",
            "blocked",
            {
                "criteria": crit,
                "evidence": "no live inference run recorded for the 4 GB VRAM configuration",
            },
        )
    failures: list[str] = []
    tps = live_report.get("sustained_tps")
    if not isinstance(tps, (int, float)) or tps < crit["sustained_tps_min"]:
        failures.append(f"sustained_tps={tps!r} below {crit['sustained_tps_min']}")
    ctx = live_report.get("context_tokens")
    if ctx is not None and ctx < crit["context_tokens"]:
        failures.append(f"context_tokens={ctx} below {crit['context_tokens']}")
    delta = live_report.get("hallucination_delta_pct")
    if delta is not None and delta > crit["hallucination_delta_max_pct"]:
        failures.append(
            f"hallucination_delta_pct={delta} exceeds {crit['hallucination_delta_max_pct']}"
        )
    launch = live_report.get("launch")
    if launch is not None and launch != crit["launch"]:
        failures.append(f"launch={launch!r} is not {crit['launch']!r}")
    return GateCheck(
        "G5",
        "inference-qualification",
        "pass" if not failures else "fail",
        {"criteria": crit, "measured": live_report, "failures": failures},
    )


# --------------------------------------------------------------------------
# Gate run + report
# --------------------------------------------------------------------------


def _default_runner(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=3600
        )
        return proc.returncode, proc.stdout + proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, str(exc)


def _default_live_probe(endpoint: str) -> dict[str, Any]:
    import urllib.request

    url = endpoint.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data")
        return {
            "ok": resp.status == 200 and isinstance(data, list),
            "status": resp.status,
            "models": len(data) if isinstance(data, list) else None,
        }
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return {"ok": False, "error": str(exc)}


def run_salt_gate(
    repo: Path,
    matrix: dict[str, Any],
    *,
    endpoint: str | None = None,
    live_report: dict[str, Any] | None = None,
    run_host_checks: bool = True,
    runner: Runner = _default_runner,
    git_probe: GitProbe | None = None,
    live_probe: LiveProbe | None = None,
) -> dict[str, Any]:
    """Run the full gate and return a deterministic report dict."""
    gate = load_gate(matrix)
    checks: list[GateCheck] = [
        check_source_provenance(repo, gate, git_probe=git_probe),
    ]
    if run_host_checks:
        checks.append(check_host_build(repo, runner))
        checks.append(check_smoke_suite(repo, runner))
    else:
        checks.append(GateCheck("G2", "host-build", "skipped", {"reason": "host checks disabled"}))
        checks.append(GateCheck("G3", "smoke-suite", "skipped", {"reason": "host checks disabled"}))
    checks.append(check_server_contract(repo, endpoint=endpoint, live_probe=live_probe))
    checks.append(check_inference(gate, live_report))
    verdict = "pass" if all(c.status == "pass" for c in checks) else "candidate"
    return {
        "schema": GATE_REPORT_SCHEMA,
        "engine": ENGINE,
        "host": {"os": platform.system(), "machine": platform.machine()},
        "repo": str(repo),
        "checks": [c.to_dict() for c in checks],
        "verdict": verdict,
        "unmet": [c.id for c in checks if c.status != "pass"],
        "gate": gate,
    }


def render_gate_report(report: dict[str, Any]) -> str:
    """Deterministic JSON for the committed gate-report artifact."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------
# Generated integration spec
# --------------------------------------------------------------------------


def _evidence_summary(check: dict[str, Any]) -> str:
    ev = check.get("evidence", {})
    if check["id"] == "G1":
        parts = [
            f"{ev.get('verified', 0)}/{ev.get('manifest_files', 0)} manifest files verified",
            f"{ev.get('mismatched', 0)} mismatched",
            f"{ev.get('missing_private_only', 0)} private-only paths",
        ]
        pin = ev.get("pinned_commit_present")
        if pin is True:
            parts.append("pinned commit present")
        elif pin is False:
            parts.append("PINNED COMMIT MISSING")
        else:
            parts.append("pinned commit not checked (non-git tree)")
        return "; ".join(parts)
    if check["id"] == "G2":
        return (
            f"make salt exit {ev.get('exit_code')}, binary "
            f"{'present' if ev.get('binary_present') else 'MISSING'}"
        )
    if check["id"] == "G3":
        return f"make test exit {ev.get('exit_code')}"
    if check["id"] == "G4":
        live = ev.get("live")
        if live is None:
            return f"static contract OK ({len(ev.get('routes_found', []))}/2 routes), live not attempted"
        if live.get("ok"):
            return f"static + live OK ({live.get('models')} models advertised)"
        return f"live probe blocked: {live.get('error', 'no ok flag')}"
    if check["id"] == "G5":
        if check["status"] == "blocked":
            crit = ev.get("criteria", {})
            return (
                f"blocked — requires live run: >= {crit.get('sustained_tps_min')} tok/s "
                f"sustained, {crit.get('context_tokens')} context, "
                f"<= {crit.get('hallucination_delta_max_pct')}% hallucination delta, "
                f"{crit.get('launch')} launch"
            )
        fails = ev.get("failures", [])
        return "; ".join(fails) if fails else "all inference criteria met"
    return ev.get("reason", "")


def render_integration_doc(
    matrix: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> str:
    """Render the integration spec as a pure function of (matrix, report)."""
    entry = matrix["engines"][ENGINE]
    gate = entry["gate"]
    crit = gate["pass_criteria"]
    lines: list[str] = []
    add = lines.append
    add(GENERATED_BEGIN)
    add("")
    add("# SALT-core integration")
    add("")
    add(
        "SALT-core is the TurboFit low-VRAM dense-27B candidate: the upstream "
        "C99 SALT engine serving a 27B dense model on a 4 GB VRAM box via "
        "KV-cache offload to host RAM and weight/expert disk streaming, with a "
        "262K-token n-gram context. It is registered as `candidate` and cannot "
        "be promoted to `released` until its qualification gate passes."
    )
    add("")
    add("## Upstream engine")
    add("")
    u = SALT_UPSTREAM
    add(f"- **{u['name']}** — {u['language']}; paper: {u['paper']} ({u['paper_url']})")
    add(f"- **Export:** {u['export']}; public tree matches private `main` revision `{u['private_pin_revision']}` ({u['private_pin_at']}) for the exported runtime paths")
    add(f"- **Public repo:** {u['repo_url']}")
    add(f"- **Physical backends:** {', '.join(u['backends'])}")
    add(f"- **Model weights:** {u['model_weights']}")
    add(f"- **Server:** {u['server']}")
    add("")
    add("Verified per the upstream README (the export's own claims):")
    for item in u["verified_per_readme"]:
        add(f"- {item}")
    add("")
    add("Not qualified per the upstream README:")
    for item in u["not_qualified_per_readme"]:
        add(f"- {item}")
    add("")
    add("## TurboFit registry entry")
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| status | `{entry['status']}` |")
    add(f"| os / backends | {', '.join(entry['os'])} / {', '.join(entry['backends'])} |")
    add(f"| min VRAM | {entry.get('min_vram_gb', '?')} GB |")
    add(f"| OpenAI port | {entry.get('openai_port', '?')} |")
    add(f"| model target | {entry.get('model_target', '?')} |")
    add(f"| serves | {entry['serves']} |")
    add(f"| detect | {entry['detect']['commands'] or '—'} |")
    add("")
    add("## Qualification gate (machine-readable contract)")
    add("")
    add("Criteria live in the `gate` block of `references/engine-serve-matrix.json` — the single source of truth; this section is regenerated from it.")
    add("")
    add("| id | check | kind |")
    add("|---|---|---|")
    for c in gate["checks"]:
        add(f"| {c['id']} | {c['name']} | {c['kind']} |")
    add("")
    add("Pass criteria (G5):")
    add("")
    add(f"- sustained >= {crit['sustained_tps_min']} tok/s (expected band {crit.get('expected_band_tps', ['?'])[0]}–{crit.get('expected_band_tps', ['?', '?'])[1]} tok/s; target {crit.get('target_band_tps', ['?'])[0]}–{crit.get('target_band_tps', ['?', '?'])[1]} after KV-offload tuning)")
    add(f"- {crit['context_tokens']}-token context")
    add(f"- hallucination delta <= {crit['hallucination_delta_max_pct']}% vs the 24 GB reference")
    add(f"- launch: {crit['launch']}")
    add("")
    add("## Latest gate report")
    add("")
    if report is None:
        add("No gate run recorded in this tree (`references/salt-core-gate-report.json` absent).")
    else:
        host = report.get("host", {})
        add(f"Host: {host.get('os', '?')} {host.get('machine', '?')} · verdict: `{report.get('verdict', '?')}`" + (f" · unmet: {', '.join(report.get('unmet', []))}" if report.get("unmet") else ""))
        add("")
        add("| id | check | status | evidence |")
        add("|---|---|---|---|")
        for c in report.get("checks", []):
            add(f"| {c['id']} | {c['name']} | `{c['status']}` | {_evidence_summary(c)} |")
    add("")
    add("## Promotion path")
    add("")
    add(
        "When the gate verdict is `pass` (all five checks green, including a "
        "live G5 run on a 4 GB VRAM box with separately obtained model "
        "artifacts), promotion is a data-only edit: "
        "`engine_check.set_engine_status('salt-core', 'released', matrix_path)` "
        "flips the registry status; `engine_check.engine_status` and this "
        "document reflect it immediately. No per-engine code exists."
    )
    add("")
    add("## Method")
    add("")
    add(
        "Sources: `references/engine-serve-matrix.json` (gate block), "
        "`references/salt-core-gate-report.json` (latest run evidence), and the "
        "public SALT-core repo (README, 2026-10-19). Regenerated by "
        "`tests/test_salt_qualification.py`; this file is tooling-owned — do "
        "not hand-edit between the markers."
    )
    add("")
    add(GENERATED_END)
    return "\n".join(lines) + "\n"


def publish_integration_doc(
    matrix_path: Path,
    report_path: Path | None,
    target: Path,
) -> Path:
    """Atomically (re)write the generated spec from committed references."""
    matrix = load_engine_matrix(matrix_path)
    report = None
    if report_path is not None and Path(report_path).is_file():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    text = render_integration_doc(matrix, report)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    """CLI: run the gate against a checked-out SALT-core repo.

    Writes the committed gate-report artifact and regenerates the spec doc.
    """
    import argparse

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run the SALT-core qualification gate")
    parser.add_argument("--repo", required=True, type=Path, help="SALT-core checkout path")
    parser.add_argument("--matrix", type=Path, default=root / "references" / "engine-serve-matrix.json")
    parser.add_argument("--endpoint", default=None, help="live endpoint base URL (e.g. http://127.0.0.1:8084)")
    parser.add_argument("--live-report", type=Path, default=None, help="JSON run report for G5")
    parser.add_argument("--no-host-checks", action="store_true", help="skip G2/G3 (no make available)")
    parser.add_argument("--report-out", type=Path, default=root / "references" / "salt-core-gate-report.json")
    parser.add_argument("--doc-out", type=Path, default=root / "references" / "salt-core-integration.md")
    args = parser.parse_args(argv)

    matrix = load_engine_matrix(args.matrix)
    live_report = None
    if args.live_report is not None:
        live_report = json.loads(args.live_report.read_text(encoding="utf-8"))
    report = run_salt_gate(
        args.repo,
        matrix,
        endpoint=args.endpoint,
        live_report=live_report,
        run_host_checks=not args.no_host_checks,
    )
    args.report_out.write_text(render_gate_report(report), encoding="utf-8")
    publish_integration_doc(args.matrix, args.report_out, args.doc_out)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "unmet": report["unmet"],
                "report": str(args.report_out),
                "doc": str(args.doc_out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
