"""Real runners and hash-bound evidence for the four Hermes Index suites.

Nous Research scores a model on the mean pass@1 of four suites run inside an
agent harness. TurboFit adopts that definition verbatim, so this module carries
one runner per suite, each pinned by harness version *and* dataset revision:

===============  =========  =====================  ===========================
suite            harness    pinned version         dataset pin
===============  =========  =====================  ===========================
terminal-bench-4 harbor     0.26.0                 terminal-bench@4.1.0
terminal-bench-  harbor     0.26.0                 terminal-bench-science@0.1.0
science
skills-bench     benchflow  0.6.3                  skillsbench@1.1 (b63b7b2)
hermes-bench     —          —                      nous-internal, unavailable
===============  =========  =====================  ===========================

Every runner drives a **local OpenAI-compatible endpoint** (a TurboFit-served
model) instead of a cloud provider. A suite that cannot run on this machine is
recorded as ``unavailable`` with the exact reason and the command that would fix
it — never as a zero, and never as an invented number.

The evidence format is deliberately boring: per-trial rewards read back from the
harness's own ``result.json`` files, summed into pass@1, then SHA-256 bound over
the canonical JSON so a published row can be re-verified byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from .intelligence import (
    HERMES_INDEX_SUITES,
    BenchmarkMeasurement,
    validate_measurement,
)

EVIDENCE_SCHEMA = "turbofit.hermes-suite-evidence/v1"
EVIDENCE_PROTOCOL = "turbofit.hermes-suite-evidence/v1"

# Pinned harness versions. These are the exact versions the runners install and
# the versions whose result format the summarizers parse.
HARBOR_VERSION = "0.26.0"
BENCHFLOW_VERSION = "0.6.3"

# Pinned datasets. Harbor resolves ``name@version`` against its registry;
# benchflow resolves ``skillsbench@1.1`` and verifies all 87 task digests.
TERMINAL_BENCH_DATASET = "terminal-bench/terminal-bench@4.1.0"
TERMINAL_BENCH_SCIENCE_DATASET = "terminal-bench-science/terminal-bench-science@0.1.0"
SKILLSBENCH_DATASET = "skillsbench@1.1"
SKILLSBENCH_REVISION = "b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af"

DEFAULT_ROOT = Path.home() / ".local" / "share" / "turbofit" / "benchmarks"
#: Harbor and benchflow keep separate agent registries; each suite names the
#: harness-native agent that speaks OpenAI-compatible chat completions.
DEFAULT_AGENT = "mini-swe-agent"
SUITE_AGENTS = {
    "terminal-bench-4": "mini-swe-agent",
    "terminal-bench-science": "mini-swe-agent",
    # benchflow 0.6.3 has no mini-swe-agent; opencode is its OpenAI-compatible
    # coding agent.
    "skills-bench": "opencode",
}
DEFAULT_API_FORMAT = "openai_chat_completions"
LOCAL_API_KEY = "turbofit-local"

_NOUS_INTERNAL = (
    "nous-internal: Hermes Bench is run by Nous Research's own harness and is "
    "reported through the Nous portal; no public runner, task set or grader is "
    "published, so it cannot be executed or scored here."
)


@dataclass(frozen=True)
class SuiteRunner:
    """One suite: what to install, what to fetch, and how to read it back."""

    suite: str
    harness: str
    harness_version: str
    dataset: str = ""
    revision: str = ""
    script: str = "turbofit-hermes-suite"
    unavailable: str = ""
    fix: str = ""

    @property
    def runnable(self) -> bool:
        return not self.unavailable


RUNNERS: dict[str, SuiteRunner] = {
    "terminal-bench-4": SuiteRunner(
        suite="terminal-bench-4",
        harness="harbor",
        harness_version=HARBOR_VERSION,
        dataset=TERMINAL_BENCH_DATASET,
        revision=TERMINAL_BENCH_DATASET,
    ),
    "terminal-bench-science": SuiteRunner(
        suite="terminal-bench-science",
        harness="harbor",
        harness_version=HARBOR_VERSION,
        dataset=TERMINAL_BENCH_SCIENCE_DATASET,
        revision=TERMINAL_BENCH_SCIENCE_DATASET,
    ),
    "skills-bench": SuiteRunner(
        suite="skills-bench",
        harness="benchflow",
        harness_version=BENCHFLOW_VERSION,
        dataset=SKILLSBENCH_DATASET,
        revision=SKILLSBENCH_REVISION,
    ),
    "hermes-bench": SuiteRunner(
        suite="hermes-bench",
        harness="none",
        harness_version="",
        unavailable=_NOUS_INTERNAL,
        fix=(
            "Ask Nous Research to publish the Hermes Bench harness, or supply a "
            "portal-issued result for the exact configuration."
        ),
    ),
}


def runner(suite: str) -> SuiteRunner:
    if suite not in RUNNERS:
        raise ValueError(
            f"unknown Hermes Index suite: {suite!r} (expected one of {', '.join(HERMES_INDEX_SUITES)})"
        )
    return RUNNERS[suite]


def availability() -> dict[str, Any]:
    """What can actually run here, and what cannot — with reasons.

    A local index is therefore always reported as ``N of 4``: with the three
    public suites wired, the best honest coverage is 3/4.
    """
    suites = {}
    for name in HERMES_INDEX_SUITES:
        definition = RUNNERS[name]
        suites[name] = {
            "status": "runnable" if definition.runnable else "unavailable",
            "harness": definition.harness or None,
            "harness_version": definition.harness_version or None,
            "dataset": definition.dataset or None,
            "revision": definition.revision or None,
            "reason": definition.unavailable or None,
            "fix": definition.fix or None,
        }
    runnable = [name for name, value in suites.items() if value["status"] == "runnable"]
    return {
        "suites": suites,
        "runnable": runnable,
        "unavailable": [name for name in HERMES_INDEX_SUITES if name not in runnable],
        "max_local_coverage": {"measured": len(runnable), "total": len(HERMES_INDEX_SUITES)},
    }


def require_runnable(suite: str) -> SuiteRunner:
    definition = runner(suite)
    if not definition.runnable:
        raise RuntimeError(f"{suite} is unavailable: {definition.unavailable}")
    return definition


# ---------------------------------------------------------------------------
# Local endpoint plumbing
# ---------------------------------------------------------------------------


def host_route_address() -> str:
    """Host address that any Docker bridge network can route back to.

    A harness starts one bridge network per trial, so ``127.0.0.1`` inside the
    agent container is the container itself. The host's primary route address is
    stable across those networks and is what the DeepSWE runner already uses.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("1.1.1.1", 53))
        address = str(sock.getsockname()[0])
    finally:
        sock.close()
    if not address or address.startswith("127."):
        raise RuntimeError("could not determine a non-loopback host route for sandboxed agents")
    return address


def container_base_url(base_url: str) -> str:
    """Translate host loopback to an address the trial container can reach."""
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return base_url.rstrip("/")
    hostname = os.environ.get("TURBOFIT_DOCKER_HOST_GATEWAY") or host_route_address()
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, hostname + port, parsed.path.rstrip("/"), parsed.query, parsed.fragment))


def _validate_endpoint(model: str, base_url: str, n_tasks: int) -> None:
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if not model or any(character.isspace() for character in model):
        raise ValueError("model must be a non-empty token")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must use http or https")


# ---------------------------------------------------------------------------
# Command construction (pure, unit-testable, no side effects)
# ---------------------------------------------------------------------------


def harbor_command(
    *,
    suite: str,
    model: str,
    base_url: str,
    jobs_dir: Path,
    n_tasks: int,
    include: Sequence[str] = (),
    agent: str | None = None,
    n_concurrent: int = 1,
    api_format: str = DEFAULT_API_FORMAT,
    api_key: str = LOCAL_API_KEY,
    harbor_bin: str = "harbor",
) -> tuple[list[str], dict[str, str]]:
    """``harbor run`` argv for a local OpenAI-compatible endpoint."""
    definition = require_runnable(suite)
    if definition.harness != "harbor":
        raise ValueError(f"{suite} is not a harbor suite")
    _validate_endpoint(model, base_url, n_tasks)
    if n_concurrent <= 0:
        raise ValueError("n_concurrent must be positive")
    agent = agent or SUITE_AGENTS.get(suite, DEFAULT_AGENT)
    endpoint = container_base_url(base_url)
    command = [
        harbor_bin, "run",
        "--dataset", definition.dataset,
        "--env", "docker",
        "--agent", agent,
        "--model", f"openai/{model}",
        "--base-url", endpoint,
        "--api-format", api_format,
        "--api-key-name", "OPENAI_API_KEY",
    ]
    host = urlsplit(endpoint).hostname
    if host and host not in {"127.0.0.1", "localhost"}:
        # The trial network is the harness's own; the endpoint host must be
        # explicitly allowed or agent egress to the model is dropped.
        command.extend(("--allow-agent-host", host))
    for name in include:
        command.extend(("--include-task-name", name))
    command.extend([
        "--n-tasks", str(n_tasks),
        "--n-concurrent", str(n_concurrent),
        "--jobs-dir", str(jobs_dir),
        "--yes",
    ])
    environment = dict(os.environ)
    environment.update({
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": endpoint,
        "OPENAI_API_BASE": endpoint,
    })
    return command, environment


def skillsbench_command(
    *,
    model: str,
    base_url: str,
    jobs_dir: Path,
    tasks_dir: Path | None = None,
    dataset: str = SKILLSBENCH_DATASET,
    include: Sequence[str] = (),
    agent: str | None = None,
    n_concurrent: int = 1,
    api_format: str = DEFAULT_API_FORMAT,
    api_key: str = LOCAL_API_KEY,
    bench_bin: str = "bench",
) -> tuple[list[str], dict[str, str]]:
    """``bench eval create`` argv for a local OpenAI-compatible endpoint.

    Either a pinned registry dataset (digest-verified) or a prepared tasks
    directory is required; SkillsBench never runs against an unpinned tree.
    """
    definition = require_runnable("skills-bench")
    _validate_endpoint(model, base_url, 1)
    if n_concurrent <= 0:
        raise ValueError("n_concurrent must be positive")
    agent = agent or SUITE_AGENTS["skills-bench"]
    endpoint = container_base_url(base_url)
    command = [
        # benchflow 0.6.3 deprecates ``create`` in favour of ``eval run``;
        # ``create`` is what this pinned version still executes.
        bench_bin, "eval", "create",
        "--sandbox", "docker",
        "--agent", agent,
        "--model", model,
        "--concurrency", str(n_concurrent),
        "--jobs-dir", str(jobs_dir),
    ]
    if tasks_dir is not None:
        command.extend(("--tasks-dir", str(tasks_dir)))
    else:
        command.extend(("--dataset", dataset))
    for name in include:
        command.extend(("--include", name))
    command.extend((
        "--agent-env", f"OPENAI_API_KEY={api_key}",
        "--agent-env", f"OPENAI_BASE_URL={endpoint}",
        "--agent-env", f"OPENAI_API_BASE={endpoint}",
        "--agent-env", f"OPENAI_API_FORMAT={api_format}",
    ))
    environment = dict(os.environ)
    environment.update({
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": endpoint,
        "OPENAI_API_BASE": endpoint,
    })
    return command, environment


def install_harness(suite: str, *, dry_run: bool = False) -> list[str]:
    """The exact install command for a suite's pinned harness."""
    definition = require_runnable(suite)
    if definition.harness == "harbor":
        spec = f"harbor=={definition.harness_version}"
    elif definition.harness == "benchflow":
        spec = f"benchflow=={definition.harness_version}"
    else:  # pragma: no cover - hermes-bench raises above
        raise RuntimeError(f"{suite} has no installable harness")
    if shutil.which("uv"):
        command = ["uv", "tool", "install", "--force", spec]
    else:
        command = [sys.executable or "python3", "-m", "pip", "install", "--user", spec]
    if not dry_run:
        subprocess.run(command, check=True)
    return command


# ---------------------------------------------------------------------------
# Throughput (the "cost per task" axis for local inference)
# ---------------------------------------------------------------------------


def probe_throughput(*, base_url: str, model: str, max_tokens: int = 128, timeout: float = 120.0) -> float:
    """Decode tokens/second from a real completion, not a catalog estimate.

    llama.cpp reports ``timings.predicted_per_second`` on non-streaming
    responses, which excludes prompt processing and is the number a throughput
    column should carry. Other OpenAI-compatible servers fall back to
    completion_tokens / wall time.
    """
    import urllib.request

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LOCAL_API_KEY}"},
    )
    started = datetime.now(timezone.utc)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    timings = body.get("timings") if isinstance(body, dict) else None
    if isinstance(timings, Mapping):
        speed = timings.get("predicted_per_second")
        if isinstance(speed, (int, float)) and not isinstance(speed, bool) and speed > 0:
            return round(float(speed), 3)
    usage = body.get("usage") if isinstance(body, dict) else None
    completion = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    if isinstance(completion, int) and completion > 0 and elapsed > 0:
        return round(completion / elapsed, 3)
    raise RuntimeError("endpoint returned no measurable decode throughput")


# ---------------------------------------------------------------------------
# Evidence parsing
# ---------------------------------------------------------------------------

_PASS_REWARD = 1.0


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _trial_payloads(jobs_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every per-trial ``result.json`` under a job tree.

    Two harnesses write this tree. Harbor records the verifier verdict under
    ``verifier_result.rewards.reward``; benchflow (the SkillsBench harness)
    records the same verdict directly under ``rewards.reward``. Both write a
    job-level ``result.json`` beside the trials, which has no ``task_name`` and
    is skipped rather than counted as a task.
    """
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(Path(jobs_dir).rglob("result.json")):
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if "task_name" not in payload:
            continue
        if "verifier_result" not in payload and "rewards" not in payload:
            continue
        found.append((path, payload))
    return found


def _reward(payload: Mapping[str, Any]) -> float | None:
    """The verifier's reward for one trial, in either harness's layout."""
    candidates: list[Any] = []
    verifier = payload.get("verifier_result")
    if isinstance(verifier, Mapping):
        rewards = verifier.get("rewards")
        if isinstance(rewards, Mapping):
            candidates.append(rewards.get("reward"))
    rewards = payload.get("rewards")
    if isinstance(rewards, Mapping):
        candidates.append(rewards.get("reward"))
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        return float(value)
    return None


def _trim(value: Any, limit: int) -> Any:
    """Keep harness failures readable without pasting 200 KB of traceback."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [{len(value) - limit} chars truncated]"
    if isinstance(value, Mapping):
        return {key: _trim(item, limit) for key, item in value.items()}
    return value


def _seconds(payload: Mapping[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(str(payload["started_at"]).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(payload["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    # Harbor stamps UTC; benchflow stamps naive local time. Only the delta
    # matters, so a naive/aware mismatch is normalised instead of discarded.
    if (started.tzinfo is None) != (finished.tzinfo is None):
        started, finished = started.replace(tzinfo=None), finished.replace(tzinfo=None)
    delta = (finished - started).total_seconds()
    return round(delta, 3) if delta >= 0 else None


def summarize_job(
    *,
    suite: str,
    jobs_dir: Path,
    configuration_id: str,
    production_recipe_sha256: str,
    output: Path | None = None,
) -> dict[str, Any]:
    """Turn a finished harness job into pass@1 evidence.

    A trial counts as passed only when the harness's verifier rewarded it 1.0 and
    the trial raised no exception. A trial with no verifier reward is recorded as
    an infrastructure failure and makes the whole run invalid — an unreachable
    endpoint must never be reported as a model that scored zero.
    """
    definition = require_runnable(suite)
    if not production_recipe_sha256.startswith("sha256:") or len(production_recipe_sha256) != 71:
        raise ValueError("production recipe must be SHA-256 bound")
    trials: list[dict[str, Any]] = []
    for path, payload in _trial_payloads(jobs_dir):
        raw = path.read_bytes()
        reward = _reward(payload)
        agent_result = payload.get("agent_result") or {}
        input_tokens = int(agent_result.get("n_input_tokens") or 0) if isinstance(agent_result, Mapping) else 0
        output_tokens = int(agent_result.get("n_output_tokens") or 0) if isinstance(agent_result, Mapping) else 0
        exception = _trim(payload.get("exception_info") or payload.get("error"), 2000)
        valid = reward is not None
        trials.append({
            "path": str(path),
            "sha256": _sha256_bytes(raw),
            "task": str(payload.get("task_name") or ""),
            "trial": str(payload.get("trial_name") or path.parent.name),
            "reward": reward,
            "passed": bool(valid and reward == _PASS_REWARD and exception is None),
            "execution_valid": valid,
            "seconds": _seconds(payload),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "agent": (payload.get("agent_info") or {}).get("name") if isinstance(payload.get("agent_info"), Mapping) else None,
            "exception": exception,
        })
    if not trials:
        raise RuntimeError(f"no trial results found under {jobs_dir}")
    total = len(trials)
    passed = sum(item["passed"] for item in trials)
    invalid = sum(not item["execution_valid"] for item in trials)
    seconds = [item["seconds"] for item in trials if item["seconds"]]
    output_tokens = sum(item["output_tokens"] for item in trials)
    wall_seconds = round(sum(seconds), 3) if seconds else None
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "protocol": EVIDENCE_PROTOCOL,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "suite": {
            "name": suite,
            "revision": definition.revision,
            "harness": definition.harness,
            "harness_version": definition.harness_version,
            "dataset": definition.dataset,
        },
        "status": "invalid" if invalid else "measured",
        "configuration_id": configuration_id,
        "production_recipe_sha256": production_recipe_sha256,
        "jobs_dir": str(jobs_dir),
        "summary": {
            "tasks_total": total,
            "tasks_passed": passed,
            "score": passed / total,
            "infrastructure_failures": invalid,
            "wall_seconds": wall_seconds,
            "seconds_per_task": round(wall_seconds / total, 3) if wall_seconds is not None else None,
            "output_tokens": output_tokens,
            "tokens_per_second": round(output_tokens / wall_seconds, 3) if wall_seconds else None,
            "agents": sorted({str(item["agent"]) for item in trials if item["agent"]}),
            "tasks": sorted({item["task"] for item in trials}),
        },
        "trials": trials,
    }
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = _sha256_bytes(canonical)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if invalid:
        raise RuntimeError(
            f"{suite} produced {invalid}/{total} trials without a verifier reward; "
            "the run is invalid, not a zero score"
        )
    return evidence


def measurement_from_evidence(
    evidence: Mapping[str, Any], *, raw_result: str | Path | None = None,
) -> BenchmarkMeasurement:
    """The hash-bound measurement a configuration record carries for one suite."""
    summary = evidence["summary"]
    item = BenchmarkMeasurement(
        name=str(evidence["suite"]["name"]),
        revision=str(evidence["suite"]["revision"]),
        score=float(summary["score"]),
        tasks_total=int(summary["tasks_total"]),
        tasks_passed=int(summary["tasks_passed"]),
        raw_result=str(raw_result or evidence.get("jobs_dir") or ""),
        raw_result_sha256=str(evidence["evidence_sha256"]),
    )
    validate_measurement(item)
    return item


# ---------------------------------------------------------------------------
# Publication block
# ---------------------------------------------------------------------------


def index_block(
    *,
    model: str,
    configuration: str,
    hardware_tier_gb: int,
    context: int,
    measurements: Iterable[Mapping[str, Any] | BenchmarkMeasurement],
    tokens_per_second: float | None,
    availability_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A Hermes-Index-format result block for an upstream PR body.

    ``index`` is the mean pass@1 over the measured Hermes suites, ``suites`` is
    ``N/4`` with every unmeasured suite named, and cost is expressed as seconds
    per task because local inference has no dollar cost.
    """
    report = dict(availability_report or availability())
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in measurements:
        payload = item if isinstance(item, Mapping) else {
            "name": item.name, "score": item.score,
            "tasks_total": item.tasks_total, "tasks_passed": item.tasks_passed,
            "revision": item.revision,
        }
        by_name[str(payload["name"])] = payload
    suites = []
    for name in HERMES_INDEX_SUITES:
        measured = by_name.get(name)
        state = report["suites"][name]
        suites.append({
            "suite": name,
            "status": "measured" if measured else state["status"],
            "pass_at_1": float(measured["score"]) if measured else None,
            "tasks_total": int(measured["tasks_total"]) if measured else 0,
            "tasks_passed": int(measured["tasks_passed"]) if measured else 0,
            "revision": (measured or {}).get("revision") or state.get("revision"),
            "reason": None if measured else state.get("reason"),
        })
    measured_scores = [row["pass_at_1"] for row in suites if row["pass_at_1"] is not None]
    tps = float(tokens_per_second) if tokens_per_second else None
    return {
        "model": model,
        "configuration": configuration,
        "hardware_tier_gb": hardware_tier_gb,
        "context": context,
        "index": round(100.0 * sum(measured_scores) / len(measured_scores), 6) if measured_scores else None,
        "suites_measured": len(measured_scores),
        "suites_total": len(HERMES_INDEX_SUITES),
        "suites": suites,
        "tokens_per_second": tps,
        "seconds_per_task_1k": round(1000.0 / tps, 3) if tps else None,
        "definition": (
            "Hermes Index = mean pass@1 across Hermes Bench, TerminalBench 4, "
            "TerminalBench Science and SkillsBench."
        ),
    }


def render_index_block(block: Mapping[str, Any]) -> str:
    lines = [
        f"### `{block['model']}` — Hermes Index {_fmt(block['index'])} "
        f"({block['suites_measured']}/{block['suites_total']} suites)",
        "",
        f"- configuration: `{block['configuration']}`",
        f"- tier: {block['hardware_tier_gb']} GB · context: {int(block['context']):,}",
        f"- tokens/s: {_fmt(block['tokens_per_second'], 2)} · seconds per 1k-token task: "
        f"{_fmt(block['seconds_per_task_1k'], 3)}",
        "",
        "| Suite | pass@1 | Tasks | Status | Pin |",
        "|---|---|---|---|---|",
    ]
    for row in block["suites"]:
        score = "—" if row["pass_at_1"] is None else f"{100 * row['pass_at_1']:.2f}"
        tasks = f"{row['tasks_passed']}/{row['tasks_total']}" if row["tasks_total"] else "—"
        status = row["status"] if row["status"] == "measured" else f"unavailable ({_short(row.get('reason'))})"
        lines.append(f"| {row['suite']} | {score} | {tasks} | {status} | `{row.get('revision') or '—'}` |")
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}"


def _short(reason: str | None) -> str:
    if not reason:
        return "no reason recorded"
    return reason.split(":")[0].strip()
