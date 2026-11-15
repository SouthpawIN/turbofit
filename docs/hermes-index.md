# The Hermes Index, run locally

Nous Research scores a model with the **Hermes Index**: the mean pass@1 across four
suites, each run inside an agent harness, reported together with the cost of a
task. TurboFit adopts that definition literally and runs the same tests against a
model it serves itself, so a local model can carry a comparable number instead of
a vendor estimate.

| # | Suite | Harness | Pinned version | Dataset pin | Tasks |
|---|---|---|---|---|---|
| 1 | Hermes Bench | Nous-internal | — | — | unavailable |
| 2 | TerminalBench 4 | [Harbor](https://www.harborframework.com/) | `harbor==0.26.0` | `terminal-bench/terminal-bench@4.1.0` | 66 |
| 3 | TerminalBench Science | Harbor | `harbor==0.26.0` | `terminal-bench-science/terminal-bench-science@0.1.0` | 70 |
| 4 | SkillsBench | [benchflow](https://github.com/benchflow-ai/skillsbench) | `benchflow==0.6.3` | `skillsbench@1.1` (`b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af`) | 87 |

`benchflow==0.6.3` is not a guess: `skillsbench@1.1` refuses to score under
`0.6.2` because that newer revision declares `>=0.6.3,<0.7`. The runner pins the
version the dataset was validated against, and re-verifies all 87 task digests at
resolve time.

## Score definition

```
index = 100 × mean(pass@1 over the Hermes suites actually measured)
suites = measured / 4
```

* Equal weight per measured suite, matching Nous's arithmetic mean.
* No other suite contributes. `deep-swe` and `agentic-pair` remain harnesses but
  are **retired as scores**; a record that carries either name is rejected.
* A suite that cannot run is `unavailable` with a reason — never `0`.
* A trial without a verifier reward is an **infrastructure failure**: the suite
  fails and its evidence file says `status: invalid`. A model is never blamed for
  an unreachable endpoint.
* Local inference has no dollar cost, so the cost axis is **seconds per
  1,000-token task** (`1000 ÷ tok/s`), measured by a real completion.

## Hermes Bench is not fabricable

Hermes Bench is run by Nous Research's own harness and reported through the Nous
portal. No public runner, task set or grader is published, so TurboFit records it
as:

```json
{"suite": "hermes-bench", "status": "unavailable",
 "reason": "nous-internal: Hermes Bench is run by Nous Research's own harness ..."}
```

A local Hermes Index is therefore **3/4 suites** at best, and every row says so.
The fix is a published harness or a portal-issued result for the exact
configuration; until then no number is invented for it.

## Commands

```bash
# What can run here, and what cannot
scripts/turbofit-hermes-suite availability

# Install the pinned harness and fetch its tasks
scripts/turbofit-hermes-suite prepare --suite terminal-bench-4
scripts/turbofit-hermes-suite prepare --suite skills-bench

# Run a small, explicit subset against a model TurboFit is serving
scripts/turbofit-hermes-suite run \
  --suite terminal-bench-4 \
  --model qwen3-8-27b-gsq-rco-iq3-s \
  --base-url http://127.0.0.1:8092/v1 \
  --include terminal-bench/bun-sourcemap-leak \
  --include terminal-bench/batched-eval-parity \
  --include terminal-bench/vllm-deepseek-streaming \
  --n-tasks 3 --jobs-dir evidence/tb4

# Score the finished job: pass@1 + SHA-256-bound evidence
scripts/turbofit-hermes-suite summarize \
  --suite terminal-bench-4 --jobs-dir evidence/tb4 \
  --configuration-id qwen3-8-27b-gsq-rco-iq3-s--none--64k \
  --production-recipe-sha256 sha256:... --output evidence/tb4.json

# Emit the Hermes-Index block for a PR body
scripts/turbofit-hermes-suite block \
  --evidence evidence/tb4.json --model qwen3-8-27b-gsq-rco-iq3-s \
  --configuration-id qwen3-8-27b-gsq-rco-iq3-s--none--64k \
  --tokens-per-second 21.5
```

The campaign drives exactly these commands per configuration; the screening level
runs **one task per suite**, because a real single-task run beats a fabricated
full-suite claim.

### Serving context bounds the agent

TerminalBench agents accumulate transcript until they submit. With Harbor's
`mini-swe-agent` there is no working step-limit kwarg (`step_limit` lives in the
agent's own config, which Harbor does not expose), so a trial ends when the agent
submits, hits the task's wall clock, or exceeds the **served context window**.
Serve at least 64K tokens, and report the context with the score: a 32K
deployment will spend its budget on transcript and score zero for a reason that
belongs to the deployment, not the model. The published row carries `context` for
exactly this reason.

## Local endpoint plumbing

The harness starts a fresh Docker bridge network per trial, so `127.0.0.1` inside
the agent is the agent itself. Runners translate host loopback to the host's
primary route address (the same rule `scripts/turbofit-deepswe` uses) and pass
that host to Harbor's `--allow-agent-host`. A deny-incoming host firewall needs a
narrow `INPUT -i br+ -p tcp --dport <port> -j ACCEPT` rule while the campaign
runs; the campaign already owns and removes that rule for its gateway port.

## Evidence

Each suite writes `turbofit.hermes-suite-evidence/v1`:

```json
{
  "suite": {"name": "terminal-bench-4", "revision": "terminal-bench/terminal-bench@4.1.0",
            "harness": "harbor", "harness_version": "0.26.0"},
  "summary": {"tasks_total": 3, "tasks_passed": 1, "score": 0.333,
              "infrastructure_failures": 0, "seconds_per_task": 245.6,
              "tokens_per_second": 4.07},
  "trials": [{"task": "terminal-bench/bun-sourcemap-leak", "reward": 1.0,
              "passed": true, "sha256": "sha256:..."}],
  "evidence_sha256": "sha256:..."
}
```

`evidence_sha256` is the SHA-256 of the canonical JSON with that field removed, so
anyone can re-read the file and confirm it was not edited after the fact. The
configuration record carries that hash as the measurement's
`raw_result_sha256`, which is what validation checks before an index exists.
