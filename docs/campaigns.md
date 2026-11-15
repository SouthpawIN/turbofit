# Benchmark and intelligence campaigns

Operator detail for physical and intelligence campaigns.

## Benchmark campaign

Run or resume the complete campaign:

```bash
# Install/verify the three pinned native runtimes used by the exact artifacts
scripts/install-native-runtimes
scripts/install-native-runtimes --check-only

PYTHONPATH=src:. scripts/turbofit-catalog-campaign run

# After matching baseline and DFlash2 rows succeed, export a comparable,
# self-hashed A/B record from their immutable campaign results.
scripts/turbofit-dflash2-ab-evidence \
  --baseline-row qwen-3-8-27b-q4-k-m-auto-64k \
  --dflash2-row qwen-3-8-27b-q4-k-m-dflash2-auto-64k \
  --output references/results/qwen38-dflash2-2x24-YYYYMMDD.json
```

Mainline `llama.cpp` serves standard GGUFs, the pinned PrismML fork serves Bonsai/Ternary custom Q1/Q2 and DSpark artifacts, and pinned `ik_llama.cpp` serves GLM 5.2 IQ2_KL with DSA, IndexShare, CPU-MoE, and native MTP. Runtime revisions and binary paths are canonical in [`references/native-runtimes.json`](references/native-runtimes.json). Validate each artifact with its required parser using `scripts/verify-gguf-artifacts`; parsing all artifacts with mainline `llama.cpp` is intentionally invalid because the custom quantization formats require their matching runtimes.

Inspect progress:

```bash
PYTHONPATH=src:. scripts/turbofit-catalog-campaign status
```

Status separates `current_recipe` coverage—whose `pending + resolved + deferred + waiting_download + waiting_memory + waiting_runtime` always equals all 290 active rows—from `historical_attempts`, which may contain obsolete runtime failures or successes and is never release eligibility. Deferred rows remain explicit and are never counted as resolved. Each `waiting_*` status names a different prerequisite the operator can act on, and each row carries exactly one verdict:

* `waiting_download` — the engine artifacts are not verifiably installed. Row selection uses the same `inspect_file` definition the dashboard uses (a sub-1 MiB stub, a truncated/failed download, a missing shard or an empty checkpoint directory is not installed), and each row reports the missing byte count instead of being attempted and failed.
* `waiting_memory` — the weights are installed but the offload planner measured them as too large for this host. The row quotes the planner's own arithmetic instead of launching an engine that must OOM.
* `waiting_runtime` — the weights and the host are fine, but the recipe's engine binary is not installed for this family. The row names the binary path and the install hint (`scripts/install.sh`, or the engine path in `references/model-recipes.json`) and is excluded from `pending`/`runnable` until the runtime lands. This is the benchmark-campaign spelling of the intelligence campaign's existing `ready → waiting-runtime → blocked-runtime` vocabulary: a missing prerequisite is not a model failure.

The launch paths keep their own refusals (`ResolvedRecipe.assert_launchable`, `CampaignBackend.start`), so a gate bug cannot turn a waiting row into an OOM or a silent pass. The campaign is resumable and records failed attempts rather than silently dropping them. Runtime failures preserve command lines, tracebacks, component/gateway logs, telemetry, and hashes in a unique immutable directory under `references/results/catalog-campaign/failures/<row>/<timestamp>/`. State records pin the canonical production-recipe and validation-protocol SHA-256; changing a runtime, artifact, offload policy, command, smoke-request length, or shared-route scheduling automatically requeues stale successes and resets the attempt budget. Each row acquires an exclusive production-service lease before the first GPU-clear gate: Turbofit's controller/gateway are paused, benchmark components run without a port/GPU race, post-run GPU clear is verified, and only services that were previously active are restored. Pinned component ports (11605 main, 11610 auxiliary) are reconciled right after the lease and **before** the clear gate, because a residual engine still holds its VRAM: a listener the campaign can prove it owns (its pid is recorded in runtime state, or it runs the same runtime binary and a campaign log for that port exists in the result directory) is terminated and the port reused; a foreign listener is never killed and the component moves to the next free port in 11600–11699 (ephemeral fallback when the window is full), logging the chosen port and the reason on stderr. Commands are rewritten so argv, runtime state and the measurement gateway agree on the chosen port. Every successful row requires:

1. immutable artifact verification;
2. exact launch recipe compilation;
3. requested context verification;
4. main and auxiliary health checks;
5. non-empty output;
6. throughput and peak-memory capture;
7. runtime-string capture;
8. process shutdown;
9. post-run memory-clear verification.

Promotion priority is lexicographic, not a weighted score:

1. strongest intelligence tier;
2. at least 128K context;
3. at least 30 output tokens/second;
4. at least 262K context;
5. at least 50 output tokens/second;
6. 1M context;
7. fastest measured result.

Raw evidence and resumable state live under [`references/results/catalog-campaign/`](references/results/catalog-campaign/). Every current-recipe attempt writes to a unique immutable attempt directory and binds its exact OS/architecture, host RAM, accelerator UUIDs, PCI topology, per-device memory, compute capability, driver revision, topology key, and raw-result SHA-256. Changing this physical-evidence protocol invalidates prior recipe success instead of silently blessing evidence that lacks the required identity. Failed rows remain unresolved and are retried with bounded exponential backoff; an arbitrary attempt count never converts failure into completion. The only terminal physical-fit classification is `classify-hardware-incompatible`, which requires current-recipe, current-fingerprint, checksum-bound failure evidence and a concrete required-memory value greater than available physical memory. Curated physical winners live in [`references/hardware-tier-tournaments.json`](references/hardware-tier-tournaments.json). Each tier's `candidates` list is not a shortlist: it is **every** catalog main whose weights plus a 64K KV cache fit that memory band (weights from `references/model-footprints.json`, KV from the fit-ladder formula, VRAM reporting tolerance from `recommend.py`), each paired with the single auxiliary `maple-preview-tq2`. Rebuild with `scripts/generate-hardware-tier-tournaments`; `winner` stays `null` until a Maple-aux configuration has hash-bound physical evidence for that class, and the leaderboard ranks measured intelligence first, then speed. Dashboard and Desktop render the same evidence rather than maintaining a second result source.

### Measured intelligence campaign

Runtime fit and decode speed do **not** imply intelligence. Turbofit therefore runs a second durable campaign against each exact production configuration after its native runtime row passes:

```bash
# Initialize or inspect all 290 configurations × three benchmark levels
PYTHONPATH=src:. scripts/turbofit-intelligence-campaign init
PYTHONPATH=src:. scripts/turbofit-intelligence-campaign status

# Run the next production configuration
PYTHONPATH=src:. scripts/turbofit-intelligence-campaign run-one

# Run the serialized native-fit + intelligence campaigns continuously
PYTHONPATH=src:. scripts/turbofit-benchmark-orchestrator --catalog-batch 50 --intelligence-batch 1
# Install the reboot-persistent user service (add --start when no campaign is active)
scripts/install-benchmark-campaign-service
systemctl --user status turbofit-benchmark-campaign.service
```

`~/.config/systemd/user/turbofit-benchmark-campaign.service` is the reboot-persistent user service. The orchestrator uses a host lock so runtime-fit and intelligence jobs never compete for the same accelerators. Failed native rows remain explicit diagnostic evidence and mark their promotion/release prerequisites blocked until their root cause changes the production-recipe identity and requeues physical validation.

A physical campaign suspends only the production **controller** that owns local model residency; the lightweight provider gateway stays online. A live PID-bound `turbofit.campaign-lease/v1` marker makes that production gateway refuse all campaign model ports and route `auto`, `active:main`, and `active:aux` only to the explicitly configured API fallback. The isolated temporary measurement gateway sets `TURBOFIT_CAMPAIGN_GATEWAY=true`, so it alone may route the exact benchmark model ports. This prevents user traffic from contaminating physical measurements while preserving Hermes availability. Nous fallback credentials are resolved through Hermes' refresh-aware auth API. If that login is unavailable, required universal-provider requests return an explicit retryable `503` rather than a false `204` success or a connection failure.

Every run launches the same quantized main/auxiliary recipe used in production and pins its canonical recipe SHA-256. It then runs the **Hermes Index** suites — the same four Nous Research averages, each through its own pinned harness against the recipe's local OpenAI-compatible route:

| Suite | Harness | Pin | Tasks |
|---|---|---|---|
| Hermes Bench | Nous-internal | none public | unavailable |
| TerminalBench 4 | `harbor` | 0.26.0 · `terminal-bench/terminal-bench@4.1.0` | 66 |
| TerminalBench Science | `harbor` | 0.26.0 · `terminal-bench-science/terminal-bench-science@0.1.0` | 70 |
| SkillsBench | `benchflow` | 0.6.3 · `skillsbench@1.1` (`b63b7b2`) | 87 |

Hermes Bench is run by Nous and has no public runner, task set or grader, so it is recorded as `unavailable: nous-internal` with the fix that would unlock it. It is never scored as zero, which is why a local index always reports `suites N/4` — 3/4 today.

Benchmark levels are deliberately explicit:

| Level | TB4 | TB Science | SkillsBench |
|---|---:|---:|---:|
| screening | 1 task | 1 task | 1 task |
| promotion | 33 | 35 | 43 |
| release | 66 | 70 | 87 |

`intelligence_score` is the **equal-weight mean pass@1 over the Hermes suites actually measured**, which is exactly Hermes Index's definition. There is one composite, not two: `combined_score` is an alias of it, and `suite_coverage` reports `measured / 4` alongside every record. DeepSWE and the Turbofit Agentic Production Pair remain available as harnesses (`scripts/turbofit-deepswe`) but are **retired as scores** — a record carrying either name is rejected by validation, so no DeepSWE number can become an index.

A suite that runs but produces a trial without a verifier reward is an infrastructure failure, not a model failure: the level fails and the evidence file records `status: invalid` instead of publishing a zero. Tokens/second remains a separate measured axis; the published cost column is seconds per 1,000-token task. Balanced ranking is the harmonic mean of intelligence and speed normalized to a 50 tok/s target.

No score is emitted unless at least one Hermes Index suite completes and immutable raw evidence, suite revisions, exact quantizations, context, and production-recipe hash are present. Intelligence records use resolved runtime aliases for `main` and `auxiliary`; every catalog row resolves its auxiliary to `maple-preview-tq2`. Legacy shared-main profiles keep the `auto:<main-alias>` identity, with raw catalog identities retained separately. Each Hermes suite additionally requires a real verifier reward for every trial; container, network and endpoint failures are infrastructure-invalid results, never model scores. For reference, the retired DeepSWE harness required physical model calls and agent steps for every PIER trial. The temporary production gateway binds to the container-reachable host route only during the benchmark. On deny-incoming hosts, the intelligence runner owns a narrowly scoped temporary `INPUT -i br+ -p tcp --dport 18092 -j ACCEPT` rule and removes it in cleanup; it never opens the port persistently or to non-Docker interfaces. PIER receives both `OPENAI_BASE_URL` and `OPENAI_API_BASE` through explicit `--agent-env` values because PIER `--env-file` changes host-side resolution but does not inject provider routing into the mini-swe-agent container. DeepSWE runner protocol v3 also pins LiteLLM to zero retries and a 300-second request timeout: deterministic context-limit or request failures terminate the trial and remain genuine measured model failures instead of wedging the campaign in exponential retries. The initial loopback translation uses the host's active non-loopback route rather than assuming Docker's `172.17.0.1`. PIER creates per-trial Compose bridges with different gateways, so Turbofit's custom mini-swe-agent adapter discovers `/proc/net/route` inside each trial and rewrites the runtime provider URL to that exact bridge gateway before the first model call. Every intelligence attempt has a unique immutable directory containing its runtime logs, DeepSWE jobs, normalized summaries, agentic evidence, aggregate, recipe, and terminal success/failure record. Missing results are displayed as **pending**, never as zero or as a catalog-derived proxy. Evidence lives under [`references/results/intelligence-campaign/`](references/results/intelligence-campaign/) and the recommendation index is [`references/intelligence-scores.json`](references/intelligence-scores.json).

Show every hardware tier with storage, host-memory status, aggregate/per-device accelerator requirements, topology, quantization/offload mode, physical-fit evidence, intelligence, and TPS:

```bash
PYTHONPATH=src:. scripts/turbofit-hardware-tiers
/turbofit tiers
```

Live serving TPS measured on this dual RTX 3090 (`2x24`) host: [`docs/hardware-tier-tps.md`](docs/hardware-tier-tps.md). Those numbers are not 8 GB / 16 GB card proof.

---
