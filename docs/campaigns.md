# Benchmark and intelligence campaigns

Operator detail for physical and intelligence campaigns.

## Benchmark campaign

Run or resume the complete campaign:

```bash
# Install/verify the three pinned native runtimes used by the exact artifacts
scripts/install-native-runtimes
scripts/install-native-runtimes --check-only

PYTHONPATH=src:. scripts/turbofit-catalog-campaign run
```

Mainline `llama.cpp` serves standard GGUFs, the pinned PrismML fork serves Bonsai/Ternary custom Q1/Q2 and DSpark artifacts, and pinned `ik_llama.cpp` serves GLM 5.2 IQ2_KL with DSA, IndexShare, CPU-MoE, and native MTP. Runtime revisions and binary paths are canonical in [`references/native-runtimes.json`](references/native-runtimes.json). Validate each artifact with its required parser using `scripts/verify-gguf-artifacts`; parsing all artifacts with mainline `llama.cpp` is intentionally invalid because the custom quantization formats require their matching runtimes.

Inspect progress:

```bash
PYTHONPATH=src:. scripts/turbofit-catalog-campaign status
```

Status separates `current_recipe` coverage—whose `pending + resolved + deferred` always equals all 1,620 active rows—from `historical_attempts`, which may contain obsolete runtime failures or successes and is never release eligibility. Deferred rows remain explicit and are never counted as resolved.

The campaign is resumable and records failed attempts rather than silently dropping them. Runtime failures preserve command lines, tracebacks, component/gateway logs, telemetry, and hashes in a unique immutable directory under `references/results/catalog-campaign/failures/<row>/<timestamp>/`. State records pin the canonical production-recipe and validation-protocol SHA-256; changing a runtime, artifact, offload policy, command, smoke-request length, or shared-route scheduling automatically requeues stale successes and resets the attempt budget. Each row acquires an exclusive production-service lease before the first GPU-clear gate: Turbofit's controller/gateway are paused, benchmark components run without a port/GPU race, post-run GPU clear is verified, and only services that were previously active are restored. Every successful row requires:

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

Raw evidence and resumable state live under [`references/results/catalog-campaign/`](references/results/catalog-campaign/). Every current-recipe attempt writes to a unique immutable attempt directory and binds its exact OS/architecture, host RAM, accelerator UUIDs, PCI topology, per-device memory, compute capability, driver revision, topology key, and raw-result SHA-256. Changing this physical-evidence protocol invalidates prior recipe success instead of silently blessing evidence that lacks the required identity. Failed rows remain unresolved and are retried with bounded exponential backoff; an arbitrary attempt count never converts failure into completion. The only terminal physical-fit classification is `classify-hardware-incompatible`, which requires current-recipe, current-fingerprint, checksum-bound failure evidence and a concrete required-memory value greater than available physical memory. Curated physical winners live in [`references/hardware-tier-tournaments.json`](references/hardware-tier-tournaments.json). Dashboard and Desktop render the same evidence rather than maintaining a second result source.

### Measured intelligence campaign

Runtime fit and decode speed do **not** imply intelligence. Turbofit therefore runs a second durable campaign against each exact production configuration after its native runtime row passes:

```bash
# Initialize or inspect all 1,620 configurations × three benchmark levels
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

Every run launches the same quantized main/auxiliary recipe used in production and pins its canonical recipe SHA-256. It then executes:

1. **DeepSWE**, pinned to `datacurve-ai/deep-swe@435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, through PIER `0.3.0` and the main production route;
2. **Turbofit Agentic Production Pair v1**, where the auxiliary route performs schema-bound tool selection and the main route synthesizes the final answer from deterministic tool results.

Benchmark levels are deliberately explicit:

| Level | DeepSWE | Agentic pair |
|---|---:|---:|
| screening | 3 tasks × 1 sample | 8 cases × 1 |
| promotion | 30 tasks × 3 samples | 8 cases × 3 |
| release | 113 tasks × 3 samples | 8 cases × 3 |

The intelligence score is `100 × geometric_mean(DeepSWE resolved rate, agentic decision accuracy)` with equal weights. Geometric aggregation prevents a model that fails one domain from hiding behind strength in the other. Tokens/second remains a separate measured axis. Balanced ranking is the harmonic mean of intelligence and speed normalized to a 50 tok/s target.

No score is emitted unless both harnesses complete and immutable raw evidence, suite revisions, exact quantizations, context, and production-recipe hash are present. Intelligence records use resolved runtime aliases for `main` and `auxiliary`; shared-main auxiliary identity is `auto:<main-alias>`, with raw catalog identities retained separately. DeepSWE additionally requires physical model calls and agent steps for every PIER trial; container/network failures are infrastructure-invalid results, never model scores. The temporary production gateway binds to the container-reachable host route only during the benchmark. On deny-incoming hosts, the intelligence runner owns a narrowly scoped temporary `INPUT -i br+ -p tcp --dport 18092 -j ACCEPT` rule and removes it in cleanup; it never opens the port persistently or to non-Docker interfaces. PIER receives both `OPENAI_BASE_URL` and `OPENAI_API_BASE` through explicit `--agent-env` values because PIER `--env-file` changes host-side resolution but does not inject provider routing into the mini-swe-agent container. DeepSWE runner protocol v3 also pins LiteLLM to zero retries and a 300-second request timeout: deterministic context-limit or request failures terminate the trial and remain genuine measured model failures instead of wedging the campaign in exponential retries. The initial loopback translation uses the host's active non-loopback route rather than assuming Docker's `172.17.0.1`. PIER creates per-trial Compose bridges with different gateways, so Turbofit's custom mini-swe-agent adapter discovers `/proc/net/route` inside each trial and rewrites the runtime provider URL to that exact bridge gateway before the first model call. Every intelligence attempt has a unique immutable directory containing its runtime logs, DeepSWE jobs, normalized summaries, agentic evidence, aggregate, recipe, and terminal success/failure record. Missing results are displayed as **pending**, never as zero or as a catalog-derived proxy. Evidence lives under [`references/results/intelligence-campaign/`](references/results/intelligence-campaign/) and the recommendation index is [`references/intelligence-scores.json`](references/intelligence-scores.json).

Show every hardware tier with storage, host-memory status, aggregate/per-device accelerator requirements, topology, quantization/offload mode, physical-fit evidence, intelligence, and TPS:

```bash
PYTHONPATH=src:. scripts/turbofit-hardware-tiers
/turbofit tiers
```

Live serving TPS measured on this dual RTX 3090 (`2x24`) host: [`docs/hardware-tier-tps.md`](docs/hardware-tier-tps.md). Those numbers are not 8 GB / 16 GB card proof.

---
