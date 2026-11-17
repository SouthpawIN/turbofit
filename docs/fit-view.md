# Fit view: what should I run on this machine?

`GET /api/plugins/turbofit/fit` answers the one question a first run needs:
*given this host, what should I run, and what am I missing to run it?*

It joins three sources that were previously only readable side by side:

| Question | Source |
|---|---|
| What was measured best on this tier? | `references/intelligence-leaderboard.json` (`intel_leaderboard`) |
| What does that configuration need? | `references/model-recipes.json` (`recipes`) — the resolved components the launcher execs |
| Is each file installed and complete? | `artifacts.inspect_file` — the campaign's definition, so stubs and truncated downloads are absent |
| Can this CPU execute the engine? | `cpu_features` — Strata needs AVX2 (AVX-512 is only a prefill accelerator) |
| Which auxiliary? | `maple-preview-tq2`, always |

`GET /fit?tier=48` narrows to one tier; a tier with no measurement still
answers, with `status: "no-measurements"` and no invented recommendation.

## Response

```json
{
  "ok": true,
  "schema": "turbofit.fit-view/v1",
  "maple_aux_id": "maple-preview-tq2",
  "cpu": {
    "flags_known": true,
    "strata_supported": false,
    "missing_strata_flags": [],
    "statement": "This CPU has AVX2, which is what Strata requires; AVX-512 is absent, so prefill is slower than an AVX-512 host."
  },
  "tiers": [
    {
      "hardware_tier_gb": 48,
      "status": "missing-artifacts",
      "measured": true,
      "configuration_id": "qwen3-8-27b-gsq-rco-iq3s--auto--262k",
      "missing_bytes": 30277192704,
      "message": "Missing 28.2 GiB to run Qwen 3.8 27B GSQ-RCO IQ3_S at 256K (model, projector, draft); run scripts/download-artifacts --family … to fetch it.",
      "recommendation": {
        "main": "Qwen 3.8 27B GSQ-RCO IQ3_S",
        "main_family": "qwen3-8-27b-gsq-rco-iq3s",
        "context": 262144,
        "intelligence_score": 46.875,
        "throughput_tps": 39.39,
        "engine": {"name": "llama.cpp", "flavor": "mainline", "installed": true, "usable": true, "reason": null},
        "auxiliary": {"id": "maple-preview-tq2", "source": "pinned", "measured_auxiliary": "auto:…"},
        "artifacts": [
          {"role": "main:model", "path": "…/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", "kind": "file",
           "present": false, "size_bytes": 0, "expected_bytes": 11771546784,
           "missing_bytes": 11771546784, "reason": "missing"},
          {"role": "main:projector", "path": "…", "present": false, "reason": "missing"}
        ],
        "missing": [
          {"role": "main:model", "present": false, "missing_bytes": 11771546784, "reason": "missing"}
        ],
        "runnable": false,
        "summary": "…"
      },
      "options": []
    }
  ],
  "summary": "Missing 28.2 GiB to run …",
  "missing_bytes": 30277192704
}
```

`status` is one of `ready`, `missing-artifacts`, `engine-unsupported`,
`no-measurements` or `unresolvable`, and every tier carries a `message` that
says the same thing in one sentence — the amount missing in GiB, the CPU feature
the engine needs, or that nothing has been measured for that tier yet.

`auxiliary.source` is `pinned` when the measured pairing used something else
(`measured_auxiliary` records what was actually benchmarked) and `shared-main`
when the tier's main model is Maple itself.
