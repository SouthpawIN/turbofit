# Daily model intelligence pipelines

These collectors update only `research/candidates.json`. They cannot promote a candidate, modify runtime profiles, or write production route state.

Authoritative Hermes cron reference: <https://hermes-agent.nousresearch.com/docs/user-guide/features/cron>

## Collectors

```bash
PYTHONPATH=. python3 research/discover_huggingface.py
PYTHONPATH=. python3 research/discover_model_news.py --url https://huggingface.co/blog/feed.xml
PYTHONPATH=. python3 research/discover_api_models.py --provider openrouter --url https://openrouter.ai/api/v1/models
```

All inputs are public, read-only endpoints. The scripts deliberately accept no token or credential argument and copy only allowlisted metadata. Every output remains `status: candidate` until the benchmark promotion gates pass.

## Active scheduled refresh

The approved local Hermes job runs `scripts/scheduled-refresh` every 12 hours. The script executes all public discovery collectors, updates tracked revisions, writes `research/benchmark-queue.json`, advances one serialized physical/intelligence batch when the campaign service is not already active, promotes an exact-tier winner when eligible, regenerates the hardware report, and rebuilds TurboFit List. It no longer treats every GPU PID as an automatic skip; campaign ownership and GPU-clear safety remain inside the benchmark runners.

The long-running `turbofit-benchmark-campaign.service` is the continuous worker. Hermes cron performs a single safe batch only when that service is inactive, so the two schedulers cannot race.

## Collector reference

### Hugging Face candidates

- Suggested schedule: `15 5 * * *`
- Workdir: `/home/sovthpaw/projects/turbofit`
- Toolsets: `terminal`
- Prompt:

```text
Run exactly `PYTHONPATH=. python3 research/discover_huggingface.py` in this project. Report the JSON diff counts. Do not download models, use credentials, edit runtime profiles, promote candidates, schedule jobs, or write anywhere except research/candidates.json.
```

### Model news

- Suggested schedule: `30 5 * * *`
- Workdir: `/home/sovthpaw/projects/turbofit`
- Toolsets: `terminal`
- Prompt:

```text
Run exactly `PYTHONPATH=. python3 research/discover_model_news.py --url https://huggingface.co/blog/feed.xml` in this project. Report the JSON diff counts. Do not use credentials, edit runtime profiles, promote candidates, schedule jobs, or write anywhere except research/candidates.json.
```

### Public API model catalog

- Suggested schedule: `45 5 * * *`
- Workdir: `/home/sovthpaw/projects/turbofit`
- Toolsets: `terminal`
- Prompt:

```text
Run exactly `PYTHONPATH=. python3 research/discover_api_models.py --provider openrouter --url https://openrouter.ai/api/v1/models` in this project. Report the JSON diff counts. Do not use credentials, call paid inference, edit runtime profiles, promote candidates, schedule jobs, or write anywhere except research/candidates.json.
```

Separate collector jobs are no longer needed on this host; `scripts/scheduled-refresh` owns the ordered discovery → queue → benchmark → List transaction. Use the snippets above only for manual collector debugging.
