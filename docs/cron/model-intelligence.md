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

## Proposed Hermes jobs

Do not create these jobs until the owner approves both schedule and delivery target. Cron sessions start without chat context, so each prompt below is self-contained. Set `workdir` to the absolute Turbofit repository and restrict tools to `terminal`.

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

When approved, create jobs with Hermes `cronjob(action="create", schedule=..., prompt=..., workdir="/home/sovthpaw/projects/turbofit", enabled_toolsets=["terminal"], deliver=...)`. Explicitly choose a gateway-connected delivery target if notifications are wanted; this TUI has no live origin-delivery channel.
