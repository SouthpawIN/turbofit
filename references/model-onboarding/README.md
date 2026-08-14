# Day-zero model onboarding

Use `scripts/turbofit-model-onboard` when the released repository, pinned commits, GGUF names, byte sizes, SHA-256 checksums, and launch methods are known.

Install the bounded official-release watcher once:

```bash
scripts/install-qwen38-watcher
```

It checks only public models owned by the official `Qwen` Hugging Face organization once per hour. Third-party prerelease repositories are recorded as untrusted and are never onboarded. State is written to `~/.local/state/turbofit/qwen38-release.json`. When the official release appears, the watcher requests a graceful benchmark pause; the catalog worker finishes its current row, then the serialized orchestrator exits cleanly so the GPUs are reserved for Qwen ingestion. A successful `turbofit-model-onboard --apply` clears the pause markers after replacing the active identities.

```bash
scripts/turbofit-model-onboard /path/to/qwen3.8-27b.json          # validate only
scripts/turbofit-model-onboard /path/to/qwen3.8-27b.json --apply  # atomic active-catalog replacement
```

The input schema is `turbofit.model-onboarding/v1` and contains exactly:

- `family`: released family name.
- `replace_model_ids`: the six active GRM variant IDs being replaced.
- `models`: six complete `turbofit.model-catalog/v2` model records with pinned 40-character revisions.
- `recipes`: one exact `model-recipes.json` variant record per new model ID.
- `artifacts`: exact destination, family IDs, Hugging Face repository, pinned revision, path, SHA-256, and byte size.

The command rejects placeholders, missing recipes, unpinned artifacts, ID collisions, and variant-count drift. It regenerates and validates the full configuration matrix, resolves every new production recipe, retains immutable historical campaign evidence under the old IDs, and marks the retired GRM IDs deferred.

After applying:

1. Download and SHA-verify every artifact.
2. Probe llama.cpp metadata and exact context support.
3. Run Qwen current-recipe catalog rows first.
4. Qwen combinations appear in Settings only after physical fit/TPS evidence succeeds.
5. Run screening → promotion → release intelligence.
6. Promote Qwen into Auto hardware rungs only after exact tier evidence; until then Auto uses the remaining Bonsai-safe rungs, never deferred GRM.
