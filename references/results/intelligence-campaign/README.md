# Intelligence campaign evidence

This directory is the machine-local destination for immutable DeepSWE and agentic-pair campaign attempts.
Raw attempts are intentionally excluded from the release repository because they contain host-specific paths, runtime logs, and trial artifacts.

Run `PYTHONPATH=src:. scripts/turbofit-intelligence-campaign status` to inspect the current host campaign. Publish only reviewed, checksum-bound aggregate scores through `references/intelligence-scores.json`.
