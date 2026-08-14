# Physical catalog campaign evidence

This directory is the machine-local destination for immutable physical-fit campaign attempts.
Raw attempts are intentionally excluded from the release repository because they contain host-specific paths, process details, and hardware fingerprints.

Run `PYTHONPATH=src:. scripts/turbofit-catalog-campaign status` to inspect the current host campaign. Publish only reviewed, portable aggregate evidence through the canonical references files.
