# Turbofit container control plane

This is the first containerized Turbofit vertical slice. Hermes points at the
stable OpenAI-compatible endpoint on port `8091`; the control plane routes to
the active local main backend and exposes `/status` with GPU and fleet state.

```bash
docker compose up --build -d
curl http://127.0.0.1:8091/health
curl http://127.0.0.1:8091/status
```

Linux GPU deployments require the NVIDIA Container Toolkit on the host. Docker
must be able to run `docker run --rm --gpus all ...`; without that host runtime,
the control plane can proxy requests but cannot see `nvidia-smi` or launch a GPU
model capsule.

## Matrix evidence runner

Run one checklist row through an OpenAI-compatible Turbofit endpoint:

```bash
python3 ../scripts/matrix-benchmark.py \
  --base-url http://127.0.0.1:8091 \
  --main GRM-2.6-Plus-Q4 \
  --aux ternary-bonsai-27b-dspark \
  --context 262K \
  --anchor grm-26-plus-ternary-bonsai-262k \
  --label 'GRM 2.6 Plus:Ternary Bonsai @ 262K context' \
  --mark-success
```

The runner writes a dated evidence page under the wiki and only marks the
anchored checklist row `[x]` when health and smoke inference both pass. Omit
`--mark-success` for a dry evidence run.

For the current host Bonsai test, `state.json` routes to the verified llama
server on `127.0.0.1:11610`. The first capsule deliberately uses a host-model
mount/runtime while the model-image publishing step is added; this lets us
verify routing without duplicating multi-gigabyte weights during development.