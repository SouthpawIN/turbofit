# Turbofit Bonsai 1-bit capsule

This capsule contains the exact llama.cpp runtime used for the verified local
Bonsai baseline. Model weights are mounted at `/models` so the same runtime can
be distributed with or without the large model layer while retaining exact
model filenames and flags.

```bash
docker build -t turbofit-bonsai-1bit:local .
docker run --rm --gpus all -p 11610:11610 \
  -v /path/to/Bonsai-27B:/models:ro \
  turbofit-bonsai-1bit:local
```