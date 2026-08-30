# darwin/arm64: unmeasured hardware-class gap

`references/successful-runtime-profiles.json` is the evidence table behind
every measured runtime profile and Fit List winner. As of this note, **every
evidence path in that table is Linux-only** (for example
`/home/sovthpaw/...`, CUDA `build-cuda` llama.cpp binaries, NVML-style GPU
indexing). There is no Apple-Silicon on-box validation path yet.

Consequences:

- `turbofit-runtime-recommend --fit-only` on darwin/arm64 correctly returns
  `[]`: no evidence entry matches an Apple fingerprint, and per spec the
  recommender must not claim measurements it does not have.
- macOS (e.g. M1 Pro 32 GB) therefore gets its main-model identity from the
  Fit List heuristic ladder (`allowed_lineup.py` → Unleashed UD-Q3_K_XL at
  24 GB+ shared memory) rather than from on-box evidence. The 24 GB runtime
  profile rungs cite the Linux 2x24 CUDA measurement
  (`qwen-3-8-27b-unleashed-ud-q3-k-xl-auto-262k`) as the closest proven
  envelope; those throughput/VRAM numbers have **not** been reproduced on
  Metal.

Rule: do not fabricate Apple evidence to fill this gap. The gap closes when
someone runs the benchmark campaign on Apple Silicon and promotes a real
`darwin/arm64` fingerprint entry into the evidence table.
