# Maple and the missing sliding-window pattern

Maple-Preview (DeepGrove, 20B-A1B) is a 3:1 hybrid: three sliding-window-512
attention layers, then one full-attention layer, repeating — `layer_types` in the
published `config.json` is `SSSF` x6 for 24 layers.

The first published GGUF conversions do **not** carry
`maple.attention.sliding_window_pattern`:

```
general.architecture                       string  = 'maple'
maple.block_count                          uint32  = 24
maple.attention.sliding_window             uint32  = 512
...                                        (no sliding_window_pattern)
```

Newer engine sources read that array with `required=true`
(`src/models/maple.cpp`, `ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, ...)`),
so those checkpoints die before a single tensor is touched:

```
error loading model: error loading model hyperparameters:
array key not found in model: maple.attention.sliding_window_pattern
```

The tensors are fine. The metadata is fine. The engine is demanding something the
arch already knows.

## The fix

`runtime-patches/maple-llama.cpp/derive-sliding-window-pattern.patch` derives the
pattern in the arch loader and lets an explicit array win when a conversion writes
one:

```cpp
    // 3:1 SWA:GA hybrid; full-attention layers at il % 4 == 3 (config layer_types).
    // Derived here so a GGUF that omits maple.attention.sliding_window_pattern
    // (maple-tq2_0.gguf, maple-q2_0.gguf) still loads with the right pattern.
    hparams.set_swa_pattern(4, false);
    // An explicit per-layer pattern wins when the conversion wrote one.
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl,
                      hparams.n_layer(), false);
```

`set_swa_pattern(4, false)` fills `is_swa_impl` with `1,1,1,0` repeating, and the
optional `get_key_or_arr(..., false)` overwrites it only when the key exists.
No file surgery, no per-launch flag, no re-download: the unmodified checkpoint
loads.

`src/turbofit_runtime/maple_swa.py` mirrors the same rule in Python
(`sliding_window_pattern()`, `resolve_pattern()`, `describe()`), so doctor and
verification paths can reason about the pattern without a 5 GiB checkpoint.

## Installation

`scripts/install-native-runtimes` now applies `runtime-patches/<runtime-id>/*.patch`
with `git apply` after checking out the pinned revision and before `cmake`. The
apply step is idempotent: a repair run over an already-patched checkout is a no-op
instead of a hard failure, and `--check-only` reports the patch list in its JSON.

```bash
git clone https://github.com/stamsam/llama.cpp.git ~/.local/share/turbofit/runtimes/maple-llama.cpp-<rev>
scripts/install-native-runtimes --runtime maple-llama.cpp --backend cuda
# -> {"id": "maple-llama.cpp", ..., "patches": ["derive-sliding-window-pattern.patch"]}
```

The build output goes to a fresh `build-cuda/bin/`, so replace the runtime's
`bin/` directory wholesale (binary *and* the `libllama*`/`libggml*` shared
libraries) rather than copying only `llama-server`.

## Verification (omarchy, RTX 3090)

```
$ llama-server -m maple-tq2_0.gguf -c 65536 -ngl 99 --port 18090
0.34.953.744 I srv  llama_server: server is listening on http://127.0.0.1:18090

$ curl -s http://127.0.0.1:18090/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Reply with exactly: hello from maple"}],"max_tokens":60}'
{"choices":[{"message":{"role":"assistant","content":"hello from maple", ...}}],
 "usage":{"completion_tokens":36,"prompt_tokens":17},
 "timings":{"prompt_per_second":98.5,"predicted_per_second":103.0}}
```

`maple-tq2_0.gguf` was untouched (sha256 unchanged, no patched copy on disk).

## Why not just patch the GGUF, or `--override-kv`?

* Rewriting the header works but changes the artifact every user downloads and
  has to be redone for every conversion.
* `--override-kv` cannot express an array at all: `common`'s parser only accepts
  `int`, `float`, `bool`, `str`, and it splits the value on commas, so
  `array:uint32:1,1,1` fails with `invalid type for KV override`. Even if it could,
  `llama_model_loader::get_arr(std::array<T, N_MAX> &, ...)` never consults
  `kv_overrides` — the override cannot reach that code path.

The engine is the only layer that can default a missing per-architecture array.
