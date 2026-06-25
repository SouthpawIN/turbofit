# Binary Selection: Atomic Fork vs Stock llama.cpp

When a model fails to load with the atomic fork binary, use this decision tree to determine the fix.

## When to use the atomic fork (`binary:` field in catalog)

- Model uses **turbo4/turbo3/turbo2 KV cache** (TurboQuant) — stock doesn't support these cache types
- Model uses **`--spec-type nextn`** — stock doesn't have the NextN code path
- Model is a **27B dense Qwen3.6** (Darwin, Prism, Carwin, Qwopus, Qwable) — the atomic fork loads their mmproj fine (n_embd=5120)

## When to use stock llama.cpp (no `binary:` field)

- Model uses **q8_0 or q4_0 KV cache** — stock supports these
- Model uses **`--spec-type draft-mtp`** — stock supports this
- Model is a **35B-A3B MoE (Carnice, Darwin APEX)** with vision — the atomic fork's clip loader crashes on n_embd=2048 mmproj files with `ggml_backend_buffer_set_usage` abort in `clip_model_loader::load_tensors`
- GPU VRAM is tight and spec decoding would double VRAM usage (draft model loads a second GGUF copy)

## Decision tree

```
Does the model need turbo2/turbo3/turbo4 KV cache?
  YES → atomic fork (binary: field required)
         Does the model also have a 35B-A3B mmproj (n_embd=2048)?
           YES → atomic fork can't load the mmproj. Two options:
                  (a) Use stock binary + q8-kv (lose turbo cache, gain vision)
                  (b) Use atomic fork without mmproj (keep turbo cache, lose vision)
              → Pick based on whether vision is needed for this model's role
  NO  → stock llama.cpp (no binary: field)
         Use q8-kv or q4-kv presets instead of turboN-kv
         Use draft-mtp instead of nextn for spec decoding
```

## VRAM considerations for spec decoding

`--model-draft <same.gguf>` loads the model GGUF **twice** — once for the main model, once for the draft. On a GPU that already has other processes (ACE-Step, training, display server), this can cause OOM. Always check `nvidia-smi --query-gpu=memory.free` before enabling spec decoding.

If VRAM is tight, remove spec decoding presets entirely. The model still works — just without the 2-3x speedup from speculative decoding.

## Verification commands

```bash
# Check if atomic fork binary works at all
LD_LIBRARY_PATH=<atomic_build_dir>/bin <atomic_binary> --version

# Test model load with mmproj (15s timeout, small ctx)
timeout 15 LD_LIBRARY_PATH=<atomic_build_dir>/bin <atomic_binary> \
  -m <model.gguf> --host 127.0.0.1 --port 19999 \
  --mmproj <mmproj.gguf> -c 4096 --flash-attn on -ngl 5 2>&1 | tail -5

# Check what spec types the binary supports
<binary> --help 2>&1 | grep "spec-type"
```
