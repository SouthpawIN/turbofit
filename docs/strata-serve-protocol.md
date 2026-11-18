# Strata's serve protocol (and why TurboFit needs a bridge)

`strata --serve` is **not an HTTP server**. It is a stdin/stdout line protocol.
Verified against the engine source (`src/program/generate.cpp`) and a live run.

## Handshake

On startup the engine prints its configuration and then:

```
INFO context=32768 kv=q4_0 expert_slots=12834 expert_cache_mib=17973 ... engine=0.1.39
READY 32768 stop
```

`READY <max_context> stop` means "I am up, my window is N, and I honour STOP".
Clients must skip unknown lines until READY (the source says servers before
0.1.8 emit extra lines).

## Commands (one per line, on stdin)

| Line | Meaning |
|---|---|
| `BGEN <token ids…>` | generate from a tokenised prompt |
| `BGENI <…>` | generate with an image attachment |
| `BSTOP <…>` | stop the current generation |
| `BYIELD` | yield between requests |
| `VRAM [reserve_mib]` | adjust the VRAM reserve between requests |
| `STOP` | request a stop |
| `QUIT` | shut down |

**stdin must stay open.** Launching with `< /dev/null` gives immediate EOF and the
engine exits right after READY — which looks exactly like a crash. Keep a pipe or
FIFO attached.

## What this means for TurboFit

The engine takes **token ids**, not text, and speaks lines, not HTTP. To offer a
Strata model on the OpenAI-compatible endpoint TurboFit publishes, a bridge must:

1. spawn the engine with a persistent stdin pipe,
2. wait for `READY <ctx> stop`,
3. tokenise incoming chat text with the model's own tokenizer,
4. write `BGEN <ids>`, stream the engine's output back as SSE chunks,
5. map `stop` to `BSTOP`, and shutdown to `QUIT`.

Until that bridge exists, Strata models can be *placed* and *launched* by
TurboFit but cannot answer `/v1/chat/completions`. That is the honest state, and
it is the next piece of work — not a flag.

## Verified two-card load (2026-11-18)

The configuration below reached READY on a 2x RTX 3090, AVX2-only CPU:

```
--serve --layer-split 24 --split-device 1 --trim-stage-weights
--max-context 32768 --expert-cache 512 --kv q4_0 --ple-io direct
--prefill 2048 --spec 4 --mtp <Strata-data>/mtp
```

```
expert_slots=12834  expert_cache_mib=17973  expert_slots_primary=546
arena_mib=33812  pool_workers=15  spec=6  mtp_max=4  engine=0.1.39
```

Both cards hold their own layers, 17.9 GiB of experts sit in VRAM and 33.8 GiB in
the host arena. No file streaming.

### The MTP error, resolved

`mtp: cannot read dense.bin` was **operator error, not an engine bug**:
`tools/mtp_rt.py --out` takes a **directory**, and writes `dense.bin`,
`experts.bin` and `dense.txt` inside it. Passing `--out …/mtp/dense.bin` created a
*directory* named `dense.bin`, so the engine could not open it as a file.

Correct invocation:

```bash
python tools/mtp_rt.py --gguf <Strata-data>/mtp/mtp-q2_0.gguf --out <Strata-data>/mtp
```

It needs numpy; on a host whose Pythons have no pip, install it with
`uv pip install --python <venv>/bin/python numpy`.
