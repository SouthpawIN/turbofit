# turbofit

Hardware-fit checker + llama.cpp launch string generator. Bridges `llmfit` (system memory analysis) to `llama-server` (the inference engine from llama.cpp). Enforces a 64K Hermes-Agent context floor.

## Install

```bash
hermes skills install https://github.com/SouthpawIN/turbofit/blob/main/SKILL.md
# or:
hermes skills install SouthpawIN/turbofit
```

## What it does

- **Memory-fit analysis** — answers "will this model fit in my system memory?" using `llmfit fit` and `llmfit plan`
- **Launch string generator** — produces a copy-pasteable `llama-server` command with sensible defaults (flash attention, jinja, 64K context floor)
- **Hardware-fit decision tree** — matches GPU/RAM tier to recommended model sizes
- **VRAM budget reference** — KV cache + model size tables at common context lengths

## When to use it

- "What model fits on my hardware?" — `llmfit fit --perfect`
- "Give me a llama-server command for this model" — bridge script
- "How much VRAM do I need for X at context Y?" — `llmfit plan`
- Picking a model to download — see top fits before pulling 20+ GB

## License

MIT
