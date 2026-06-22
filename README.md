# llmfit-turbohaul

Generate optimized llama-server launch strings and turbohaul-manager manifests using llmfit hardware detection + TurboQuant flag doctrine.

## Install

```bash
hermes skills install https://github.com/SouthpawIN/llmfit-turbohaul/blob/main/SKILL.md
# or:
hermes skills install SouthpawIN/llmfit-turbohaul
```

## What it does

- **Hardware scan** — runs `llmfit system` and `llmfit fit` to find models that fit your GPU
- **Launch string generator** — produces a copy-pasteable `llama-server` command with the TurboQuant Flag Doctrine baked in
- **Turbohaul manifest builder** — generates the YAML for `PUT /api/manifests/{tag}` including the 5 required TQ flags
- **Bridge script** — `llmfit → turbohaul` pipeline in one command

## When to use it

- Setting up a new model in turbohaul-manager and need the right flags
- Asking "what model fits on my GPU" and want a launch string immediately
- Testing turbohaul-manager (MrTrench's Ollama-shape inference engine) on your hardware
- Comparing what runs on different hardware before recommending models

## License

MIT
