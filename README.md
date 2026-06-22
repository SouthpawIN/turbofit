# turbofit

llama.cpp installer + launcher + Hermes-Agent config manager. Brings `serve` and `name` shell commands that wire `llmfit` (memory fit), `llama-server` (inference engine), and `~/.hermes/config.yaml` (model routing) into one workflow.

## Install

```bash
hermes skills install SouthpawIN/turbofit
~/.hermes/skills/turbofit/scripts/install.sh
source ~/.bashrc
serve install
```

## What it does

- **Auto-installs llama.cpp** from source (`serve install`), keeps it updated (`serve update`)
- **Checks model fit** via `llmfit` (VRAM + RAM) — refuses to launch models that won't fit
- **Launches servers detached** — survives shell death, logs to `~/.local/share/turbofit/logs/`
- **Wires models into Hermes-Agent** as main or auxiliary (all 9 aux tasks)

## Commands

```bash
serve install                              # Install llama.cpp from source
serve update                               # Update to latest master
serve check                                # Show version status
serve fit <model>                          # Run llmfit fit check
serve string <alias>                       # Print launch string
serve <alias>                              # Launch detached, show port + logs
serve stop <alias>                         # Stop a server
serve list                                 # List running servers
serve catalog                              # Show registered aliases
name <alias> <path>                        # Register a model alias
serve main <alias>                         # Launch + set Hermes main + start hermes
serve aux <alias>                          # Launch + set Hermes aux + start hermes
serve herm <alias>                         # Launch + main + herm TUI + hermes
serve herm aux <alias>                     # Launch + aux + herm TUI + hermes
```

## Shell aliases (installed by install.sh)

```bash
name qwen-8b ~/models/Qwen3-8B.Q4_K_M.gguf    # register

serve qwen-8b              # launch + show port/logs
serve main qwen-8b         # launch + main + hermes TUI
serve aux qwen-8b          # launch + aux + hermes TUI
serve herm qwen-8b         # launch + main + herm TUI + hermes
serve herm aux qwen-8b     # launch + aux + herm TUI + hermes
serve main qwen-8b --gateway  # launch + main + hermes gateway
serve aux qwen-8b --gateway   # launch + aux + hermes gateway
```

## Enforces a 64K context floor

Every launch string and server uses `ctx_size: 65536` minimum (Hermes-Agent requirement). Smaller context values are clamped automatically.

## License

MIT