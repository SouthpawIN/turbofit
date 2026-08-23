# Turbofit

![Turbofit — unified backend, amber and mint aesthetic](assets/turbofit-hero.png)

**One model provider. Every machine. The best local configuration the hardware can safely sustain.**

Turbofit is a first-class [Hermes Agent](https://github.com/NousResearch/hermes-agent) provider and adaptive local-inference runtime. It inventories physical compute and memory, recommends an evidence-backed ladder, and exposes one OpenAI-compatible endpoint. The client-facing name stays `auto` while the backing model, context, and auxiliary mode change.

![Turbofit settings in Hermes Desktop, including fallback routing, multimodal model selection, and hardware-fit recommendations](assets/hermes-desktop-turbofit-settings.png)

```text
provider: custom:turbofit
model: auto
```

**TurboFit Check** scans this machine and applies Auto or a selected compatible lane. The **[TurboFit List](docs/turbofit-list.md)** is the separate evidence-only winner table by hardware class.

> Catalog entries are **candidates** until the physical campaign passes. Compile, download, or estimated fit is not a winner.

## 2.3 lineup

| Model | Entries | Role | Status |
|---|---|---|---|
| **Qwen 3.8 27B Unleashed** | `UD-IQ3_XXS`, `UD-Q3_K_XL` | Uncensored dense 27B, 262K, vision | Active catalog candidates |
| **Ornith 1.5 35A3B** | `Q4_K_M` MoE | Default auxiliary | Active auxiliary candidate |
| **Qwen 3.8 27B** | `Q4_K_M`, `Q8_0`, `BF16` ± MTP | Native image/video | Active catalog candidates |
| **MiniMax Music 3** | `minimax-music3` | Full-song music | Pinned integration candidate |
| **NVIDIA Parakeet TDT 0.6B v3** | `parakeet-tdt-0-6b-v3` | Local STT | Pinned integration candidate |
| **Soprano TTS** | `soprano-tts` | Local TTS | Pinned integration candidate |

| Usable memory | Main path |
|---|---|
| 96 GB+ | Qwen 3.8 27B 16-bit until Unleashed FP16 GGUF exists |
| 24–95 GB | Unleashed UD-Q3_K_XL |
| 16 GB | Unleashed UD-IQ3_XXS |
| 8 GB | Bonsai 27B |
| Below 8 GB dedicated | Bonsai 27B Q1 shared-main, portable-fit until benching that box |

Auxiliary is **Ornith 1.5 35A3B**, optional **Carwin Nano**, or **auto**. Pressure steps: offload Ornith experts → lower context → aux auto → smaller Unleashed → Bonsai → keyless Nous free models. Optional FreeToken is a pinned NVIDIA MoE **candidate**, never Auto: [`docs/freetoken-runtime.md`](docs/freetoken-runtime.md).

![Turbofit 2.3 auto-fit model ladder from Bonsai through Unleashed and Nous keyless free](assets/turbofit-2.3-model-ladder.png)

## Install

Plugin install registers tools, slash commands, and the provider schema. It does **not** download models or bind `:8091`.

| Layer | What | Starts when |
|---|---|---|
| Hermes messaging gateway | Discord / Telegram / cron | `hermes gateway …` |
| Turbofit provider gateway | OpenAI `/v1` on **`127.0.0.1:8091`** | Desktop Apply, `/turbofit shift`, or native service |
| Native model server | llama-server / backend | Turbofit selection |
| Tailscale Serve | Private HTTPS to other tailnet devices | `/turbofit serve` |

```bash
hermes plugins install --enable https://github.com/SouthpawIN/Turbofit.git
```

Use the **default** Hermes profile (not Sirvir). Fully quit Desktop or start a fresh CLI session so the plugin reloads.

```text
/turbofit status
/turbofit setup
```

`/turbofit setup` refreshes Hermes **Desktop → Turbofit**. Dashboard is deprecated.

```text
/turbofit                 # scan + intelligence / balanced / speed
/turbofit update          # plugin + Desktop surface + Sirvir
/turbofit shift up        # next smarter measured combo
/turbofit shift down      # next lighter measured combo
/turbofit shift bonsai    # recommended combo for that model
/turbofit shift intelligence
/turbofit serve           # publish :8091 on your tailnet
/turbofit serve status
```

```bash
curl -fsS http://127.0.0.1:8091/v1/models
```

JSON with `auto` = healthy. Connection refused (`WinError 10061` on Windows) means the Turbofit stack is not running — not a firewall miss, and not a Hermes messaging-gateway problem.

Windows headless path: [`docs/windows-native-install.md`](docs/windows-native-install.md).

## Tailscale — one-setup model server

Turbofit still uses **Tailscale Serve** (private). It never uses Funnel.

On the machine that already answers `http://127.0.0.1:8091/v1/models`:

```text
/turbofit serve
```

That:

1. requires `tailscale` logged in and connected;
2. publishes the provider gateway (`127.0.0.1:8091`) as `https://<magicdns>:9443/v1`;
3. writes that HTTPS URL into the local Hermes `custom:turbofit` provider;
4. also publishes Desktop/setup on `https://<magicdns>:9444/` when that local port is live.

Any other device on the same tailnet uses:

```yaml
providers:
  turbofit:
    base_url: https://<this-machine>.<tailnet>.ts.net:9443/v1
    api_key: not-needed
    model: auto
```

Desktop: **Serve on Tailscale**. Same as `/turbofit serve`. Check with `/turbofit serve status` or `tailscale serve status`.

Public binds stay rejected. Loopback and verified tailnet addresses are the only plain-HTTP exceptions.

## Hermes configuration

```yaml
model:
  provider: custom:turbofit
  default: auto

providers:
  turbofit:
    base_url: http://127.0.0.1:8091/v1
    api_key: not-needed
    model: auto
    model_name: auto
    provider: turbofit
    tool_format: hermes

fallback_providers:
  - provider: nous
    model: stealth/ox-alpha
  - provider: nous
    model: stepfun/step-3.7-flash:free
  - provider: nous
    model: tencent/hy3:free
  - provider: nous
    model: poolside/laguna-s-2.1:free
  - provider: nous
    model: poolside/laguna-xs-2.1:free
  - provider: custom:turbofit
    model: auto
```

Fallback rows are `provider` + `model` only. Desktop edits the whole ordered chain.

## Sirvir

![Sirvir — GitHub-current Turbofit support](https://raw.githubusercontent.com/SouthpawIN/sirvir/main/assets/sirvir-hero.png)

[Sirvir](https://github.com/SouthpawIN/sirvir) is GitHub-current Turbofit customer service: install, Q&A with source/commit citations, and tested pull requests. **Install Sirvir** from Desktop or `/turbofit update`. **Sirvir installs Turbofit when it is missing.** Reciprocal bootstrap:

```bash
git clone https://github.com/SouthpawIN/sirvir.git
cd sirvir
scripts/install
```

Verify Turbofit from the **default profile first**. Sirvir cannot chat until `http://127.0.0.1:8091/v1/models` answers. Profile install must use `https://github.com/SouthpawIN/sirvir.git`.

## How it adapts

```text
quality-main + auxiliary + 262K
              │ pressure
              ▼
smaller context / shared aux / smaller model
              │
              ▼
keyless Nous free fallback
```

Pressure drops fast; healing is slower and hysteretic. Transitions lock, fail closed, and roll back. `/turbofit shift` walks the same measured ladder manually. Hardware pools, backends, and offload: [`docs/runtime-backends.md`](docs/runtime-backends.md).

## Desktop

Hermes Desktop **Turbofit** is the setup surface: hardware, score bars, shift, update, Tailscale serve, fallbacks, runtimes, Sirvir, multimodal. Source: [`desktop/plugin.js`](desktop/plugin.js).

## More detail

| Topic | Doc |
|---|---|
| Evidence-only winners | [`docs/turbofit-list.md`](docs/turbofit-list.md) |
| Check vs List | [`docs/turbofit-check.md`](docs/turbofit-check.md) |
| Model matrix, Unleashed, DFlash2, Bonsai | [`docs/model-matrix.md`](docs/model-matrix.md) |
| Physical + intelligence campaigns | [`docs/campaigns.md`](docs/campaigns.md) |
| Multimodal pins | [`docs/multimodal.md`](docs/multimodal.md) |
| Live 2x24 TPS | [`docs/hardware-tier-tps.md`](docs/hardware-tier-tps.md) |
| Windows native service | [`docs/windows-native-install.md`](docs/windows-native-install.md) |

## Developer verification

```bash
PYTHONPATH=src:. python3 -m pytest -q
node --check dashboard/dist/index.js
node --check desktop/plugin.js
scripts/release-check
```

## License

See [`LICENSE`](LICENSE).
