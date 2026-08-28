# Remote Mac model server with Hermes in a Proxmox LXC

Turbofit can run on one machine while a Hermes Agent gateway runs on another. This is useful when the Mac has the Apple Silicon memory bandwidth and Metal acceleration, while the Hermes runtime, tools, files, sessions, and credentials live in a Proxmox LXC.

```text
Mac                                  Proxmox LXC
Turbofit + local model server  <───  Hermes Agent + hermes serve
                                     tools, files, sessions, state
```

The connection has two separate hops:

1. Hermes Desktop on the Mac connects to `hermes serve` in the LXC.
2. Hermes in the LXC connects to Turbofit's OpenAI-compatible `/v1` endpoint on the Mac.

## 1. Publish Turbofit on the private network

Install and configure Turbofit on the Mac as usual. The local provider gateway listens on `127.0.0.1:8091` by default. For a private cross-machine connection, use Tailscale Serve:

```text
/turbofit serve
```

This publishes the provider gateway on a private HTTPS URL such as:

```text
https://mac-name.example-tailnet.ts.net:9443/v1
```

Turbofit uses Tailscale Serve, not Funnel. Do not expose the provider gateway directly to the public internet.

If you are using a trusted LAN instead of Tailscale, the model server or Turbofit gateway may use a private address such as `192.168.1.50`. Restrict the Mac firewall to the LXC or trusted LAN. Plain HTTP is intentionally accepted only for loopback, private-network, and Tailscale addresses; public endpoints must use HTTPS.

## 2. Verify connectivity from the LXC

Run this from inside the Proxmox LXC:

```bash
curl -fsS https://mac-name.example-tailnet.ts.net:9443/v1/models | jq .
# Or, for a private LAN endpoint:
curl -fsS http://192.168.1.50:8091/v1/models | jq .
```

If this fails, fix routing, Tailscale membership, firewall rules, or the Mac-side Turbofit service before changing Hermes configuration. `localhost` in the LXC refers to the LXC itself, not the Mac.

## 3. Configure the remote Hermes profile

The provider configuration belongs to the Hermes profile running in the LXC. Add the Mac endpoint under `providers:` and select it as the active model route:

```yaml
model:
  provider: custom:mac-turbofit
  default: auto

providers:
  mac-turbofit:
    name: Mac Turbofit
    api: https://mac-name.example-tailnet.ts.net:9443/v1
    api_key: not-needed
    transport: chat_completions
    default_model: auto
    models:
      auto: {}
      active:main: {}
      active:aux: {}
```

For a LAN deployment, replace the `api` value with the Mac's private address and port. Keep the endpoint in the LXC's profile config; the Mac Desktop client's local Hermes config is a different machine and a different profile.

Start a new Hermes session after changing the provider. In the LXC, `hermes model` can select the named **Mac Turbofit** provider. If the endpoint advertises multiple models, the picker can use the returned model inventory. The Hermes tools and filesystem still execute in the LXC.

## 4. Select it from Hermes Desktop

On the Mac:

1. Open Hermes Desktop and select the remote Proxmox gateway under **Settings → Gateways**.
2. Select the intended remote Hermes profile.
3. Open the model picker.
4. Choose **Mac Turbofit** and its model.

The picker is scoped to the selected remote gateway/profile. Selecting the provider changes the remote Hermes profile; it does not start a second local Hermes backend or move tool execution to the Mac.

## Security notes

- Prefer Tailscale Serve for the provider URL.
- Use HTTPS for public or otherwise untrusted networks.
- Do not put credentials in fallback entries or checked-in files.
- If the model server requires authentication, configure a `key_env` in the LXC's Hermes provider and store the secret in the LXC's `~/.hermes/.env`.
- Keep Turbofit's provider gateway private; it is an inference endpoint, not a public web service.
