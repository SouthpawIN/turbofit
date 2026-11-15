# Tailnet peer federation — verified inventory and failure modes

TurboFit federates models across a Tailscale tailnet with no registry: each
gateway lists its own models, and every gateway probes its online peers for a
gateway on the provider port. This page records what the federation path
actually does, the peer inventory verified from a *different* tailnet node, and
the defects that made a live peer disappear silently.

## How discovery works

`turbofit_runtime.tailnet.scan_turbofit_peers()`:

1. `tailscale status --json` (honouring `TURBOFIT_TAILSCALE_SOCKET`).
2. Keep `Online == true` peers that carry a `DNSName`.
3. TCP-probe each peer's provider port **concurrently** under a wall-clock
   deadline, and classify the outcome:
   - `peers` — the port answered: a gateway is there, federate it.
   - `gateway_down` — the peer answered with `ECONNREFUSED`: the machine is up
     and reachable, nothing is listening on that port (gateway stopped, or bound
     to loopback).
   - `unreachable` — no TCP answer (peer down, firewall, no route).
   - `unresolved` — the MagicDNS name did not resolve.
4. `discover_turbofit_peers()` returns only the `peers` bucket; the full
   classification stays available via `last_peer_scan()` and is reported by the
   gateway's `/health` under `federation`.

The gateway caches the federated catalog per peer configuration for
`TURBOFIT_REMOTE_CACHE_TTL` seconds (default 30) with stale-while-revalidate, so
a transient probe failure never drops a live model mid-session. Remote entries
keep their own `residency: remote` / `mode: tailnet` metadata and carry the
peer's advertised RAM/VRAM bands (`metadata.peer_bands`).

Federation is not a routing table: a peer's models are usable from another node
by asking for `<peer-host>/<model-id>` on its own gateway, and a gateway with no
local model delegates `auto` to the first healthy peer.

## Verified peer inventory (probed 2026-11-15 from `hermes-vps`)

Probes ran from a **rootless, userspace-tailscaled container** (no TUN), so the
tailnet was reached through the daemon's SOCKS listener on `127.0.0.1:1055`.
That is the only honest vantage point: a probe from the peer's own box proves
nothing about whether other machines can reach it.

```bash
# raw port reachability (SOCKS5 CONNECT, no proxy env needed)
python3 skills/tailscale-device-mesh/scripts/probe-ports.py omarchy.tail6ff78b.ts.net 22 8091 9119 9443 9900

# HTTP through the userspace SOCKS proxy
curl -s --socks5-hostname 127.0.0.1:1055 http://omarchy.tail6ff78b.ts.net:8091/health
curl -s --socks5-hostname 127.0.0.1:1055 http://omarchy.tail6ff78b.ts.net:8091/status
curl -s --socks5-hostname 127.0.0.1:1055 http://omarchy.tail6ff78b.ts.net:8091/v1/models

# this node's own peer classification
TURBOFIT_TAILSCALE_SOCKET=/opt/data/state/tailscale/tailscaled.sock python3 -c \
  'from turbofit_runtime.tailnet import scan_turbofit_peers; import json; print(json.dumps(scan_turbofit_peers(), indent=2))'
```

| Peer / port | Result | What it means |
| --- | --- | --- |
| `omarchy` `100.68.42.68:22` | open | shell (Tailscale SSH / peer sshd) |
| `omarchy:8091` | **open** | TurboFit gateway reachable from another node — the port is *not* loopback-bound |
| `omarchy:9119` | refused (`reply=1`) | Hermes dashboard not running |
| `omarchy:9443` | open | `tailscale serve` → `127.0.0.1:8091` |
| `omarchy:9900` | refused (`reply=1`) | Hermes A2A/status server not running |
| `surface-duo-2:8091` | refused (`reply=1`) | Android peer online, no gateway — `gateway_down` |
| `christophers-macbook-pro`, `chriss-s10` | not probed | `Online: false` in `tailscale status` |

`omarchy` `/health` at probe time:

```json
{"ok": true, "main": "api-fallback", "aux": "ready", "remote_models": 0,
 "backend": "api-fallback", "local": false, "cloud_fallback": true,
 "declared_rung": "strata-flashnext-32k"}
```

Its `/v1/models` listed `auto`, `qwen3-8-flash-next-strata`,
`qwen3.8-flash-next-iq2_xs`, all with
`metadata.observed_at: null, freshness.stale: true` — the gateway was healthy
and reachable but **no local model was resident**; answers were served by the
hosted Nous API (`main: api-fallback`). A reachable gateway is not a loaded
model; check `/status` and a one-token completion before claiming otherwise.

`remote_models: 0` on that node is correct: `surface-duo-2` has no gateway, and
the MacBook/S10 were offline.

### Cross-node request, verified

Federating from this node through a raw-TCP relay (a proxy-unaware gateway
cannot use SOCKS):

```bash
python3 /opt/data/bin/socks-relay.py 18091 omarchy.tail6ff78b.ts.net 8091
TURBOFIT_GATEWAY_HOST=0.0.0.0 TURBOFIT_GATEWAY_PORT=18092 \
  TURBOFIT_PEERS=http://127.0.0.1:18091 TURBOFIT_PEER_DISCOVERY=0 \
  python scripts/turbofit-gateway.py
```

`GET /v1/models` on 18092 returned `127.0.0.1/qwen3-8-flash-next-strata` and
`127.0.0.1/qwen3.8-flash-next-iq2_xs` with `source: remote`,
`metadata.mode: tailnet`. A completion for `127.0.0.1/qwen3-8-flash-next-strata`
returned HTTP 200 with `content: "FEDERATION-OK"` — the request crossed the
tailnet to omarchy's gateway and came back. (The tokens were produced by
omarchy's hosted fallback, not a local model; the federation path is what that
request proves.)

## Defects this verified, and the fix

| Defect | Effect | Fix |
| --- | --- | --- |
| Gateway bound `127.0.0.1` by default (`build_gateway_launch`, `turbofit-gateway.py`) | Locally perfect, invisible to every peer; discovery dropped the node and nothing logged an error | Default bind is `0.0.0.0`, `EngineLaunch.network_bind` opts in explicitly, and a loopback bind logs a warning that federation is off |
| Refused port reported as "peer down" | A peer with a stopped gateway was indistinguishable from an offline machine | `probe_gateway()` returns `open` / `refused` / `unreachable` / `unresolved`; `scan_turbofit_peers()` keeps `gateway_down` separate and `/health` reports it |
| Serial probes, no overall bound, DNS not covered by the socket timeout | N peers × timeout, and a MagicDNS stall could hang the catalog | One daemon thread per peer, joined against a wall-clock deadline; DNS failures surface as `unresolved` |
| `tailscale` invoked without `--socket` | Rootless/userspace daemons reported `available: false` and discovery returned `[]` | `TURBOFIT_TAILSCALE_SOCKET` / `TAILSCALE_SOCKET` are honoured by every call |
| Catalog advertised model IDs only | Peers could not weigh a machine's RAM/VRAM when recommending a remote model | `/v1/models` carries a `node` envelope (`bands.ram_gb`, `bands.vram_gb`, pool kind, accelerators); remote entries expose it as `metadata.peer_bands` |
| Local role metadata overwrote remote entries | A remote model was relabelled `mode: local`, `residency: ready` | `provider_models()` leaves `source: remote` entries untouched |
| Gateway had no source admission while bound to all interfaces | An unauthenticated model proxy could be reached through a public port-forward | `client_address_allowed()` admits loopback, RFC1918/LAN, link-local and `100.64.0.0/10` only; other clients get 403 |

## Operating notes

- A peer upgraded before these changes does not advertise `node`; remote entries
  then carry no `peer_bands` until that machine runs a build with the envelope.
- `serve` still proxies `http://127.0.0.1:8091`, so the HTTPS route keeps
  working with either bind.
- Plain HTTP to the gateway is only for loopback, private networks and tailnet
  addresses. Publishing a gateway past the tailnet means HTTPS plus an
  authenticating proxy.
