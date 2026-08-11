# Sirvir

Sirvir is the customer-service profile bundled with Turbofit. It helps users install, configure, use, and troubleshoot Turbofit on their own hardware, and turns reusable support findings into evidence-backed pull-request suggestions.

## Install or update

The normal path is the **Install Sirvir customer service profile** option in `/turbofit setup`, Hermes Dashboard, or Hermes Desktop. Turbofit replaces only distribution-owned profile files and preserves Sirvir's memories, sessions, credentials, and other user state.

After installation, start a new session with the Sirvir profile using the profile controls available in your Hermes client.

## What Sirvir can help with

- checking prerequisites and selecting the correct plugin installation path;
- verifying that Hermes loaded Turbofit's tools and slash command;
- choosing Auto or exact evidence-backed model configurations;
- configuring Turbofit as Hermes' local primary provider;
- installing supported native runtimes and optional Desktop/Dashboard surfaces;
- understanding `auto`, `active:main`, `active:aux`, pressure adaptation, and hardware tiers;
- diagnosing plugin, provider, gateway, artifact, native-runtime, route, hardware, and benchmark failures;
- drafting concrete PR suggestions with reproduction evidence and acceptance tests.

Sirvir does not publish a PR merely because it suggested one. Repository writes require an explicit follow-up request.

## Local-only boundary

Sirvir's bundled default model route is Turbofit itself. Every model route remains local. Under pressure, Turbofit steps through local auxiliary sharing, smaller local contexts/models, and its minimum local floor. If that floor cannot serve a request, it fails closed and preserves diagnostics. Complete bootstrap diagnosis from another working local model or an ordinary shell, then return to Sirvir after the local Turbofit gateway is healthy. Do not copy credentials into the Sirvir distribution files.

## Support evidence

For the fastest useful diagnosis, provide the exact user-facing error and—when safe—the output from:

```bash
scripts/turbofit-runtime status
curl -fsS http://127.0.0.1:8091/v1/models
```

Redact usernames, hostnames, private URLs, tokens, and keys. Never share `.env`, `auth.json`, or complete private logs.
