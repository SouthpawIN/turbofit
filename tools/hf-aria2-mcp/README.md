# hf-aria2-mcp

MCP stdio server. aria2c runs as the download daemon; aria2p is its RPC client.
Fetches the pinned files in `runtime-profiles/downloads.json` from Hugging Face and
HF-layout mirrors: up to 16 mirror URIs per file, 16 connections, `file-allocation=falloc`,
`continue=true`, sha-256 checked by aria2 against the catalog.

Requirements: `aria2c` on PATH, and `pip install aria2p` for the server (`catalog` needs neither).

    hf_catalog   groups, file counts, GB, mirrors
    hf_fetch     {group} and/or {files[]}; skips files already present or queued
    hf_status    progress, speed, live file-allocation/continue settings
    hf_control   pause | resume | remove by gid

Hermes (`~/.hermes/config.yaml`):

    mcp_servers:
      turbofit-hf-aria2:
        command: python3
        args: [/path/to/turbofit/tools/hf-aria2-mcp/hf_aria2_mcp.py]
        env:
          HF_MIRRORS: https://huggingface.co,https://hf-mirror.com

Destinations are the catalog `destination` under `TURBOFIT_MODEL_ROOT` (default
`~/Models/storage/gguf`), the same base as `scripts/download-models.py`.

Env: `HF_MIRRORS` (max 16), `HF_TOKEN` (then only huggingface.co is used, so the token never
goes to a mirror), `TURBOFIT_MODEL_ROOT`, `HF_ARIA2_PORT` (6870), `HF_ARIA2_STATE`.
