#!/usr/bin/env python3
"""Turbofit MCP server: aria2c runs as the download daemon, aria2p is its RPC client.

Fetches the pinned files of runtime-profiles/downloads.json (same schema, groups and
destinations as scripts/download-models.py). Each file is one aria2 job: up to 16
Hugging Face-layout mirror URIs, 16 connections, file-allocation=falloc, continue=true,
sha-256 checked by aria2 against the catalog.

  python3 tools/hf-aria2-mcp/hf_aria2_mcp.py              MCP stdio server
  python3 tools/hf-aria2-mcp/hf_aria2_mcp.py catalog [G]  list groups
  python3 tools/hf-aria2-mcp/hf_aria2_mcp.py fetch G      queue a group
  python3 tools/hf-aria2-mcp/hf_aria2_mcp.py status       progress

Env: HF_MIRRORS (comma list, max 16), HF_TOKEN (then huggingface.co only),
TURBOFIT_MODEL_ROOT (default ~/Models/storage/gguf), HF_ARIA2_PORT (6870), HF_ARIA2_STATE.
Needs aria2c on PATH and `pip install aria2p` (imported lazily, so `catalog` needs neither).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from turbofit_runtime.downloads import DownloadCatalog, DownloadFile  # noqa: E402

MAX = 16
HF = "https://huggingface.co"
CATALOG = ROOT / "runtime-profiles" / "downloads.json"
BASE = Path(os.environ.get("TURBOFIT_MODEL_ROOT", "~/Models/storage/gguf")).expanduser()
STATE = Path(os.environ.get("HF_ARIA2_STATE", "~/.cache/hf-aria2-mcp")).expanduser()
PORT = int(os.environ.get("HF_ARIA2_PORT", "6870"))
FIELDS = ("repo_id", "revision", "path", "destination", "sha256", "size_bytes")
FILE_OPTS = {"file-allocation": "falloc", "continue": "true", "max-tries": "0", "retry-wait": "5",
             "allow-overwrite": "false", "auto-file-renaming": "false"}


def mirrors() -> list[str]:
    raw = os.environ.get("HF_MIRRORS", f"{HF},https://hf-mirror.com")
    bases = list(dict.fromkeys(b.strip().rstrip("/") for b in raw.split(",") if b.strip()))
    if os.environ.get("HF_TOKEN"):  # aria2 sends headers to every URI: keep the token on HF only
        bases = [HF]
    return bases[:MAX]


def uris(item: DownloadFile) -> list[str]:
    return [f"{b}/{item.repo_id}/resolve/{item.revision}/{item.path}" for b in mirrors()]


def dest_of(item: DownloadFile, base: Path) -> Path:
    return (base / item.destination).resolve()


def options(item: DownloadFile, base: Path) -> dict:
    dest = dest_of(item, base)
    o: dict = {"dir": str(dest.parent), "out": dest.name, "split": str(MAX),
         "max-connection-per-server": str(MAX), "min-split-size": "8M",
         "checksum": f"sha-256={item.sha256}", **FILE_OPTS}
    if os.environ.get("HF_TOKEN"):
        o["header"] = [f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
    return o


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _up(api) -> bool:
    try:
        api.get_stats()
        return True
    except Exception:
        return False


def daemon():
    """aria2p client for 127.0.0.1:PORT; starts aria2c detached if none is listening."""
    import aria2p  # lazy: only the server path needs it

    STATE.mkdir(parents=True, exist_ok=True)
    secret_file = STATE / "secret"
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(16))
        secret_file.chmod(0o600)
    secret = secret_file.read_text().strip()
    api = aria2p.API(aria2p.Client(host="http://127.0.0.1", port=PORT, secret=secret, timeout=10))
    if _up(api):
        return api
    exe = shutil.which("aria2c")
    if not exe:
        raise RuntimeError("aria2c not found on PATH")
    session = STATE / "session.txt"
    session.touch()
    (STATE / "stats").touch()
    subprocess.Popen(
        [exe, "--enable-rpc", f"--rpc-listen-port={PORT}", f"--rpc-secret={secret}",
         f"--input-file={session}", f"--save-session={session}", "--save-session-interval=15",
         "--file-allocation=falloc", "--continue=true", "--uri-selector=adaptive",
         f"--server-stat-of={STATE}/stats", f"--server-stat-if={STATE}/stats",
         "--max-concurrent-downloads=4", "--disk-cache=64M", f"--log={STATE}/aria2.log",
         "--log-level=warn"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)  # detached: the queue survives MCP restarts via the session file
    for _ in range(100):
        time.sleep(0.1)
        if _up(api):
            return api
    raise RuntimeError(f"aria2c RPC did not come up on :{PORT} (see {STATE}/aria2.log)")


def brief(dl) -> dict:
    f = dl.files[0] if dl.files else None
    return {"gid": dl.gid, "status": dl.status, "path": str(f.path) if f else None,
            "pct": round(dl.progress, 1), "bytes": dl.completed_length, "total": dl.total_length,
            "MBps": round(dl.download_speed / 1e6, 2), "conns": dl.connections,
            "mirrors": len({u["uri"] for u in f.uris}) if f else 0, "error": dl.error_message or None}


# ---- tools ------------------------------------------------------------------------------------
def t_catalog(group: str | None = None) -> dict:
    cat = DownloadCatalog.load(CATALOG)
    groups = [group] if group else list(cat.groups)
    return {"mirrors": mirrors(), "groups": {
        g: {"files": len(cat.groups[g]),
            "GB": round(sum(f.size_bytes for f in cat.files_for_group(g)) / 1e9, 2)} for g in groups}}


def t_fetch(group: str | None = None, files: list | None = None, base_dir: str | None = None) -> dict:
    items = list(DownloadCatalog.load(CATALOG).files_for_group(group)) if group else []
    items += [DownloadFile.from_mapping(f"adhoc-{i}", v) for i, v in enumerate(files or [])]
    if not items:
        raise ValueError("give a catalog group or files[]")
    base = Path(base_dir).expanduser().resolve() if base_dir else BASE.resolve()
    api = daemon()
    live = {str(f.path): dl.gid for dl in api.get_downloads() if dl.is_active or dl.is_waiting
            for f in dl.files}
    out = []
    for it in items:
        dest = dest_of(it, base)
        if str(dest) in live:
            out.append({"path": str(dest), "gid": live[str(dest)], "state": "queued"})
            continue
        partial = dest.with_name(dest.name + ".aria2").exists()
        if dest.is_file() and not partial:
            ok = dest.stat().st_size == it.size_bytes and sha256(dest) == it.sha256
            out.append({"path": str(dest),
                        "state": "present" if ok else "conflict: exists but differs from catalog"})
            continue
        dl = api.add_uris(uris(it), options(it, base))
        out.append({"path": str(dest), "gid": dl.gid, "state": "resumed" if partial else "added",
                    "mirrors": len(uris(it))})
    return {"downloads": out}


def t_status(gids: list | None = None) -> dict:
    api = daemon()
    dls = [api.get_download(g) for g in gids] if gids else api.get_downloads()
    g = api.client.get_global_option()
    return {"file-allocation": g.get("file-allocation"), "continue": g.get("continue"),
            "MBps": round(int(api.client.get_global_stat()["downloadSpeed"]) / 1e6, 2),
            "downloads": [brief(d) for d in dls]}


def t_control(gid: str, action: str) -> dict:
    dl = daemon().get_download(gid)
    ok = {"pause": dl.pause, "resume": dl.resume, "remove": dl.remove}[action]()
    return {"gid": gid, "action": action, "ok": bool(ok)}


FILE = {"type": "object", "required": list(FIELDS), "properties": {
    k: {"type": "integer" if k == "size_bytes" else "string"} for k in FIELDS}}
TOOLS = {
    "hf_catalog": (t_catalog, "List downloads.json groups with file counts, GB and active mirrors.",
                   {"group": {"type": "string"}}),
    "hf_fetch": (t_fetch, "Queue a catalog group and/or explicit files into aria2 (<=16 mirrors, "
                          "16 connections, falloc, continue, sha-256). Skips present and queued files.",
                 {"group": {"type": "string"}, "files": {"type": "array", "items": FILE},
                  "base_dir": {"type": "string"}}),
    "hf_status": (t_status, "Progress for all or given gids, plus the live file-allocation and continue settings.",
                  {"gids": {"type": "array", "items": {"type": "string"}}}),
    "hf_control": (t_control, "Pause, resume or remove one download by gid.",
                   {"gid": {"type": "string"}, "action": {"enum": ["pause", "resume", "remove"]}}),
}


# ---- MCP stdio --------------------------------------------------------------------------------
def handle(msg: dict) -> dict | None:
    mid, method, p = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:
        return None  # notification
    if method == "initialize":
        res = {"protocolVersion": p.get("protocolVersion", "2025-06-18"),
               "capabilities": {"tools": {}}, "serverInfo": {"name": "turbofit-hf-aria2", "version": "0.2.0"}}
    elif method == "ping":
        res = {}
    elif method == "tools/list":
        res = {"tools": [{"name": n, "description": d, "inputSchema": {"type": "object", "properties": s}}
                         for n, (_, d, s) in TOOLS.items()]}
    elif method == "tools/call":
        try:
            out, err = TOOLS[p["name"]][0](**(p.get("arguments") or {})), False
        except Exception as e:  # tool failures are results, not protocol errors
            out, err = {"error": f"{type(e).__name__}: {e}"}, True
        res = {"content": [{"type": "text", "text": json.dumps(out, indent=1, default=str)}], "isError": err}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": res}


def main() -> None:
    cmd = sys.argv[1:]
    if cmd and cmd[0] in ("catalog", "fetch", "status"):  # small CLI for scripts and humans
        fn = {"catalog": lambda: t_catalog(*cmd[1:2]), "fetch": lambda: t_fetch(cmd[1]),
              "status": lambda: t_status(cmd[1:] or None)}[cmd[0]]
        print(json.dumps(fn(), indent=1, default=str))
        return
    for line in sys.stdin:
        if line.strip():
            reply = handle(json.loads(line))
            if reply:
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
