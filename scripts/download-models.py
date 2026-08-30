#!/usr/bin/env python3
"""Download pinned Turbofit model groups with resume and SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from turbofit_runtime.downloads import DownloadCatalog, DownloadFile


DEFAULT_BASE = Path(os.environ.get("TURBOFIT_MODEL_ROOT", "~/Models/storage/gguf")).expanduser()
DEFAULT_CATALOG = ROOT / "runtime-profiles" / "downloads.json"
MAX_RETRIES = 10
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_file(item: DownloadFile, destination: Path) -> bool:
    return (
        destination.is_file()
        and destination.stat().st_size == item.size_bytes
        and sha256_file(destination) == item.sha256
    )


def missing_bytes(files: tuple[DownloadFile, ...], base_dir: Path) -> int:
    total = 0
    for item in files:
        destination = base_dir / item.destination
        if not destination.is_file() or destination.stat().st_size != item.size_bytes:
            total += item.size_bytes
    return total


def require_disk_capacity(files: tuple[DownloadFile, ...], base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    needed = missing_bytes(files, base_dir)
    free = shutil.disk_usage(base_dir).free
    reserve = max(2 * 1024**3, needed // 20)
    if needed + reserve > free:
        raise RuntimeError(
            f"insufficient disk space: need {needed + reserve} bytes including reserve, have {free}"
        )


def write_receipt(path: Path, files: tuple[DownloadFile, ...], base_dir: Path) -> None:
    payload = {
        "schema": "turbofit.download-verification/v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "id": item.id,
                "path": str(base_dir / item.destination),
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in files
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_with_resume(item: DownloadFile, base_dir: Path) -> bool:
    destination = base_dir / item.destination
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"[CHECK] {item.id}: verifying {destination}")
        if verify_file(item, destination):
            print(f"[OK] {item.id}")
            return True
        destination.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        have = temporary.stat().st_size if temporary.exists() else 0
        if have > item.size_bytes:
            temporary.unlink()
            have = 0
        if have == item.size_bytes:
            if sha256_file(temporary) == item.sha256:
                os.replace(temporary, destination)
                print(f"[OK] {item.id}")
                return True
            temporary.unlink()
            have = 0

        percent = have * 100 // item.size_bytes
        print(
            f"[ATTEMPT {attempt}/{MAX_RETRIES}] {item.id}: "
            f"{have / 1024**3:.2f}/{item.size_bytes / 1024**3:.2f} GiB ({percent}%)"
        )
        headers = {"User-Agent": "turbofit-downloader/2.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(item.url, headers=headers)
        downloaded = 0
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = have > 0 and response.status == 206
                mode = "ab" if append else "wb"
                if not append:
                    have = 0
                with temporary.open(mode) as handle:
                    while chunk := response.read(CHUNK):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        current = have + downloaded
                        elapsed = max(0.001, time.monotonic() - start)
                        speed = downloaded / elapsed / 1024**2
                        print(
                            f"\r  {current * 100 // item.size_bytes:3d}% | "
                            f"{current / 1024**3:.2f}/{item.size_bytes / 1024**3:.2f} GiB | "
                            f"{speed:.0f} MiB/s",
                            end="",
                            flush=True,
                        )
        except Exception as exc:
            print(f"\n[RETRY] {item.id}: {exc}")
            time.sleep(3)
            continue

        print()
        if temporary.stat().st_size == item.size_bytes and sha256_file(temporary) == item.sha256:
            os.replace(temporary, destination)
            print(f"[OK] {item.id}")
            return True
        if temporary.stat().st_size >= item.size_bytes:
            temporary.unlink()
        time.sleep(2)

    print(f"[FAIL] {item.id}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--group", action="append", dest="groups")
    parser.add_argument(
        "--file", action="append", dest="file_ids",
        help="single file/quant id to download within (or instead of) a group; repeatable",
    )
    parser.add_argument("--list-groups", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    catalog = DownloadCatalog.load(args.catalog)
    if args.list_groups:
        for group in sorted(catalog.groups):
            print(group)
        return 0
    groups = args.groups or (["production-floor"] if not args.file_ids else [])
    selected: list[DownloadFile] = []
    seen: set[str] = set()
    for group in groups:
        for item in catalog.files_for_group(group):
            if item.id not in seen:
                selected.append(item)
                seen.add(item.id)
    for file_id in args.file_ids or []:
        if file_id not in catalog.files:
            known = ", ".join(sorted(catalog.files)) or "(none)"
            raise SystemExit(
                f"error: unknown download file id: {file_id}\n"
                f"Known ids: {known}"
            )
        item = catalog.files[file_id]
        if item.id not in seen:
            selected.append(item)
            seen.add(item.id)
    files = tuple(selected)
    if not files:
        raise SystemExit(
            "error: nothing selected; pass --group <group> and/or --file <file-id>"
        )
    if args.verify_only:
        failures = [item.id for item in files if not verify_file(item, args.base_dir / item.destination)]
        if failures:
            print("UNVERIFIED: " + ", ".join(failures))
            return 1
        if args.receipt:
            write_receipt(args.receipt, files, args.base_dir)
        print("ALL_VERIFIED")
        return 0

    require_disk_capacity(files, args.base_dir)
    success = all(download_with_resume(item, args.base_dir) for item in files)
    if success and args.receipt:
        write_receipt(args.receipt, files, args.base_dir)
    print("ALL_DONE" if success else "SOME_FAILED")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
