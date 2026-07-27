#!/usr/bin/env python3
"""Download TurboFit GGUF models with resume, retry, and SHA-256 verification.

Usage:
    python download-models.py [--base-dir <path>]

Default base directory:
    Windows: %USERPROFILE%\\.turbohaul\\models
    Linux:   ~/.turbohaul/models
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

DEFAULT_BASE = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    ".turbohaul", "models",
)

MODELS = [
    {
        "name": "Bonsai-27B-Q1_0.gguf",
        "url": "https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/f10afb355f104535e3e3e98cf7ab7795c72bd292/Bonsai-27B-Q1_0.gguf",
        "subdir": "prism-ml--Bonsai-27B-gguf",
        "sha256": "17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0",
        "size": 3803452480,
    },
    {
        "name": "grm-2.6-plus-0628-Q4_K_M-reasoning-imat.gguf",
        "url": "https://huggingface.co/DAXZEIT/GRM-2.6-Plus-0628-MTP-reasoning-i1-GGUF/resolve/cc2ed138ba38ac7d1db051c210b19843e00687e2/grm-2.6-plus-0628-Q4_K_M-reasoning-imat.gguf",
        "subdir": "DAXZEIT--GRM-2.6-Plus-0628-MTP-reasoning-i1-GGUF",
        "sha256": "268cfdb6df2c73a8a3d8591c86e52d72f4386d1dcd4ef7d2d259638df02c6c25",
        "size": 16810713984,
    },
]

MAX_RETRIES = 10
CHUNK = 8 * 1024 * 1024  # 8MB


def sha256_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_with_resume(model, base_dir):
    """Download a file with HTTP Range resume and retry on connection drops."""
    dest = os.path.join(base_dir, model["subdir"], model["name"])
    tmp = dest + ".tmp"
    expected = model["size"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Already complete and verified?
    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        print(f"[CHECK] {model['name']} exists ({expected} bytes), verifying SHA...")
        if sha256_file(dest) == model["sha256"]:
            print(f"[OK] {model['name']} SHA-256 verified!")
            return True
        print("[WARN] SHA mismatch on final file, re-downloading")
        os.unlink(dest)

    for attempt in range(1, MAX_RETRIES + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0

        if have >= expected:
            print(f"\n[VERIFY] {model['name']} download complete ({have} bytes), checking SHA...")
            if sha256_file(tmp) == model["sha256"]:
                os.replace(tmp, dest)
                print(f"[OK] {model['name']} SHA-256 verified and saved!")
                return True
            else:
                print("[CORRUPT] SHA mismatch, deleting and restarting from scratch")
                os.unlink(tmp)
                have = 0
                continue

        pct = have * 100 // expected if expected else 0
        print(f"\n[ATTEMPT {attempt}/{MAX_RETRIES}] {model['name']}: "
              f"resuming from {have/1024**3:.2f}/{expected/1024**3:.2f} GB ({pct}%)")

        headers = {"User-Agent": "turbofit-dl/1.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"

        req = urllib.request.Request(model["url"], headers=headers)
        start = time.time()
        downloaded_this_attempt = 0

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = resp.status
                if have > 0 and status == 200:
                    print("  Server sent 200 (not 206), restarting download")
                    have = 0
                    mode = "wb"
                elif status == 206:
                    mode = "ab"
                else:
                    mode = "wb"

                with open(tmp, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_this_attempt += len(chunk)
                        total = have + downloaded_this_attempt
                        elapsed = time.time() - start
                        speed = downloaded_this_attempt / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        pct = total * 100 // expected
                        print(f"\r  {pct:3d}% | {total/1024**3:.2f}/{expected/1024**3:.2f} GB "
                              f"| {speed:.0f} MB/s", end="", flush=True)

        except Exception as exc:
            elapsed = time.time() - start
            speed = downloaded_this_attempt / elapsed / 1024 / 1024 if elapsed > 0 else 0
            print(f"\n  [DROP] Connection lost after {downloaded_this_attempt/1024**2:.0f} MB "
                  f"({speed:.0f} MB/s): {exc}")
            print("  Will retry in 3 seconds...")
            time.sleep(3)
            continue

        total = have + downloaded_this_attempt
        if total >= expected:
            print("\n[VERIFY] Checking SHA-256...")
            if sha256_file(tmp) == model["sha256"]:
                os.replace(tmp, dest)
                print(f"[OK] {model['name']} SHA-256 verified and saved!")
                return True
            else:
                print("[CORRUPT] SHA mismatch, restarting")
                os.unlink(tmp)
                continue
        else:
            print(f"\n  [INCOMPLETE] Got {total}/{expected} bytes, retrying...")
            time.sleep(2)
            continue

    print(f"[FAIL] {model['name']} failed after {MAX_RETRIES} attempts")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TurboFit GGUF models")
    parser.add_argument("--base-dir", default=DEFAULT_BASE,
                        help=f"Model storage directory (default: {DEFAULT_BASE})")
    args = parser.parse_args()

    print(f"Model directory: {args.base_dir}")
    ok = True
    for m in MODELS:
        if not download_with_resume(m, args.base_dir):
            ok = False
    print("\n" + "=" * 60)
    print("ALL_DONE" if ok else "SOME_FAILED")
    print("=" * 60)
    sys.exit(0 if ok else 1)
