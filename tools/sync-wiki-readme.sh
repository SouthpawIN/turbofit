#!/usr/bin/env bash
# Sync the TurboFit repo README from the wiki hub (single source of truth).
# Usage: tools/sync-wiki-readme.sh [wiki_hub_path]
#
# Vault-only links (raw/..., wiki/..., index.md, .html widgets) are rewritten to
# the bot-wiki `wiki` branch so the public README never carries a dead relative
# link. Added 2026-10-28 after the GitHub consolidation pass.
set -euo pipefail
WIKI_HUB="${1:-$HOME/Documents/Projects/Projects/wiki-topics/turbofit/README.md}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_DIR/README.md"
WIKI_BLOB="https://github.com/SouthpawIN/bot-wiki/blob/wiki/wiki-topics/turbofit"

python3 - "$WIKI_HUB" "$DEST" "$REPO_DIR" "$WIKI_BLOB" <<'PY'
import os, re, sys
hub, dest, repo, blob = sys.argv[1:5]
text = open(hub).read()

def fix(m):
    target = m.group(2)
    if target.startswith(("http", "#", "mailto")):
        return m.group(0)
    if os.path.exists(os.path.join(repo, target.split("#")[0])):
        return m.group(0)
    return "%s%s%s" % (m.group(1), blob + "/" + target, m.group(3))

text = re.sub(r"(\]\()([^)]+)(\))", fix, text)
footer = ("\n\n---\n> _This README is generated from the [TurboFit wiki](%s/README.md) — "
          "the single source of truth. To change it, edit the wiki hub and re-run "
          "`tools/sync-wiki-readme.sh`._\n" % blob)
if "generated from the [TurboFit wiki]" in text:
    text = text.split("\n---\n> _This README")[0]
open(dest, "w").write(text.rstrip() + footer)
print("Synced with link rewrite:", hub, "->", dest)
PY
