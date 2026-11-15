"""Shared pytest configuration.

Disables live Tailscale peer discovery for the whole test session so the
suite is hermetic: tests must never depend on which machines happen to be
reachable on the developer's tailnet. Individual tests that exercise peer
discovery monkeypatch the probe directly.
"""
import os
import sys

os.environ.setdefault("TURBOFIT_PEER_DISCOVERY", "0")

# The built-in Hermes local runtime (``hermes_cli.local_runtime``) lives in the
# Hermes Agent checkout, not in this repo. Make it importable for the suite so
# tests don't depend on an out-of-band PYTHONPATH. Overridable via
# TURBOFIT_HERMES_HOME; no-op when ``hermes_cli`` is already importable.
_HERMES_HOME = os.path.expanduser(
    os.environ.get("TURBOFIT_HERMES_HOME", "~/.hermes/hermes-agent")
)
if os.path.isdir(_HERMES_HOME) and _HERMES_HOME not in sys.path:
    sys.path.insert(0, _HERMES_HOME)
