"""PIER agent adapter that resolves each trial's Docker bridge gateway."""
from __future__ import annotations

import json
import shlex
from urllib.parse import urlsplit, urlunsplit

from pier.agents.installed.mini_swe_agent import MiniSweAgent  # type: ignore[reportMissingImports]

from .pier_limits import agent_step_limit_flag


class TurbofitMiniSWEAgent(MiniSweAgent):
    """Route local OpenAI-compatible traffic through the active trial bridge."""

    def __init__(self, step_limit: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._turbofit_step_limit = step_limit

    def _build_config_flags(self, *, custom_config_path: str | None = None) -> str:
        return super()._build_config_flags(
            custom_config_path=custom_config_path,
        ) + agent_step_limit_flag(self._turbofit_step_limit)

    async def run(self, instruction, environment, context) -> None:
        result = await self.exec_as_agent(
            environment,
            command=(
                "python3 -c \"import socket,struct; "
                "route=next(line.split() for line in open('/proc/net/route') "
                "if line.split()[1]=='00000000'); "
                "print(socket.inet_ntoa(struct.pack('<L',int(route[2],16))))\""
            ),
            timeout_sec=30,
        )
        gateway = result.stdout.strip()
        if not gateway:
            raise RuntimeError("PIER container has no discoverable default bridge gateway")
        source = self._get_env("OPENAI_BASE_URL") or self._get_env("OPENAI_API_BASE")
        if not source:
            raise RuntimeError("Turbofit PIER agent requires an explicit OpenAI base URL")
        parsed = urlsplit(source)
        port = f":{parsed.port}" if parsed.port else ""
        routed = urlunsplit((parsed.scheme, gateway + port, parsed.path, parsed.query, parsed.fragment))
        self._extra_env["OPENAI_BASE_URL"] = routed
        self._extra_env["OPENAI_API_BASE"] = routed
        evidence = json.dumps({
            "schema": "turbofit.pier-container-route/v1",
            "bridge_gateway": gateway,
            "provider_base_url": routed,
        }, sort_keys=True)
        await self.exec_as_agent(
            environment,
            command=f"printf '%s\\n' {shlex.quote(evidence)} > /logs/agent/turbofit-route.json",
            timeout_sec=30,
        )
        await super().run(instruction, environment, context)
