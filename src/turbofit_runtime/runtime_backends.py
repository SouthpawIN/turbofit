"""Runtime adapters for modular CUDA, ROCm, Metal, CPU, and Lemonade serving."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Mapping, Sequence


Transport = Callable[[str, str, dict | None, float], tuple[int, dict]]


def llama_environment(
    backend: str,
    *,
    devices: str = "0",
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if backend not in {"cuda", "rocm", "metal", "cpu"}:
        raise ValueError(f"unsupported llama backend: {backend}")
    env = dict(os.environ if base is None else base)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(name, None)
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = devices
    elif backend == "rocm":
        env["HIP_VISIBLE_DEVICES"] = devices
        env["ROCR_VISIBLE_DEVICES"] = devices
    elif backend == "metal":
        env["GGML_METAL"] = "1"
    return env


class LemonadeClient:
    """Bounded OpenAI/Lemonade API client for an existing Lemonade Server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:13305/api/v1",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: Transport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Lemonade base_url must use http or https")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._transport = transport or self._request

    def health(self) -> dict:
        return self._call("GET", "/health")

    def models(self) -> list[dict]:
        response = self._call("GET", "/models")
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Lemonade models response is missing data")
        return [item for item in data if isinstance(item, dict)]

    def system_info(self) -> dict:
        return self._call("GET", "/system-info")

    def pull(
        self,
        model_name: str,
        *,
        checkpoint: str | None = None,
        recipe: str | None = None,
    ) -> dict:
        payload: dict[str, object] = {"model_name": _model_name(model_name)}
        if checkpoint is not None or recipe is not None:
            if not checkpoint or not recipe or not model_name.startswith("user."):
                raise ValueError("custom Lemonade models require user.* name, checkpoint, and recipe")
            payload.update({"checkpoint": checkpoint, "recipe": recipe})
        return self._call("POST", "/pull", payload)

    def load(
        self,
        model_name: str,
        *,
        context: int | None = None,
        backend: str | None = None,
        llama_args: Sequence[str] = (),
        pinned: bool = False,
        save_options: bool = False,
    ) -> dict:
        payload: dict[str, object] = {"model_name": _model_name(model_name)}
        if context is not None:
            if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
                raise ValueError("context must be a positive integer")
            payload["ctx_size"] = context
        if backend is not None:
            if backend not in {"vulkan", "rocm", "metal", "cpu"}:
                raise ValueError(f"unsupported Lemonade llama.cpp backend: {backend}")
            payload["llamacpp_backend"] = backend
        if llama_args:
            payload["llamacpp_args"] = " ".join(str(argument) for argument in llama_args)
        if pinned:
            payload["pinned"] = True
        if save_options:
            payload["save_options"] = True
        return self._call("POST", "/load", payload)

    def unload(self, model_name: str) -> dict:
        return self._call("POST", "/unload", {"model_name": _model_name(model_name)})

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        status, response = self._transport(method, self.base_url + path, payload, self.timeout_s)
        if status < 200 or status >= 300:
            raise RuntimeError(f"Lemonade request failed with HTTP {status}: {response}")
        if not isinstance(response, dict):
            raise RuntimeError("Lemonade response must be a JSON object")
        if response.get("status") == "error":
            raise RuntimeError(f"Lemonade operation failed: {response}")
        return response

    def _request(self, method: str, url: str, payload: dict | None, timeout: float) -> tuple[int, dict]:
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                error = json.load(exc)
            except (ValueError, OSError):
                error = {"error": exc.read().decode(errors="replace")}
            return exc.code, error
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"error": str(exc)}


def _model_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(character.isspace() for character in value):
        raise ValueError("model_name must be a non-empty token")
    return value
