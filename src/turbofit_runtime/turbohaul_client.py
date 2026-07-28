"""Small, typed HTTP client for the Turbohaul Manager v0.7 API."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class TurbohaulClientError(RuntimeError):
    """Base error for Turbohaul transport and response failures."""


class TurbohaulHTTPError(TurbohaulClientError):
    """A non-success response from Turbohaul."""

    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        self.status = status
        self.method = method
        self.path = path
        self.detail = detail
        super().__init__(f"Turbohaul {method} {path} returned HTTP {status}: {detail}")


@dataclass(frozen=True)
class TurbohaulResponse:
    payload: dict[str, Any]
    etag: str | None = None


class TurbohaulClient:
    """Bounded-timeout client for Turbofit's supported Turbohaul operations."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 10.0,
        activation_timeout_s: float = 600.0,
        acquisition_timeout_s: float = 3600.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if activation_timeout_s <= 0:
            raise ValueError("activation_timeout_s must be positive")
        if acquisition_timeout_s <= 0:
            raise ValueError("acquisition_timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.activation_timeout_s = activation_timeout_s
        self.acquisition_timeout_s = acquisition_timeout_s

    def status(self) -> dict[str, Any]:
        return self._request_json("GET", "/status")

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list) or not all(isinstance(item, dict) for item in models):
            raise TurbohaulClientError("Turbohaul GET /api/tags returned invalid models")
        return models

    def show_model(self, name: str) -> dict[str, Any]:
        if not name:
            raise ValueError("name must be non-empty")
        return self._request_json("GET", f"/api/show?{urlencode({'name': name})}")

    def put_manifest(
        self,
        tag: str,
        manifest: dict[str, Any],
        *,
        etag: str | None = None,
    ) -> TurbohaulResponse:
        if not tag:
            raise ValueError("tag must be non-empty")
        headers = {} if etag is None else {"If-Match": etag}
        return self._request(
            "PUT",
            f"/api/manifests/{tag}",
            payload=manifest,
            headers=headers,
        )

    def pull_hf(
        self,
        *,
        repo_id: str,
        filename: str,
        revision: str,
        expected_sha256: str,
    ) -> dict[str, Any]:
        for name, value in (
            ("repo_id", repo_id),
            ("filename", filename),
            ("revision", revision),
            ("expected_sha256", expected_sha256),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        return self._request_json(
            "POST",
            "/api/pull-hf",
            payload={
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "expected_sha256": expected_sha256,
            },
            timeout_s=self.acquisition_timeout_s,
        )

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ValueError("chat payload requires a non-empty model")
        if not isinstance(payload.get("messages"), list) or not payload["messages"]:
            raise ValueError("chat payload requires non-empty messages")
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            timeout_s=self.activation_timeout_s,
        )

    def unload_model(
        self,
        model: str,
        *,
        verification_timeout_s: float = 30.0,
        poll_interval_s: float = 0.25,
    ) -> dict[str, Any]:
        if not model:
            raise ValueError("model must be non-empty")
        if verification_timeout_s < 0:
            raise ValueError("verification_timeout_s must be non-negative")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.chat_completion(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply exactly OK"}],
                "max_tokens": 1,
                "temperature": 0,
                "keep_alive": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        deadline = time.monotonic() + verification_timeout_s
        while True:
            status = self.status()
            if not _model_is_resident(status, model):
                return status
            if time.monotonic() >= deadline:
                raise TurbohaulClientError(
                    f"Turbohaul did not unload model {model!r} within "
                    f"{verification_timeout_s:g}s"
                )
            time.sleep(poll_interval_s)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            method,
            path,
            payload=payload,
            headers=headers,
            timeout_s=timeout_s,
        ).payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> TurbohaulResponse:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request_headers = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s if timeout_s is None else timeout_s) as response:
                raw = response.read()
                etag = response.headers.get("ETag")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TurbohaulHTTPError(exc.code, method, path, detail) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TurbohaulClientError(f"Turbohaul {method} {path} failed: {exc}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TurbohaulClientError(f"Turbohaul {method} {path} returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise TurbohaulClientError(f"Turbohaul {method} {path} returned non-object JSON")
        return TurbohaulResponse(payload=decoded, etag=etag)


def _model_is_resident(status: dict[str, Any], model: str) -> bool:
    def matches(value: Any) -> bool:
        if isinstance(value, str):
            return value == model
        if isinstance(value, dict):
            return any(value.get(key) == model for key in ("model_tag", "model", "name", "tag"))
        return False

    def readable(value: Any) -> bool:
        """Can this entry be compared to a model name at all?"""
        return isinstance(value, str) or (
            isinstance(value, dict)
            and any(key in value for key in ("model_tag", "model", "name", "tag"))
        )

    if matches(status.get("active")) or matches(status.get("idle_hot")):
        return True
    residents = status.get("residents")
    if isinstance(residents, list):
        if any(matches(item) for item in residents):
            return True
        # An empty list, or a list of entries we can read and did not match, is a proven
        # absence. A list whose entries we cannot read is not.
        if not residents or all(readable(item) for item in residents):
            return False
    # A status we cannot read is not a model we have proven absent. `unload_model` treats
    # False as "the model is gone" and returns, so answering False for an unrecognised shape
    # reports a false unload and makes its own timeout unreachable. Stay resident instead and
    # let the caller time out.
    return True
