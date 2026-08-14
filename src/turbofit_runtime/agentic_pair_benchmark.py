"""Joint production-path benchmark for Turbofit main/auxiliary pairs."""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "turbofit.agentic-pair-suite/v1"
EVIDENCE_SCHEMA = "turbofit.agentic-pair-evidence/v1"


@dataclass(frozen=True)
class AgenticCase:
    id: str
    request: str
    tools: tuple[Mapping[str, Any], ...]
    expected_tool: Mapping[str, Any]
    tool_result: Mapping[str, Any]
    final_instruction: str
    final_validator: Mapping[str, Any]


@dataclass(frozen=True)
class AgenticSuite:
    name: str
    revision: str
    cases: tuple[AgenticCase, ...]
    identity: str

    @classmethod
    def load(cls, path: str | Path) -> "AgenticSuite":
        source = Path(path)
        raw_bytes = source.read_bytes()
        payload = json.loads(raw_bytes)
        if set(payload) != {"schema", "name", "revision", "cases"} or payload["schema"] != SCHEMA:
            raise ValueError("invalid agentic pair suite root")
        cases = []
        for index, raw in enumerate(payload["cases"]):
            expected = {"id", "request", "tools", "expected_tool", "tool_result", "final_instruction", "final_validator"}
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise ValueError(f"invalid agentic case at index {index}")
            case = AgenticCase(
                id=str(raw["id"]), request=str(raw["request"]), tools=tuple(raw["tools"]),
                expected_tool=raw["expected_tool"], tool_result=raw["tool_result"],
                final_instruction=str(raw["final_instruction"]), final_validator=raw["final_validator"],
            )
            _validate_case(case)
            cases.append(case)
        if not payload["name"] or not payload["revision"] or not cases:
            raise ValueError("agentic suite must be named, versioned, and non-empty")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("agentic case ids must be unique")
        return cls(
            name=str(payload["name"]), revision=str(payload["revision"]), cases=tuple(cases),
            identity="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        )


def _validate_case(case: AgenticCase) -> None:
    if not case.id or not case.request or not case.final_instruction or not case.tools:
        raise ValueError("agentic case fields must be non-empty")
    if set(case.expected_tool) != {"name", "arguments"}:
        raise ValueError("expected_tool must contain name and arguments")
    if not isinstance(case.expected_tool["arguments"], Mapping):
        raise ValueError("expected tool arguments must be an object")
    if set(case.final_validator) != {"kind", "value"} or case.final_validator["kind"] != "exact":
        raise ValueError("agentic final validator must be exact")


def _post_chat(
    *, base_url: str, model: str, messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None, timeout: float = 600.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = list(tools)
        body["tool_choice"] = "required"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer turbofit-local"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:1000]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("agentic endpoint returned a non-object")
    return payload


def _message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        value = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("agentic response has no assistant message") from exc
    if not isinstance(value, Mapping):
        raise ValueError("agentic assistant message is invalid")
    return value


def _extract_tool_call(response: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    message = _message(response)
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("auxiliary response has no tool call")
    function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
    if not isinstance(function, Mapping):
        raise ValueError("auxiliary tool call has no function")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError("auxiliary tool name is missing")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, Mapping):
        raise ValueError("auxiliary tool arguments are not an object")
    return name, dict(arguments)


def _final_content(response: Mapping[str, Any]) -> str:
    value = _message(response).get("content")
    if not isinstance(value, str):
        raise ValueError("main response has no text content")
    return value.strip()


def _timing(response: Mapping[str, Any]) -> dict[str, float]:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in timings.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    }


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    aux = sum(item.get("aux_passed") is True for item in cases)
    main = sum(item.get("main_passed") is True for item in cases)
    both = sum(item.get("aux_passed") is True and item.get("main_passed") is True for item in cases)
    return {
        "tasks_total": total,
        "tasks_passed": both,
        "aux_tool_accuracy": 0.0 if not total else aux / total,
        "main_synthesis_accuracy": 0.0 if not total else main / total,
        "score": 0.0 if not total else (aux + main) / (2 * total),
    }


def run_suite(
    *, suite: AgenticSuite, gateway_base_url: str, configuration_id: str,
    main_model: str, auxiliary_model: str, production_recipe_sha256: str,
    output_path: str | Path, timeout: float = 600.0,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in suite.cases:
        started = time.monotonic()
        aux_response: dict[str, Any] | None = None
        main_response: dict[str, Any] | None = None
        aux_error: str | None = None
        main_error: str | None = None
        tool_name = ""
        tool_arguments: Mapping[str, Any] = {}
        try:
            aux_response = _post_chat(
                base_url=gateway_base_url.rstrip("/") + "/aux/v1",
                model=auxiliary_model,
                messages=({"role": "user", "content": case.request},),
                tools=case.tools,
                timeout=timeout,
            )
            tool_name, tool_arguments = _extract_tool_call(aux_response)
            aux_passed = (
                tool_name == case.expected_tool["name"]
                and dict(tool_arguments) == dict(case.expected_tool["arguments"])
            )
        except Exception as exc:
            aux_error = f"{type(exc).__name__}: {exc}"
            aux_passed = False
        try:
            main_prompt = (
                f"User request: {case.request}\n"
                f"Auxiliary selected tool: {tool_name or '<none>'}\n"
                f"Tool arguments: {json.dumps(dict(tool_arguments), sort_keys=True)}\n"
                f"Tool result: {json.dumps(dict(case.tool_result), sort_keys=True)}\n"
                f"{case.final_instruction}"
            )
            main_response = _post_chat(
                base_url=gateway_base_url.rstrip("/") + "/main/v1",
                model=main_model,
                messages=({"role": "user", "content": main_prompt},),
                timeout=timeout,
            )
            final_content = _final_content(main_response)
            main_passed = final_content == str(case.final_validator["value"])
        except Exception as exc:
            main_error = f"{type(exc).__name__}: {exc}"
            final_content = ""
            main_passed = False
        cases.append({
            "id": case.id,
            "aux_passed": aux_passed,
            "main_passed": main_passed,
            "expected_tool": dict(case.expected_tool),
            "observed_tool": {"name": tool_name, "arguments": dict(tool_arguments)},
            "expected_final": case.final_validator["value"],
            "observed_final": final_content,
            "aux_error": aux_error,
            "main_error": main_error,
            "aux_response": aux_response,
            "main_response": main_response,
            "aux_timings": _timing(aux_response or {}),
            "main_timings": _timing(main_response or {}),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        })
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "suite": {"name": suite.name, "revision": suite.revision, "identity": suite.identity},
        "configuration_id": configuration_id,
        "main_model": main_model,
        "auxiliary_model": auxiliary_model,
        "production_recipe_sha256": production_recipe_sha256,
        "gateway_base_url": gateway_base_url.rstrip("/"),
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence
