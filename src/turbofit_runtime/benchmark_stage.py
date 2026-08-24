"""Evidence-first benchmark stage for local OpenAI-compatible runtimes."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "turbofit.benchmark-stage/v1"
EVIDENCE_SCHEMA = "turbofit.benchmark-evidence/v1"
_CATEGORIES = {"quality", "context", "throughput"}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    category: str
    prompt: str | Mapping[str, Any]
    max_tokens: int
    validator: Mapping[str, Any] | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkCase":
        _exact(value, {"id", "category", "prompt", "max_tokens", "validator"}, "benchmark case")
        item = cls(
            id=str(value["id"]), category=str(value["category"]), prompt=value["prompt"],
            max_tokens=value["max_tokens"], validator=value["validator"],
        )
        if not item.id or item.category not in _CATEGORIES:
            raise ValueError("invalid benchmark case identity")
        if not isinstance(item.prompt, (str, Mapping)):
            raise ValueError("benchmark prompt must be text or a prompt specification")
        if isinstance(item.max_tokens, bool) or not isinstance(item.max_tokens, int) or item.max_tokens <= 0:
            raise ValueError("benchmark max_tokens must be positive")
        if item.validator is not None:
            _validate_validator(item.validator)
        if item.category != "throughput" and item.validator is None:
            raise ValueError("quality and context cases require a validator")
        return item


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    cases: tuple[BenchmarkCase, ...]

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkSuite":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        _exact(raw, {"schema", "name", "cases"}, "benchmark suite")
        if raw["schema"] != SCHEMA:
            raise ValueError("unsupported benchmark suite schema")
        if not isinstance(raw["cases"], list) or not raw["cases"]:
            raise ValueError("benchmark suite cases must be a non-empty list")
        suite = cls(str(raw["name"]), tuple(BenchmarkCase.from_mapping(case) for case in raw["cases"]))
        if not suite.name or len({case.id for case in suite.cases}) != len(suite.cases):
            raise ValueError("benchmark suite name and case ids must be unique")
        return suite


@dataclass(frozen=True)
class ResourceSample:
    timestamp: float
    system_ram_used_mib: int
    gpu_memory_used_mib: tuple[int, ...]
    process_rss_mib: int | None


class ResourceMonitor:
    def __init__(self, process_id: int | None = None, interval: float = 0.25) -> None:
        self.process_id = process_id
        self.interval = interval
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceMonitor":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        self.samples.append(ResourceSample(
            timestamp=time.time(),
            system_ram_used_mib=_system_ram_used_mib(),
            gpu_memory_used_mib=_gpu_memory_used_mib(),
            process_rss_mib=_process_rss_mib(self.process_id),
        ))


def build_prompt(value: str | Mapping[str, Any]) -> str:
    if isinstance(value, str):
        return value
    _exact(value, {"kind", "target_words", "passkey"}, "passkey prompt")
    if value["kind"] != "passkey":
        raise ValueError("unsupported prompt specification")
    target_words = value["target_words"]
    passkey = str(value["passkey"])
    if isinstance(target_words, bool) or not isinstance(target_words, int) or target_words < 256:
        raise ValueError("passkey target_words must be at least 256")
    if not passkey:
        raise ValueError("passkey must not be empty")
    filler = "Local inference keeps private workloads on the host while adaptive placement balances memory pressure. "
    prefix_target = target_words // 2
    words: list[str] = []
    filler_words = filler.split()
    while len(words) < prefix_target:
        words.extend(filler_words)
    prefix = " ".join(words[:prefix_target])
    suffix_words = max(0, target_words - prefix_target)
    suffix = " ".join((filler_words * ((suffix_words // len(filler_words)) + 1))[:suffix_words])
    return (
        f"{prefix}\nThe passkey is {passkey}. Remember it exactly.\n{suffix}\n"
        "What is the passkey? Return only the passkey with no explanation."
    )


def _final_answer(content: str) -> str:
    answer = content.strip()
    while answer.startswith("<think>"):
        closing = answer.find("</think>")
        if closing < 0:
            return ""
        answer = answer[closing + len("</think>"):].strip()
    return answer


def evaluate(content: str, validator: Mapping[str, Any]) -> bool:
    _validate_validator(validator)
    if validator["kind"] == "exact":
        return _final_answer(content) == str(validator["value"])
    raise ValueError("unsupported validator")


def run_benchmark(
    *, suite: BenchmarkSuite, base_url: str, model: str, candidate: str,
    configuration: str, output_path: str | Path, api_key: str | None = None,
    process_id: int | None = None, timeout_seconds: float = 900.0,
    artifact_sha256: str | None = None,
    request_options: Mapping[str, Any] | None = None,
    require_gpu_samples: bool = True,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    all_samples: list[ResourceSample] = []
    for case in suite.cases:
        prompt = build_prompt(case.prompt)
        started = time.monotonic()
        with ResourceMonitor(process_id) as monitor:
            try:
                response = _chat_completion(
                    base_url=base_url, model=model, prompt=prompt,
                    max_tokens=case.max_tokens, api_key=api_key, timeout_seconds=timeout_seconds,
                    request_options=request_options,
                )
                elapsed = time.monotonic() - started
                content = _response_content(response)
                usage = _usage(response)
                timings = _timings(response)
                passed = evaluate(content, case.validator) if case.validator is not None else None
                cases.append({
                    "id": case.id, "category": case.category, "passed": passed,
                    "elapsed_seconds": round(elapsed, 6), "usage": usage,
                    "timings": timings, "response_content": content, "error": None,
                })
            except Exception as exc:
                elapsed = time.monotonic() - started
                cases.append({
                    "id": case.id, "category": case.category, "passed": False,
                    "elapsed_seconds": round(elapsed, 6), "usage": None,
                    "timings": None, "response_content": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        all_samples.extend(monitor.samples)

    compacted_samples = compact_samples(all_samples)
    summary = summarize(cases, compacted_samples)
    request_failures = [case["id"] for case in cases if case["error"] is not None]
    validator_failures = [case["id"] for case in cases if case["passed"] is False]
    resource_failures = []
    resource_warnings = []
    if not summary["peak_gpu_memory_used_mib"]:
        if require_gpu_samples:
            resource_failures.append("gpu-memory-sampling-unavailable")
        else:
            resource_warnings.append("gpu-memory-sampling-unavailable")
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite.name,
        "candidate": candidate,
        "configuration": configuration,
        "endpoint": base_url.rstrip("/"),
        "requested_model": model,
        "request_options": dict(request_options or {}),
        "artifact_sha256": artifact_sha256,
        "hardware": hardware_snapshot(),
        "status": "pass" if not request_failures and not validator_failures and not resource_failures else "fail",
        "request_failures": request_failures,
        "validator_failures": validator_failures,
        "resource_failures": resource_failures,
        "resource_warnings": resource_warnings,
        "summary": summary,
        "cases": cases,
        "resource_samples": [sample.__dict__ for sample in compacted_samples],
    }
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["evidence_sha256"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def verify_evidence(evidence: Mapping[str, Any]) -> bool:
    claimed = evidence.get("evidence_sha256")
    if not isinstance(claimed, str):
        return False
    content = dict(evidence)
    content.pop("evidence_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return claimed == "sha256:" + hashlib.sha256(canonical).hexdigest()


def compact_samples(
    samples: Sequence[ResourceSample], *, minimum_interval_seconds: float = 1.0
) -> list[ResourceSample]:
    if len(samples) <= 2:
        return list(samples)
    kept = [samples[0]]
    last_periodic = samples[0].timestamp
    peak_indices = {
        max(range(len(samples)), key=lambda index: samples[index].system_ram_used_mib)
    }
    process_indices = [
        index for index, sample in enumerate(samples) if sample.process_rss_mib is not None
    ]
    if process_indices:
        peak_indices.add(
            max(process_indices, key=lambda index: samples[index].process_rss_mib or 0)
        )
    gpu_count = max((len(sample.gpu_memory_used_mib) for sample in samples), default=0)
    for gpu in range(gpu_count):
        gpu_indices = [
            index
            for index, sample in enumerate(samples)
            if gpu < len(sample.gpu_memory_used_mib)
        ]
        if gpu_indices:
            peak_indices.add(
                max(gpu_indices, key=lambda index: samples[index].gpu_memory_used_mib[gpu])
            )
    for index, sample in enumerate(samples[1:-1], start=1):
        new_peak = index in peak_indices
        periodic = sample.timestamp - last_periodic >= minimum_interval_seconds
        if new_peak or periodic:
            kept.append(sample)
            if periodic:
                last_periodic = sample.timestamp
    if samples[-1] != kept[-1]:
        kept.append(samples[-1])
    return kept


def summarize(cases: Sequence[Mapping[str, Any]], samples: Sequence[ResourceSample]) -> dict[str, Any]:
    def pass_rate(category: str) -> float | None:
        selected = [case for case in cases if case["category"] == category]
        return None if not selected else sum(case["passed"] is True for case in selected) / len(selected)

    throughput = [case for case in cases if case["category"] == "throughput" and case.get("usage")]
    output_tokens = sum(case["usage"]["completion_tokens"] for case in throughput)
    output_seconds = sum(case["elapsed_seconds"] for case in throughput)
    gpu_count = max((len(sample.gpu_memory_used_mib) for sample in samples), default=0)
    process_values = [sample.process_rss_mib for sample in samples if sample.process_rss_mib is not None]
    ttft_values = [
        case["timings"]["ttft_ms"]
        for case in cases
        if case.get("timings") and case["timings"].get("ttft_ms") is not None
    ]
    return {
        "quality_pass_rate": pass_rate("quality"),
        "context_pass_rate": pass_rate("context"),
        "effective_output_tokens_per_second": None if not output_seconds else round(output_tokens / output_seconds, 6),
        "mean_ttft_ms": None if not ttft_values else round(sum(ttft_values) / len(ttft_values), 6),
        "measured_prompt_tokens": sum(case["usage"]["prompt_tokens"] for case in cases if case.get("usage")),
        "measured_completion_tokens": sum(case["usage"]["completion_tokens"] for case in cases if case.get("usage")),
        "peak_system_ram_used_mib": max((sample.system_ram_used_mib for sample in samples), default=None),
        "peak_gpu_memory_used_mib": [
            max((sample.gpu_memory_used_mib[index] for sample in samples if len(sample.gpu_memory_used_mib) > index), default=None)
            for index in range(gpu_count)
        ],
        "peak_process_rss_mib": max(process_values, default=None),
    }


def hardware_snapshot() -> dict[str, Any]:
    meminfo = _meminfo()
    gpu_names: list[str] = []
    gpu_totals: list[int] = []
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        for line in output.splitlines():
            name, total = line.rsplit(",", 1)
            gpu_names.append(name.strip())
            gpu_totals.append(int(total.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    accelerator_memory_kind = "dedicated"
    if not gpu_names and platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        gpu_names = ["Apple Silicon Metal (unified memory)"]
        gpu_totals = [meminfo.get("MemTotal", 0) // 1024]
        accelerator_memory_kind = "unified"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "system_ram_total_mib": meminfo.get("MemTotal", 0) // 1024,
        "gpu_names": gpu_names,
        "gpu_memory_total_mib": gpu_totals,
        "accelerator_memory_kind": accelerator_memory_kind,
    }


def _chat_completion(
    *, base_url: str, model: str, prompt: str, max_tokens: int,
    api_key: str | None, timeout_seconds: float,
    request_options: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    options = dict(request_options or {})
    reserved = {"model", "messages", "temperature", "max_tokens", "stream"}
    if reserved.intersection(options):
        raise ValueError("request options cannot override benchmark-controlled fields")
    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request_body.update(options)
    payload = json.dumps(request_body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    if not isinstance(result, Mapping):
        raise ValueError("chat completion response must be an object")
    return result


def _response_content(response: Mapping[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("chat completion response has no message content") from exc
    if not isinstance(content, str):
        raise ValueError("chat completion content must be text")
    return content


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("chat completion response has no measured usage")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        raise ValueError("chat completion usage is incomplete")
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _timings(response: Mapping[str, Any]) -> dict[str, float] | None:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        return None
    result: dict[str, float] = {}
    for key in (
        "cache_n",
        "prompt_n",
        "prompt_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_ms",
        "predicted_per_second",
        "predicted_per_token_ms",
    ):
        value = timings.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(value)
    prompt_ms = result.get("prompt_ms")
    token_ms = result.get("predicted_per_token_ms")
    if prompt_ms is not None:
        result["ttft_ms"] = prompt_ms + (token_ms or 0.0)
    return result or None


def _validate_validator(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {"kind", "value"} or value.get("kind") != "exact":
        raise ValueError("unsupported benchmark validator")
    if not isinstance(value.get("value"), str):
        raise ValueError("exact validator value must be text")


def _system_ram_used_mib() -> int:
    values = _meminfo()
    return (values.get("MemTotal", 0) - values.get("MemAvailable", 0)) // 1024


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    if platform.system() != "Darwin":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, rest = line.split(":", 1)
                values[name] = int(rest.strip().split()[0])
        except (OSError, ValueError):
            pass
        return values
    try:
        total_bytes = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        vm_output = subprocess.run(
            ["vm_stat"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        page_size = 4096
        first_line = vm_output.splitlines()[0] if vm_output else ""
        if "page size of" in first_line:
            page_size = int(first_line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
        pages: dict[str, int] = {}
        for line in vm_output.splitlines()[1:]:
            if ":" not in line:
                continue
            name, raw = line.split(":", 1)
            pages[name.strip()] = int(raw.strip().rstrip("."))
        available_pages = sum(
            pages.get(name, 0)
            for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        )
        values["MemTotal"] = total_bytes // 1024
        values["MemAvailable"] = available_pages * page_size // 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return values


def _gpu_memory_used_mib() -> tuple[int, ...]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        return tuple(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            return (_system_ram_used_mib(),)
        return ()


def _process_rss_mib(process_id: int | None) -> int | None:
    if process_id is None:
        return None
    try:
        for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        if platform.system() == "Darwin":
            try:
                output = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(process_id)],
                    check=True, capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                return int(output) // 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
    return None


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields do not match schema")
