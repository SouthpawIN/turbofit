from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from turbofit_runtime.benchmark_stage import (
    BenchmarkSuite,
    ResourceSample,
    _chat_completion,
    build_prompt,
    compact_samples,
    evaluate,
    summarize,
    verify_evidence,
)


ROOT = Path(__file__).parents[1]


def test_stage_suite_is_strict_and_covers_every_evidence_axis() -> None:
    suite = BenchmarkSuite.load(ROOT / "benchmarks" / "stage-v1.json")

    assert {case.category for case in suite.cases} == {"quality", "context", "throughput"}
    assert len({case.id for case in suite.cases}) == len(suite.cases)


def test_passkey_prompt_is_deterministic_and_places_key_away_from_edges() -> None:
    spec = {"kind": "passkey", "target_words": 1000, "passkey": "739184"}

    prompt = build_prompt(spec)

    assert prompt == build_prompt(spec)
    assert 900 <= len(prompt.split()) <= 1100
    assert 100 < prompt.index("739184") < len(prompt) - 100


def test_exact_evaluation_does_not_accept_explanatory_text() -> None:
    validator = {"kind": "exact", "value": "391"}

    assert evaluate(" 391\n", validator)
    assert evaluate("<think>private trace</think>\n391", validator)
    assert not evaluate("The answer is 391.", validator)


def test_summary_uses_real_usage_and_resource_peaks() -> None:
    cases = [
        {"category": "quality", "passed": True, "elapsed_seconds": 2.0, "usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        {"category": "context", "passed": False, "elapsed_seconds": 4.0, "usage": {"prompt_tokens": 100, "completion_tokens": 4}},
        {"category": "throughput", "passed": None, "elapsed_seconds": 5.0, "usage": {"prompt_tokens": 15, "completion_tokens": 50}},
    ]
    samples = [
        ResourceSample(100.0, 1000, (2000, 3000), 400),
        ResourceSample(101.0, 1500, (2500, 3500), 600),
    ]

    result = summarize(cases, samples)

    assert result["quality_pass_rate"] == 1.0
    assert result["context_pass_rate"] == 0.0
    assert result["effective_output_tokens_per_second"] == 10.0
    assert result["peak_system_ram_used_mib"] == 1500
    assert result["peak_gpu_memory_used_mib"] == [2500, 3500]
    assert result["peak_process_rss_mib"] == 600


def test_resource_compaction_preserves_peaks_and_endpoints() -> None:
    samples = [
        ResourceSample(timestamp=float(i) / 4, system_ram_used_mib=100 + i, gpu_memory_used_mib=(i, 10 - i), process_rss_mib=200 + i)
        for i in range(10)
    ]

    compacted = compact_samples(samples, minimum_interval_seconds=1.0)

    assert compacted[0] == samples[0]
    assert compacted[-1] == samples[-1]
    assert max(sample.gpu_memory_used_mib[0] for sample in compacted) == 9
    assert max(sample.gpu_memory_used_mib[1] for sample in compacted) == 10
    assert len(compacted) < len(samples)


def test_chat_completion_forwards_recorded_request_options(monkeypatch) -> None:
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fake_urlopen(request, timeout):
        seen.update(json.loads(request.data))
        return Response(b'{"choices":[{"message":{"content":"391"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    _chat_completion(
        base_url="http://example/v1",
        model="candidate",
        prompt="question",
        max_tokens=16,
        api_key=None,
        timeout_seconds=1,
        request_options={"chat_template_kwargs": {"enable_thinking": False}},
    )

    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


def test_recorded_smoke_evidence_hash_is_reproducible() -> None:
    evidence = json.loads(
        (ROOT / "references" / "results" / "benchmark-smoke-hardware-48gb.json").read_text()
    )

    assert evidence["status"] == "pass"
    assert evidence["resource_failures"] == []
    assert evidence["summary"]["peak_gpu_memory_used_mib"]
    assert verify_evidence(evidence)


def test_suite_rejects_unknown_validator(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "benchmarks" / "stage-v1.json").read_text())
    raw["cases"][0]["validator"]["kind"] = "semantic-vibes"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="validator"):
        BenchmarkSuite.load(path)
