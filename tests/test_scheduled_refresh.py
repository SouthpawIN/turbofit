from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scheduled_refresh_runs_discovery_campaign_and_list_pipeline() -> None:
    text = (ROOT / "scripts" / "scheduled-refresh").read_text()

    for command in (
        "research/discover_huggingface.py",
        "research/discover_model_news.py",
        "research/discover_api_models.py",
        "scripts/update-model-intel",
        "scripts/turbofit-discovery-queue",
        "scripts/turbofit-benchmark-orchestrator",
        "scripts/turbofit-promote-list-winner",
        "scripts/turbofit-list",
    ):
        assert command in text
    assert "nvidia-smi --query-compute-apps=pid" not in text
    assert "--once" in text
