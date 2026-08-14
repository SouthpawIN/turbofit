from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_model_intelligence_workflow_runs_all_collectors_and_opens_reviewable_pr() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "model-intelligence.yml").read_text())

    assert "schedule" in workflow[True]
    assert workflow["permissions"] == {"contents": "write", "pull-requests": "write"}
    commands = "\n".join(
        step.get("run", "")
        for step in workflow["jobs"]["collect"]["steps"]
        if isinstance(step, dict)
    )
    assert "research/discover_huggingface.py" in commands
    assert "research/discover_model_news.py" in commands
    assert "research/discover_api_models.py" in commands
    assert "research/discover_external_benchmarks.py" in commands
    assert "gh pr create" in commands
    assert "git push --force-with-lease" in commands
