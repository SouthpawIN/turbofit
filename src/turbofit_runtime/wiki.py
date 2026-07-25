"""Generate deterministic wiki views from canonical Turbofit data."""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .profile_io import load_yaml_profile
from .runtime_profile import Turbofile

CLASSES = (8, 16, 24, 48, 96, 200, 300)
README_START = "<!-- turbofit-generated:recommendations:start -->"
README_END = "<!-- turbofit-generated:recommendations:end -->"
CHECKLIST_START = "<!-- turbofit-generated:evidence:start -->"
CHECKLIST_END = "<!-- turbofit-generated:evidence:end -->"


def validate_profile_evidence(profile: Turbofile, identities: set[str]) -> None:
    missing = sorted({rung.evidence for rung in profile.rungs} - identities)
    if missing:
        raise ValueError(f"profile {profile.id} has unresolved evidence: {missing}")


class WikiPublisher:
    def __init__(self, repo_root: str | Path, wiki_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.wiki_root = Path(wiki_root).resolve()
        self.topic_root = self.wiki_root / "topics" / "turbofit"
        self.readme = self.topic_root / "README.md"
        self.checklist = self.topic_root / "main-aux-inference-checklist.md"

    def validate_evidence_index(self, index: Mapping[str, Any]) -> None:
        for identity, raw in index.items():
            if not isinstance(raw, Mapping) or not isinstance(raw.get("source"), str):
                raise ValueError(f"invalid evidence index entry: {identity}")
            source = self._source_path(raw["source"])
            if not source.is_file():
                raise ValueError(f"missing evidence source: {source}")

    def render(self) -> dict[Path, str]:
        profiles = [
            load_yaml_profile(self.repo_root / "runtime-profiles" / f"{gb}gb.yaml")
            for gb in CLASSES
        ]
        measured = self._json(self.repo_root / "runtime-profiles" / "evidence-index.json")
        class_evidence = self._json(
            self.repo_root / "runtime-profiles" / "class-evidence-index.json"
        )
        self.validate_evidence_index(measured)
        self.validate_evidence_index(class_evidence)
        evidence = {**measured, **class_evidence}
        identities = set(evidence)
        for profile in profiles:
            validate_profile_evidence(profile, identities)
        candidates = self._json(self.repo_root / "research" / "candidates.json")
        if candidates.get("schema") != "turbofit.research-candidates/v1":
            raise ValueError("unsupported candidate queue")
        queue = candidates.get("candidates")
        if not isinstance(queue, list):
            raise ValueError("candidate queue must be a list")
        if any(not isinstance(item, Mapping) or item.get("status") != "candidate" for item in queue):
            raise ValueError("wiki cannot publish unchecked/promoted research items")

        readme_section = self._render_readme(profiles, evidence, queue)
        checklist_section = self._render_checklist(profiles, evidence)
        readme_text = self.readme.read_text(encoding="utf-8")
        checklist_text = self.checklist.read_text(encoding="utf-8")
        return {
            self.readme: _replace_generated(
                readme_text, README_START, README_END, readme_section, "## Key Paths"
            ),
            self.checklist: _replace_generated(
                checklist_text,
                CHECKLIST_START,
                CHECKLIST_END,
                checklist_section,
                "## Checklist",
            ),
        }

    def publish(self) -> tuple[Path, ...]:
        rendered = self.render()
        changed: list[Path] = []
        for path, content in rendered.items():
            if path.read_text(encoding="utf-8") == content:
                continue
            _atomic_write(path, content)
            changed.append(path)
        return tuple(changed)

    def _render_readme(
        self,
        profiles: list[Turbofile],
        evidence: Mapping[str, Any],
        candidates: list[Mapping[str, Any]],
    ) -> str:
        lines = [
            README_START,
            "## Generated Adaptive Recommendations",
            "",
            "> Generated from repository Turbofiles and evidence indexes; this page is not runtime authority.",
            "",
            "| Class | Topology | State | Target | Evidence |",
            "|---:|---|---|---|---|",
        ]
        for profile in profiles:
            local = profile.rungs[:-1]
            target = local[0].id if local else "API terminal"
            identity = profile.rungs[0].evidence
            link = self._evidence_link(evidence[identity])
            lines.append(
                f"| {profile.hardware.class_vram_gb} GB | `{profile.hardware.topology}` | "
                f"{profile.policy.recommendation} | `{target}` | {link} |"
            )
        counts = Counter(str(item.get("kind")) for item in candidates)
        summary = ", ".join(f"{kind}: {counts[kind]}" for kind in sorted(counts)) or "empty"
        lines.extend([
            "",
            "### Candidate queue",
            "",
            "Canonical source: `research/candidates.json`. All entries remain candidates until benchmark promotion.",
            "",
            f"Current queue: **{len(candidates)}** ({summary}).",
            "",
            "See [the generated evidence index](main-aux-inference-checklist.md#generated-profile-evidence).",
            README_END,
        ])
        return "\n".join(lines)

    def _render_checklist(
        self, profiles: list[Turbofile], evidence: Mapping[str, Any]
    ) -> str:
        lines = [
            CHECKLIST_START,
            "## Generated Profile Evidence",
            "",
            "> Generated from canonical profiles. [Back to recommendations](README.md#generated-adaptive-recommendations).",
            "",
        ]
        for profile in profiles:
            links = []
            for identity in dict.fromkeys(rung.evidence for rung in profile.rungs):
                links.append(self._evidence_link(evidence[identity]))
            lines.append(
                f"- **{profile.hardware.class_vram_gb} GB / `{profile.hardware.topology}`** "
                f"— `{profile.policy.recommendation}` — {', '.join(links)}"
            )
        lines.append(CHECKLIST_END)
        return "\n".join(lines)

    def _evidence_link(self, raw: Mapping[str, Any]) -> str:
        source = self._source_path(str(raw["source"]))
        try:
            relative = source.relative_to(self.topic_root)
            href = relative.as_posix()
        except ValueError:
            href = source.as_uri()
        return f"[evidence]({href})"

    def _source_path(self, source: str) -> Path:
        path = Path(source).expanduser()
        return path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()

    @staticmethod
    def _json(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON mapping: {path}")
        return data


def _replace_generated(
    text: str, start: str, end: str, section: str, anchor: str
) -> str:
    if start in text or end in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"malformed generated markers: {start}")
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        return before + section + after
    if anchor not in text:
        raise ValueError(f"wiki insertion anchor missing: {anchor}")
    return text.replace(anchor, section + "\n\n" + anchor, 1)


def _atomic_write(path: Path, content: str) -> None:
    """Replace one wiki document without exposing a partial write."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
