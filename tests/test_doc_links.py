"""Agent-facing docs are mostly links; a stale one silently misleads the agent.

`AGENTS.md`, the skills, and the prompt files route agents to source and test
files by relative path. Nothing else checks those paths, so a rename leaves the
docs confidently pointing at files that no longer exist -- and an agent reading
`.github/skills/code-review/SKILL.md` has no way to tell.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
DOCS = [
    ROOT / "AGENTS.md",
    *ROOT.glob("skills/*/SKILL.md"),
    *ROOT.glob(".github/skills/*/SKILL.md"),
    *ROOT.glob(".github/prompts/*.prompt.md"),
]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path):
    """Every repo-relative markdown link in the doc points at a real file."""
    broken = [
        href
        for href in LINK.findall(doc.read_text(encoding="utf-8"))
        # Skip external URLs and same-document anchors; keep the path of a
        # link that carries a fragment (`file.md#section`).
        if not href.startswith(("http://", "https://", "mailto:", "#"))
        and not (doc.parent / href.split("#")[0]).exists()
    ]
    assert not broken, f"{doc.relative_to(ROOT)} links to missing files: {broken}"
