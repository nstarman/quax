"""Agent-facing docs are mostly links; a stale one silently misleads the agent.

`AGENTS.md`, the skills, and the prompt files route agents to source and test
files by relative path. Nothing else checks those paths, so a rename leaves the
docs confidently pointing at files that no longer exist -- and an agent reading
`.github/skills/code-review/SKILL.md` has no way to tell.
"""

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = [
    ROOT / "AGENTS.md",
    *ROOT.glob("skills/*/SKILL.md"),
    *ROOT.glob(".github/skills/*/SKILL.md"),
    *ROOT.glob(".github/prompts/*.prompt.md"),
]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def test_doc_discovery_is_sane():
    """`DOCS` is built at import, so a shrunk glob drops tests instead of failing.

    A wrong `ROOT` surfaces loudly, since the unconditional `AGENTS.md` entry
    then fails to read. A *renamed* `skills/` or `.github/prompts/` does not:
    those are globs, so they would quietly contribute nothing and the link
    checking would shrink to `AGENTS.md` alone, still green.
    """
    assert (ROOT / "pyproject.toml").is_file(), f"`ROOT` is not the repo root: {ROOT}"
    assert len(DOCS) > 1, f"only {DOCS} discovered -- have skills/prompts moved?"


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
