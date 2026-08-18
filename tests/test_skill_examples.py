"""The quax skill ships example code that agents copy verbatim.

An example that stops running as JAX moves is a real defect, so every
````python```` block in the skill is executed here. Blocks run in order into one
shared namespace, because later snippets build on names bound by earlier ones.
Illustrative fragments that are not meant to run are fenced as ````py```` rather
than ````python````, and are skipped.

Each block gets its own parametrized test case (so a failure in block 3 doesn't
hide whether blocks 4+ would have passed), but all blocks still execute in
order into one shared namespace via a module-scoped fixture -- later blocks
depend on names bound by earlier ones, so they can't run independently.
"""

import re
from pathlib import Path

import pytest


SKILL = Path(__file__).parents[1] / "skills" / "quax" / "SKILL.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
# Explicit encoding: the skill contains non-ASCII punctuation (µ, —, ≤, ×),
# and `read_text()` would otherwise decode it with the platform default.
BLOCKS = BLOCK.findall(SKILL.read_text(encoding="utf-8"))


def test_skill_has_python_blocks():
    """Sanity check that the extraction regex still matches -- has the skill moved?"""
    assert BLOCKS, f"no ```python blocks found in {SKILL}"


@pytest.fixture(scope="module")
def block_outcomes() -> list[Exception | None]:
    """Execute every block once, in order, into one shared namespace.

    Returns the exception raised by each block (or None on success) so
    individual tests can report per-block results without re-running anything.
    """
    namespace: dict[str, object] = {}
    outcomes: list[Exception | None] = []
    for i, block in enumerate(BLOCKS):
        try:
            exec(compile(block, f"{SKILL.name}[block {i}]", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            outcomes.append(exc)
        else:
            outcomes.append(None)
    return outcomes


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_skill_example_block(block_outcomes: list[Exception | None], i: int):
    """Each ```python block in the skill executes without error."""
    exc = block_outcomes[i]
    if exc is not None:
        msg = f"{SKILL.name} block {i} failed: {exc!r}\n\n{BLOCKS[i]}"
        raise AssertionError(msg) from exc
