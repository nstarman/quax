"""The quax skill ships example code that agents copy verbatim.

An example that stops running as JAX moves is a real defect, so every
````python```` block in the skill is executed here. Blocks run in order into one
shared namespace, because later snippets build on names bound by earlier ones.
Illustrative fragments that are not meant to run are fenced as ````py```` rather
than ````python````, and are skipped.
"""

import re
from pathlib import Path


SKILL = Path(__file__).parents[1] / "skills" / "quax" / "SKILL.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def test_skill_examples_run():
    """Every ```python block in the skill executes without error."""
    # Explicit encoding: the skill contains non-ASCII punctuation (µ, —, ≤, ×),
    # and `read_text()` would otherwise decode it with the platform default.
    blocks = BLOCK.findall(SKILL.read_text(encoding="utf-8"))
    assert blocks, f"no ```python blocks found in {SKILL} -- has the skill moved?"

    namespace: dict[str, object] = {}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{SKILL.name}[block {i}]", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            msg = f"{SKILL.name} block {i} failed: {exc!r}\n\n{block}"
            raise AssertionError(msg) from exc
