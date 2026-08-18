"""Extract the runnable code blocks from `README.md`.

The README's examples are the first thing anyone runs, so they are executed
rather than trusted. Two consumers share this: `tests/test_readme_examples.py`
runs the blocks that need only quax's own dependencies, and
`tests/integration/test_readme_diffrax.py` runs the ones that do not.

A block that needs an optional dependency is tagged in the markdown source
with a comment that renders as nothing:

    <!-- test: requires diffrax -->
    ```python
    ...
    ```

Tagged blocks must stand alone -- they run in a fresh namespace, since the
suite that runs them is not the suite that ran everything above.
"""

import re
from pathlib import Path


README = Path(__file__).parents[1] / "README.md"

_BLOCK = re.compile(
    r"(?:<!-- test: requires (?P<requires>[\w-]+) -->\n)?^```python\n(?P<code>.*?)^```",
    re.DOTALL | re.MULTILINE,
)


def readme_blocks() -> list[tuple[str | None, str]]:
    """Return `(requires, code)` for every ```python block, in document order."""
    text = README.read_text(encoding="utf-8")
    found = [(m.group("requires"), m.group("code")) for m in _BLOCK.finditer(text)]
    assert found, f"no ```python blocks found in {README} -- has the README moved?"
    return found
