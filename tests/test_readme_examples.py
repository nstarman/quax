"""Run the README's examples.

`testpaths` has listed `README` since before this file existed, but the path
does not match `README.md` and no `--doctest-glob` is set, so nothing in the
README was ever executed. These run it.

Blocks share one namespace and run in document order, so a later snippet may
use names an earlier one bound. Blocks tagged with an optional dependency are
left to `tests/integration/test_readme_diffrax.py`.
"""

from .readme_blocks import readme_blocks


def test_readme_examples_run():
    """Every untagged ```python block in the README executes without error."""
    blocks = [(i, code) for i, (req, code) in enumerate(readme_blocks()) if not req]
    assert blocks, "no untagged README blocks -- has the tagging convention changed?"

    namespace: dict[str, object] = {}
    for i, code in blocks:
        try:
            exec(compile(code, f"README.md[block {i}]", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            msg = f"README.md block {i} failed: {exc!r}\n\n{code}"
            raise AssertionError(msg) from exc
