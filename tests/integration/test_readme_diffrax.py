"""Run the README blocks that need an optional dependency.

These live here rather than beside `tests/test_readme_examples.py` so CI's
integration job -- the one that installs the `test-integration` group -- runs
them. Without the dependency they skip rather than silently pass, so a README
example cannot rot unnoticed.
"""

import pytest

from ..readme_blocks import readme_blocks


TAGGED = [(i, req, code) for i, (req, code) in enumerate(readme_blocks()) if req]


def test_readme_has_tagged_blocks():
    """Guard the tag convention itself.

    If the marker syntax changes, `TAGGED` silently empties and every test
    below vanishes rather than failing.
    """
    assert TAGGED, "no `<!-- test: requires ... -->` blocks found in README.md"


@pytest.mark.parametrize(
    ("index", "requires", "code"), TAGGED, ids=[f"block{i}" for i, _, _ in TAGGED]
)
def test_readme_optional_dependency_block(index, requires, code):
    """A tagged block runs standalone, once its dependency is installed."""
    pytest.importorskip(requires)

    try:
        exec(compile(code, f"README.md[block {index}]", "exec"), {})  # noqa: S102
    except Exception as exc:
        msg = f"README.md block {index} (requires {requires}) failed: {exc!r}\n\n{code}"
        raise AssertionError(msg) from exc
