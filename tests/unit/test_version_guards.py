"""Guard against JAX compatibility shims outliving the supported floor.

Every version guard in the codebase names the JAX release it is there for --
either as a ``JAX_GE_<major>_<minor>_<patch>`` flag from `quax._compat`, or as
an inline ``Version("...")`` compared against ``JAX_VERSION``. Never probe with
``hasattr``: it hides *which* release changed, so the shim can never be
confidently deleted.

This test reads the ``jax>=`` floor from `pyproject.toml` and fails for any
guard at or below it, since such a guard is by then a constant. Bumping the
floor therefore produces a list of exactly the shims to delete.
"""

import operator
import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


# Resolved, so the self-skip in `_guards` is a reliable comparison whatever
# `__file__` looks like under a given pytest import mode, and through symlinked
# checkouts.
SELF = Path(__file__).resolve()
REPO_ROOT = SELF.parents[2]
SCAN_DIRS = ("src", "tests")

# A `JAX_GE_0_11_2`-style flag name. Each is defined as `JAX_VERSION >= ...`, so
# a reference to one is a `>=` guard.
FLAG_RE = re.compile(r"JAX_GE_(\d+)_(\d+)_(\d+)")

# An inline comparison, in either order. Requiring `JAX_VERSION` adjacent to the
# literal is what keeps unrelated `Version("...")` calls out of the results.
INLINE_RE = re.compile(r'JAX_VERSION\s*(>=|<=|>|<)\s*Version\("([0-9][0-9.]*)"\)')
REVERSED_RE = re.compile(r'Version\("([0-9][0-9.]*)"\)\s*(>=|<=|>|<)\s*JAX_VERSION')

# `Version("0.9") <= JAX_VERSION` guards the same releases as
# `JAX_VERSION >= Version("0.9")`, so a reversed comparison is normalised by
# flipping its operator.
FLIP = {">=": "<=", "<=": ">=", ">": "<", "<": ">"}

# When a floor makes each comparison a constant. Given `JAX_VERSION >= floor`,
# `>=` and `<` are constant as soon as the floor *reaches* the guarded version.
# The strict `>` and the inclusive `<=` only become constant once it *passes*:
# at exactly the floor both still discriminate, since JAX may be newer.
DEAD_AT = {">=": operator.ge, "<": operator.ge, ">": operator.gt, "<=": operator.gt}


def _jax_floor() -> Version:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    for dep in pyproject["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "jax":
            (spec,) = [s for s in req.specifier if s.operator == ">="]
            return Version(spec.version)
    msg = "no `jax` dependency found in pyproject.toml"
    raise AssertionError(msg)


def _line_guards(line: str) -> set[tuple[str, Version]]:
    """The `(operator, version)` guards on one line, all normalised to JAX-first."""
    guards = {(">=", Version(".".join(m))) for m in FLAG_RE.findall(line)}
    guards |= {(op, Version(v)) for op, v in INLINE_RE.findall(line)}
    guards |= {(FLIP[op], Version(v)) for v, op in REVERSED_RE.findall(line)}
    return guards


def _guards() -> list[tuple[Path, int, str, Version]]:
    """Every JAX version guard in the codebase, with its location and operator."""
    found = set()
    for directory in SCAN_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path.resolve() == SELF:  # this file's own examples are not guards
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                found |= {
                    (path.relative_to(REPO_ROOT), lineno, op, v)
                    for op, v in _line_guards(line)
                }
    return sorted(found, key=lambda g: (str(g[0]), g[1], g[3]))


def test_no_version_guard_below_supported_floor() -> None:
    """Every JAX version guard is still meaningful at the supported floor."""
    floor = _jax_floor()
    guards = _guards()
    assert guards, "found no version guards at all -- the scan is broken"
    assert not [g for g in guards if g[0] == SELF.relative_to(REPO_ROOT)], (
        "this file's own docstring examples were scanned -- the self-skip broke"
    )

    dead = [(p, n, op, v) for p, n, op, v in guards if DEAD_AT[op](floor, v)]
    assert not dead, "JAX version guards made dead by the jax>={} floor:\n{}".format(
        floor, "\n".join(f"  {p}:{n}: `JAX_VERSION {op} {v}`" for p, n, op, v in dead)
    )


def test_no_hasattr_probing_of_jax_internals() -> None:
    """Compatibility shims name a version rather than probing with `hasattr`."""
    compat = (REPO_ROOT / "src" / "quax" / "_compat.py").read_text()
    assert "hasattr" not in compat, (
        "`quax._compat` probes with `hasattr`; find the JAX release that "
        "introduced the feature and add a `JAX_GE_*` flag instead"
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # Both comparison orders describe the same guarded releases.
        ('if JAX_VERSION >= Version("0.11.2"):', {(">=", Version("0.11.2"))}),
        ('if Version("0.11.2") <= JAX_VERSION:', {(">=", Version("0.11.2"))}),
        ('if JAX_VERSION < Version("0.10"):', {("<", Version("0.10"))}),
        ('if Version("0.10") > JAX_VERSION:', {("<", Version("0.10"))}),
        # A flag name is a `>=` guard by construction.
        ("if JAX_GE_0_9_2:", {(">=", Version("0.9.2"))}),
        # A two-sided range yields both bounds, JAX-first.
        (
            'Version("0.9") <= JAX_VERSION < Version("0.10")',
            {(">=", Version("0.9")), ("<", Version("0.10"))},
        ),
        # A `Version(...)` unrelated to `JAX_VERSION` is not a guard.
        ('if Version(importlib.metadata.version("plum")) >= Version("2.9"):', set()),
    ],
)
def test_line_guards_normalises_both_comparison_orders(
    line: str, expected: set[tuple[str, Version]]
) -> None:
    """Guards are recognised in either order and normalised to JAX-first."""
    assert _line_guards(line) == expected


if __name__ == "__main__":
    pytest.main([__file__])
