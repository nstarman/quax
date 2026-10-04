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

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


REPO_ROOT = Path(__file__).parents[2]
SCAN_DIRS = ("src", "tests")

# A `JAX_GE_0_11_2`-style flag name, or a `Version("0.11.2")` literal on a line
# that also mentions `JAX_VERSION` (so unrelated version literals are ignored).
FLAG_RE = re.compile(r"JAX_GE_(\d+)_(\d+)_(\d+)")
INLINE_RE = re.compile(r'Version\("([0-9][0-9.]*)"\)')


def _jax_floor() -> Version:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    for dep in pyproject["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "jax":
            (spec,) = [s for s in req.specifier if s.operator == ">="]
            return Version(spec.version)
    msg = "no `jax` dependency found in pyproject.toml"
    raise AssertionError(msg)


def _guards() -> list[tuple[Path, int, Version]]:
    found = set()
    for directory in SCAN_DIRS:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path == Path(__file__):
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                versions = [".".join(m) for m in FLAG_RE.findall(line)]
                if "JAX_VERSION" in line:
                    versions += INLINE_RE.findall(line)
                found |= {
                    (path.relative_to(REPO_ROOT), lineno, Version(v)) for v in versions
                }
    return sorted(found, key=lambda g: (str(g[0]), g[1], g[2]))


def test_no_version_guard_below_supported_floor() -> None:
    """Every JAX version guard is still meaningful at the supported floor."""
    floor = _jax_floor()
    guards = _guards()
    assert guards, "found no version guards at all -- the scan is broken"

    dead = [(p, n, v) for p, n, v in _guards() if floor >= v]
    assert not dead, "JAX version guards made dead by the jax>={} floor:\n{}".format(
        floor, "\n".join(f"  {p}:{n}: guards JAX {v}" for p, n, v in dead)
    )


def test_no_hasattr_probing_of_jax_internals() -> None:
    """Compatibility shims name a version rather than probing with `hasattr`."""
    compat = (REPO_ROOT / "src" / "quax" / "_compat.py").read_text()
    assert "hasattr" not in compat, (
        "`quax._compat` probes with `hasattr`; find the JAX release that "
        "introduced the feature and add a `JAX_GE_*` flag instead"
    )


if __name__ == "__main__":
    pytest.main([__file__])
