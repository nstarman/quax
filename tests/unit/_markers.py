"""Shared pytest marks for the `test_lax`/`test_numpy` function-coverage tables.

These were previously redefined identically in each of test_lax/test_jax_array.py,
test_lax/test_myarray.py, test_numpy/test_jax_array.py, and test_numpy/test_myarray.py.
"""

import pytest
from packaging.version import Version

from quax._compat import JAX_VERSION


mark_todo = pytest.mark.skip(reason="TODO")
mark_nomd = pytest.mark.xfail(reason="Can't be supported with MD on primitives")
xfail_quax58 = pytest.mark.xfail(
    reason="https://github.com/patrick-kidger/quax/issues/58"
)
skip_removed_jax_0_10_0 = pytest.mark.skipif(
    JAX_VERSION >= Version("0.10"),
    reason="removed in JAX v0.10.0",
)
xfail_deprecated_jax_0_9_0 = (
    pytest.mark.xfail(
        Version("0.9") <= JAX_VERSION < Version("0.10"),
        raises=DeprecationWarning,
        reason="deprecated in JAX v0.9.0",
        strict=True,
    ),
    # Promote DeprecationWarning to an error so the xfail `raises` check works;
    # warnings.warn() alone won't trigger raises= without this.
    pytest.mark.filterwarnings("error::DeprecationWarning"),
)
