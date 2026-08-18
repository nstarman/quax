"""Tests for JAX compatibility features in quax._compat."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import plum
import pytest

from quax._compat import JAX_GE_0_8_0


# Skip all tests if JAX version doesn't support TypedInt
pytestmark = pytest.mark.skipif(
    not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required for TypedInt"
)


@pytest.mark.parametrize(
    ("cls_name", "value", "dtype"),
    [
        ("TypedInt", 5, np.dtype(np.int32)),
        ("TypedFloat", 3.14, np.dtype(np.float32)),
        ("TypedComplex", 1 + 2j, np.dtype(np.complex64)),
        ("TypedInt", 0, np.dtype(np.int32)),
        ("TypedFloat", -2.5, np.dtype(np.float32)),
        ("TypedComplex", 0j, np.dtype(np.complex64)),
        ("TypedInt", 2**30, np.dtype(np.int32)),
        ("TypedFloat", 1.23456789, np.dtype(np.float32)),
        ("TypedComplex", 3j, np.dtype(np.complex64)),
    ],
    ids=[
        "int-basic",
        "float-basic",
        "complex-basic",
        "int-zero",
        "float-negative",
        "complex-zero",
        "int-large-value",
        "float-precision",
        "complex-with-imaginary",
    ],
)
def test_typed_conversion(cls_name, value, dtype):
    """TypedInt/TypedFloat/TypedComplex convert to jax.Array, preserving
    value and dtype. Floats use an isclose check (float32 precision);
    ints/complex compare exactly."""
    literals = importlib.import_module("jax._src.literals")
    cls = getattr(literals, cls_name)

    ti = cls(value, dtype)
    arr = plum.convert(ti, jnp.ndarray)

    if cls_name == "TypedFloat":
        assert np.isclose(arr.item(), value, rtol=1e-6)
    else:
        assert arr.item() == value
    assert arr.dtype == dtype


# ---------------------------------------------------------------------------
# `jax.Array` faithfulness (plum dispatch caching)
#
# `quax._compat` marks `jax.Array` as "faithful" so that `plum` can cache method
# resolution. Newer JAX gives `jax.Array` a custom metaclass ``__instancecheck__``
# that `plum` would otherwise treat as non-faithful, which disables the
# resolution cache for any `plum` function registering a `jax.Array` method --
# including the global ``convert`` run on the return value of every dispatched
# function. These tests guard against that regression.


def _plum_attr(candidates, attr):
    """Fetch a plum attribute whose private module path varies across versions."""
    for modname in candidates:
        try:
            return getattr(importlib.import_module(modname), attr)
        except (ImportError, AttributeError):  # pragma: no cover
            continue
    pytest.skip(f"plum.{attr} is not importable in this plum version")


def test_jax_array_marked_faithful():
    """`quax._compat` marks `jax.Array` faithful, and `plum` recognises it."""
    assert getattr(jax.Array, "__faithful__", False) is True
    is_faithful = _plum_attr(("plum.type", "plum._type"), "is_faithful")
    assert is_faithful(jax.Array) is True


def _convert_dispatcher():
    """Return plum's internal ``convert`` dispatcher, or skip if relocated."""
    return _plum_attr(("plum.promotion", "plum._promotion"), "_convert")


def test_convert_resolver_stays_faithful():
    """The `jax.Array` `convert` methods must not disable plum's resolver cache.

    Without ``jax.Array.__faithful__ = True``, the ``TypedInt/Float/Complex ->
    jax.Array`` conversions registered in ``quax._compat`` make plum's global
    ``convert`` resolver non-faithful, disabling caching for every conversion.
    """
    convert = _convert_dispatcher()
    # Trigger resolution of the (lazily registered) conversion methods.
    plum.convert(jnp.asarray(1.0), jax.Array)
    assert convert._resolver.is_faithful


def test_convert_resolution_is_cached():
    """A faithful resolver actually caches resolution (no re-resolve per call)."""
    convert = _convert_dispatcher()
    if not hasattr(convert, "clear_cache"):  # pragma: no cover
        pytest.skip("plum.Function has no clear_cache in this version")
    convert.clear_cache()
    assert len(convert._cache) == 0
    plum.convert(jnp.asarray(1.0), jax.Array)
    assert len(convert._cache) > 0
