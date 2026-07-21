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


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_int_conversion():
    """Test that TypedInt can be converted to jax.Array."""
    from jax._src.literals import TypedInt

    # Create a TypedInt
    ti = TypedInt(5, np.dtype(np.int32))

    # Convert to Array using plum
    arr = plum.convert(ti, jnp.ndarray)

    # Verify value and dtype are preserved
    assert arr.item() == 5
    assert arr.dtype == np.int32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_float_conversion():
    """Test that TypedFloat can be converted to jax.Array."""
    from jax._src.literals import TypedFloat

    # Create a TypedFloat
    tf = TypedFloat(3.14, np.dtype(np.float32))

    # Convert to Array using plum
    arr = plum.convert(tf, jnp.ndarray)

    # Verify value and dtype are preserved
    assert np.isclose(arr.item(), 3.14, rtol=1e-6)
    assert arr.dtype == np.float32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_complex_conversion():
    """Test that TypedComplex can be converted to jax.Array."""
    from jax._src.literals import TypedComplex

    # Create a TypedComplex
    tc = TypedComplex(1 + 2j, np.dtype(np.complex64))

    # Convert to Array using plum
    arr = plum.convert(tc, jnp.ndarray)

    # Verify value and dtype are preserved
    assert arr.item() == (1 + 2j)
    assert arr.dtype == np.complex64


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_int_zero():
    """Test TypedInt conversion with zero value."""
    from jax._src.literals import TypedInt

    ti = TypedInt(0, np.dtype(np.int32))
    arr = plum.convert(ti, jnp.ndarray)

    assert arr.item() == 0
    assert arr.dtype == np.int32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_float_negative():
    """Test TypedFloat conversion with negative value."""
    from jax._src.literals import TypedFloat

    tf = TypedFloat(-2.5, np.dtype(np.float32))
    arr = plum.convert(tf, jnp.ndarray)

    assert np.isclose(arr.item(), -2.5, rtol=1e-6)
    assert arr.dtype == np.float32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_complex_zero():
    """Test TypedComplex conversion with zero value."""
    from jax._src.literals import TypedComplex

    tc = TypedComplex(0j, np.dtype(np.complex64))
    arr = plum.convert(tc, jnp.ndarray)

    assert arr.item() == 0j
    assert arr.dtype == np.complex64


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_int_large_value():
    """Test TypedInt conversion with large value that fits in int32."""
    from jax._src.literals import TypedInt

    # Use a value that fits in int32 (max int32 is 2^31 - 1)
    ti = TypedInt(2**30, np.dtype(np.int32))
    arr = plum.convert(ti, jnp.ndarray)

    assert arr.item() == 2**30
    assert arr.dtype == np.int32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_float_precision():
    """Test TypedFloat conversion preserves precision for float32."""
    from jax._src.literals import TypedFloat

    tf = TypedFloat(1.23456789, np.dtype(np.float32))
    arr = plum.convert(tf, jnp.ndarray)

    # float32 has ~7 decimal digits of precision
    assert np.isclose(arr.item(), 1.23456789, rtol=1e-6)
    assert arr.dtype == np.float32


@pytest.mark.skipif(not JAX_GE_0_8_0, reason="JAX >= 0.8.0 required")
def test_typed_complex_with_imaginary():
    """Test TypedComplex conversion with pure imaginary number."""
    from jax._src.literals import TypedComplex

    tc = TypedComplex(3j, np.dtype(np.complex64))
    arr = plum.convert(tc, jnp.ndarray)

    assert arr.item() == 3j
    assert arr.dtype == np.complex64


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
