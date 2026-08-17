import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import meters, seconds, Unitful


def test_integer_pow_exponentiates_array():
    """`integer_pow` must raise the array to the power, not just scale the units.

    Regression: the handler scaled `units` by `y` but returned `x.array`
    unchanged, so `Unitful(a, {m: 1}) ** 2` gave the array `a` (not `a ** 2`)
    with units `m ** 2` — a silent wrong value.
    """
    x = Unitful(jnp.array([2.0, 3.0]), meters)

    out = quax.quaxify(lambda a: a**2)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 2}
    assert jnp.array_equal(out.array, jnp.array([4.0, 9.0]))


def test_integer_pow_units_and_values_agree():
    """Cube a velocity-like quantity: both value and units are exponentiated."""
    v = Unitful(jnp.array(2.0), {meters: 1, seconds: -1})

    out = quax.quaxify(lambda a: a**3)(v)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 3, seconds: -3}
    assert jnp.array_equal(out.array, jnp.array(8.0))


def test_copy_preserves_units():
    """`jnp.copy` keeps the units of its operand."""
    x = Unitful(jnp.array([2.0, 3.0]), meters)

    out = quax.quaxify(jnp.copy)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.array([2.0, 3.0]))


def test_select_n_requires_matching_units():
    """`jnp.where` between two Unitfuls keeps units, and rejects a mismatch."""
    x = Unitful(jnp.array([2.0, 3.0]), meters)
    y = Unitful(jnp.array([4.0, 5.0]), meters)
    pred = jnp.array([True, False])

    out = quax.quaxify(jnp.where)(pred, x, y)
    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.array([2.0, 5.0]))

    bad = Unitful(jnp.array([4.0, 5.0]), seconds)
    with pytest.raises(ValueError, match="units"):
        quax.quaxify(jnp.where)(pred, x, bad)
