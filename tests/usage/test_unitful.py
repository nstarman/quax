import jax.numpy as jnp

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
