"""A loop carry's structure must settle.

A `Value`'s metadata lives in its pytree structure rather than its aval, so
JAX's carry-stability check cannot see it. A body that changes it used to pass
that check and have the change dropped, returning a correct array wearing the
wrong units.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

import quax
from quax.examples.unitful import meters, Unitful

from ..unit.myarray import DenseArray


def _while_squares(x):
    return lax.while_loop(
        lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] * c[1]), (0, x)
    )[1]


def _fori_squares(x):
    return lax.fori_loop(0, 3, lambda i, c: c * c, x)


def _scan_squares(x):
    return lax.scan(lambda c, _: (c * c, None), x, None, length=2)[0]


@pytest.mark.parametrize(
    "fn", [_while_squares, _fori_squares, _scan_squares], ids=["while", "fori", "scan"]
)
def test_changing_carry_metadata_raises(fn):
    """Squaring takes the units somewhere new on every iteration."""
    x = Unitful(jnp.asarray([2.0]), meters)

    with pytest.raises(TypeError, match="carry changed structure"):
        quax.quaxify(fn)(x)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: lax.while_loop(
            lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] + c[1]), (0, x)
        )[1],
        lambda x: lax.fori_loop(0, 3, lambda i, c: c + c, x),
        lambda x: lax.scan(lambda c, _: (c + c, None), x, None, length=2)[0],
    ],
    ids=["while", "fori", "scan"],
)
def test_stable_carry_metadata_still_runs(fn):
    """Doubling keeps the units, so the loop is representable."""
    x = Unitful(jnp.asarray([2.0]), meters)

    out = quax.quaxify(fn)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}


def test_materialising_carry_is_not_an_error():
    """Losing the wrapper is the documented fallback, not the failure above."""

    # `DenseArray` has no registered rules, so `c + x` materialises it.
    def f(carry, xs):
        return lax.scan(lambda c, x: (c + x, None), carry, xs)[0]

    out = quax.quaxify(jax.jit(f))(DenseArray(jnp.asarray(1.0)), jnp.arange(3.0))

    assert not isinstance(out, quax.ArrayValue)
    assert jnp.allclose(out, 1.0 + 0.0 + 1.0 + 2.0)


def test_materialised_while_carry_is_not_rewrapped():
    """`while_loop` labels its result from the settled body, not the input.

    It used to unflatten with the input treedef, so a materialised carry came
    back as the `Value` it no longer was.
    """

    def loop(x):
        return lax.while_loop(
            lambda c: c[0] < 3, lambda c: (c[0] + 1, c[1] + 1.0), (0, x)
        )[1]

    out = quax.quaxify(jax.jit(loop))(DenseArray(jnp.asarray([1.0])))

    assert not isinstance(out, quax.ArrayValue)
    assert jnp.allclose(out, 4.0)
