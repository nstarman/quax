"""A loop body may not change its carry's static metadata.

JAX enforces a stable carry, but only over what its type system sees. A
`Value`'s static metadata is part of its pytree structure rather than its aval,
so a body that changes it used to pass JAX's check and have the change silently
dropped -- returning a correct array wearing the wrong units.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

import quax
from quax.examples.unitful import meters, Unitful


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
    """Squaring changes the units every iteration, which cannot be represented."""
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
    """Doubling keeps the units, so the loop is representable and must still work."""
    x = Unitful(jnp.asarray([2.0]), meters)

    out = quax.quaxify(fn)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}


def test_materialising_carry_is_not_an_error():
    """A body that materialises its carry has fallen back, not changed metadata.

    `MyArrayLike` has no registered rules, so `c + x` materialises it to a plain
    array. The result is honestly plain rather than a mislabelled `Value`, which
    is the documented fallback -- not the failure this check is for.
    """
    import equinox as eqx

    class Plainish(quax.ArrayValue):
        array: jax.Array = eqx.field(converter=jnp.asarray)

        def materialise(self) -> jax.Array:
            return self.array

        def aval(self) -> jax.core.ShapedArray:
            return jax.typeof(self.array)

    def f(carry, xs):
        return lax.scan(lambda c, x: (c + x, None), carry, xs)[0]

    out = quax.quaxify(jax.jit(f))(Plainish(jnp.asarray(1.0)), jnp.arange(3.0))

    assert not isinstance(out, quax.ArrayValue)
    assert jnp.allclose(out, 1.0 + 0.0 + 1.0 + 2.0)
