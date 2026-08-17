"""A full `diffrax.diffeqsolve` under `quaxify`.

`diffeqsolve` allocates its `SaveAt` output buffer with `jnp.full`, so the
buffer is a plain array no matter what the state is. A later `lax.cond` then
puts a slice of that buffer opposite the carried `Value`, and the two branches
disagree on structure. `cond_quax` used to reject that outright, which made
`diffeqsolve` unreachable under `quaxify` for every state type; it now falls
back to `materialise`, so any `Value` that materialises gets through.

The state therefore arrives back as a plain array -- the buffer erased it --
but the solve runs and the numbers are right, with no bridging rules
registered at all. `test_diffrax_unitful.py` covers the other side: a `Value`
that refuses to materialise still stops here, which is what it is for.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import quax


diffrax = pytest.importorskip("diffrax")


class Dense(quax.ArrayValue):
    """An `ArrayValue` that materialises freely, as most real ones do."""

    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self) -> jax.Array:
        return self.array

    def aval(self) -> jax.core.ShapedArray:
        return jax.typeof(self.array)


def _solve(y0):
    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    return diffrax.diffeqsolve(
        term,
        diffrax.Euler(),
        t0=0.0,
        t1=1.0,
        dt0=0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
    ).ys


def test_diffeqsolve_completes_under_quaxify():
    """The full solve runs and agrees with plain JAX."""
    y0 = jnp.array([1.0])
    expected = _solve(y0)

    got = quax.quaxify(_solve)(Dense(y0))

    assert jnp.allclose(got, expected)
