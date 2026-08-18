"""A full `diffrax.diffeqsolve` under `quaxify`.

`diffeqsolve` allocates its `SaveAt` buffer with `jnp.full`, so a `lax.cond`
downstream puts a plain buffer slice opposite the carried `Value`. That used to
be rejected outright; `cond_quax` now materialises instead, so any `Value` that
materialises gets through -- as a plain array, but with the right numbers, and
with no bridging rules registered. `test_diffrax_unitful.py` covers the type
that refuses.
"""

import jax.numpy as jnp
import pytest

import quax

from ..unit.myarray import DenseArray


diffrax = pytest.importorskip("diffrax")


def _solve(y0):
    term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
    saveat = diffrax.SaveAt(t1=True)
    return diffrax.diffeqsolve(
        term, diffrax.Euler(), 0.0, 1.0, 0.1, y0, saveat=saveat
    ).ys


def test_diffeqsolve_completes_under_quaxify():
    """The full solve runs and agrees with plain JAX."""
    y0 = jnp.array([1.0])

    got = quax.quaxify(_solve)(DenseArray(y0))

    assert jnp.allclose(got, _solve(y0))
