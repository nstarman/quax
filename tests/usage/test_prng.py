import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

import quax
import quax.examples.prng as prng
import quax.examples.prng._core as prng_core
from quax._compat import JAX_GE_0_10_2


def test_uniform():
    key = prng.ThreeFry(0)
    prng.uniform(key)


def test_normal():
    key = prng.ThreeFry(0)
    prng.normal(key)


def test_normal_complex():
    """`normal` supports complex dtypes (as `jax.random.normal` does).

    Regression: the dtype guard tested `jnp.floating`, which excludes complex,
    so any complex dtype raised before reaching the (fully written) complex
    branch -- making that branch dead code.
    """
    key = prng.ThreeFry(0)
    out = prng.normal(key, shape=(5,), dtype=jnp.complex64)
    assert out.dtype == jnp.complex64
    assert out.shape == (5,)


def test_normal_rejects_integer():
    key = prng.ThreeFry(0)
    with pytest.raises(ValueError):
        prng.normal(key, shape=(2,), dtype=jnp.int32)


def test_uniform_array_bounds():
    """`uniform` accepts array-valued `minval`/`maxval` (as `jax.random.uniform`).

    Regression: the bounds were broadcast to an all-ones shape of the wrong
    rank, so any non-scalar bound raised instead of giving a per-element range.
    """
    key = prng.ThreeFry(0)
    minval = jnp.array([0.0, 10.0, 20.0])
    maxval = jnp.array([1.0, 11.0, 21.0])

    # Per-element bounds.
    out = prng.uniform(key, shape=(3,), minval=minval, maxval=maxval)
    assert out.shape == (3,)
    assert bool((out >= minval).all() and (out < maxval).all())

    # Bounds broadcast against a higher-rank output shape.
    out2 = prng.uniform(key, shape=(4, 3), minval=minval, maxval=maxval)
    assert out2.shape == (4, 3)
    assert bool((out2 >= minval).all() and (out2 < maxval).all())


def test_cannot_add():
    key = prng.ThreeFry(0)
    with pytest.raises(TypeError):
        key + 1

    @jax.jit
    def run(key):
        return key + 1

    with pytest.raises(TypeError):
        run(key)


def test_where():
    pred1 = jnp.array(True)
    pred2 = jnp.array(False)
    key1 = prng.ThreeFry(0)
    key2 = prng.ThreeFry(1)

    @jax.jit
    @quax.quaxify
    def run(pred, key1, key2):
        return jnp.where(pred, key1, key2)

    assert key1 != key2
    assert run(pred1, key1, key2) == key1
    assert run(pred2, key1, key2) == key2


def test_brownian():
    @jax.jit
    def run(key):
        def body(carry, _):
            cumval, key = carry
            new_key, subkey = prng.split(key)
            val = prng.normal(subkey)
            new_cumval = cumval + val
            new_carry = new_cumval, new_key
            return new_carry, cumval

        _, cumvals = lax.scan(body, (0.0, key), xs=None, length=10)
        return cumvals

    run(prng.ThreeFry(0))


def test_threefry_backend_import():
    expected_module_name = (
        "jax._src.random.threefry2x32" if JAX_GE_0_10_2 else "jax._src.prng"
    )
    assert prng_core._jax_threefry.__name__ == expected_module_name
