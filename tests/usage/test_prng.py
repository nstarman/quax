import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

import quax
import quax.examples.prng as prng


def test_uniform():
    key = prng.ThreeFry(0)
    prng.uniform(key)


def test_normal():
    key = prng.ThreeFry(0)
    prng.normal(key)


def test_cannot_add():
    key = prng.ThreeFry(0)
    with pytest.raises(TypeError):
        key + 1  # pyright: ignore

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
