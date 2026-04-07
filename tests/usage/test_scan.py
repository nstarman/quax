import jax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import kilograms, meters, Unitful


def _outer_fn(const, init, xs):
    def _body_fn(carry, x):
        return carry + const + x, (carry * x, carry * (const + x))

    res = jax.lax.scan(_body_fn, init=init, xs=xs)
    return res


def test_scan_basic():
    const = Unitful(jnp.asarray(2.0), {meters: 1})
    init = Unitful(jnp.asarray(1.0), {meters: 1})
    xs = Unitful(jnp.asarray([10.0, 20.0]), {meters: 1})
    final_carry, (ys1, ys2) = quax.quaxify(_outer_fn)(const, init, xs)
    assert final_carry.array == 1 + 2 + 10 + 2 + 20
    assert final_carry.units == {meters: 1}
    assert ys1.array[0] == 1 * 10
    assert ys1.array[1] == (1 + 2 + 10) * 20
    assert ys1.units == {meters: 2}
    assert ys2.array[0] == 1 * (2 + 10)
    assert ys2.array[1] == (1 + 2 + 10) * (2 + 20)
    assert ys2.units == {meters: 2}


def test_scan_different_units():
    const = Unitful(jnp.asarray(2.0), {meters: 1})
    init = Unitful(jnp.asarray(1.0), {meters: 1})
    xs = Unitful(jnp.asarray([10.0, 20.0]), {kilograms: 1})
    with pytest.raises(Exception):
        quax.quaxify(_outer_fn)(const, init, xs)


def test_scan_jit():
    const = Unitful(jnp.asarray(2.0), {meters: 1})
    init = Unitful(jnp.asarray(1.0), {meters: 1})
    xs = Unitful(jnp.asarray([10.0, 20.0]), {meters: 1})
    final_carry, (ys1, ys2) = quax.quaxify(jax.jit(_outer_fn))(const, init, xs)
    assert final_carry.array == 1 + 2 + 10 + 2 + 20
    assert final_carry.units == {meters: 1}


def test_scan_vmap():
    const = Unitful(jnp.asarray([0.0, 3.0]), {meters: 1})
    init = Unitful(jnp.asarray([1.0, 0.0]), {meters: 1})
    xs = Unitful(
        jnp.repeat(jnp.arange(2, 13, 2)[None, :] * 1.0, 2, axis=0), {meters: 1}
    )
    vmap_fn = jax.vmap(quax.quaxify(_outer_fn))
    final_carry, (ys1, ys2) = vmap_fn(const, init, xs)
    assert final_carry.shape == (2,)
    assert ys1.shape == (2, xs.shape[1])
    assert ys2.shape == (2, xs.shape[1])

    assert final_carry.array[0] == 1 + xs.array[0].sum()
    assert final_carry.array[1] == xs.array[1].sum() + 3 * xs.shape[1]

    assert final_carry.units == {meters: 1}
    assert ys1.units == ys2.units == {meters: 2}


def test_scan_grad():
    const = Unitful(jnp.asarray(2.0), {meters: 1})
    init = Unitful(jnp.asarray(1.0), {meters: 1})
    xs = Unitful(jnp.asarray([10.0, 20.0]), {meters: 1})

    dummy = (
        Unitful(jnp.asarray(1.0), {meters: 1}),
        Unitful(jnp.asarray(1.0), {meters: 1}),
        Unitful(jnp.asarray([1.0, 1.0]), {meters: 1}),
    )

    primals = (const, init, xs)
    p_out, t_out = jax.jvp(quax.quaxify(_outer_fn), primals, dummy)

    direct_out = quax.quaxify(_outer_fn)(const, init, xs)

    assert jnp.array_equal(p_out[0].array, direct_out[0].array)
    assert t_out[1][0].units == {meters: 2}
