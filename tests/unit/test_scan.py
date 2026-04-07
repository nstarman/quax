import jax
import jax.numpy as jnp

import quax

from .myarray import is_myarray, MyArray


def _outer_fn(const, init, xs):
    def _body_fn(carry, x):
        return carry + const + x, (carry * x, carry * (const + x))

    return jax.lax.scan(_body_fn, init=init, xs=xs)


def test_scan_basic():
    const = MyArray(jnp.asarray(2.0))
    init = MyArray(jnp.asarray(1.0))
    xs = MyArray(jnp.asarray([10.0, 20.0]))

    final_carry, (ys1, ys2) = quax.quaxify(_outer_fn)(const, init, xs)

    assert is_myarray(final_carry)
    assert is_myarray(ys1)
    assert is_myarray(ys2)

    assert final_carry.array == 1 + 2 + 10 + 2 + 20
    assert ys1.array[0] == 1 * 10
    assert ys1.array[1] == (1 + 2 + 10) * 20
    assert ys2.array[0] == 1 * (2 + 10)
    assert ys2.array[1] == (1 + 2 + 10) * (2 + 20)


def test_scan_jit():
    const = MyArray(jnp.asarray(2.0))
    init = MyArray(jnp.asarray(1.0))
    xs = MyArray(jnp.asarray([10.0, 20.0]))

    final_carry, (ys1, ys2) = quax.quaxify(jax.jit(_outer_fn))(const, init, xs)

    assert is_myarray(final_carry)
    assert final_carry.array == 1 + 2 + 10 + 2 + 20


def test_scan_vmap():
    # Batch of 2: consts [0, 3], inits [1, 0], xs shape (2, 6)
    const = MyArray(jnp.asarray([0.0, 3.0]))
    init = MyArray(jnp.asarray([1.0, 0.0]))
    xs = MyArray(jnp.repeat(jnp.arange(2, 13, 2)[None, :] * 1.0, 2, axis=0))

    vmap_fn = jax.vmap(quax.quaxify(_outer_fn))
    final_carry, (ys1, ys2) = vmap_fn(const, init, xs)

    assert is_myarray(final_carry)
    assert is_myarray(ys1)
    assert is_myarray(ys2)
    assert final_carry.shape == (2,)
    assert ys1.shape == (2, xs.array.shape[1])
    assert ys2.shape == (2, xs.array.shape[1])


def test_scan_jvp():
    const = MyArray(jnp.asarray(2.0))
    init = MyArray(jnp.asarray(1.0))
    xs = MyArray(jnp.asarray([10.0, 20.0]))

    primals = (const, init, xs)
    tangents = (
        MyArray(jnp.asarray(1.0)),
        MyArray(jnp.asarray(1.0)),
        MyArray(jnp.asarray([1.0, 1.0])),
    )

    p_out, t_out = jax.jvp(quax.quaxify(_outer_fn), primals, tangents)
    direct_out = quax.quaxify(_outer_fn)(const, init, xs)

    assert is_myarray(p_out[0])
    assert is_myarray(t_out[0])
    assert jnp.array_equal(p_out[0].array, direct_out[0].array)
