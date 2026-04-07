import jax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import kilograms, meters, Unitful


def _outer_fn(a: jax.Array, b: jax.Array, c: jax.Array):
    # body has b as static argument
    def _body_fn(a: jax.Array):
        return a + b

    # cond has c as static argument
    def _cond_fn(a: jax.Array):
        return (a < c).squeeze()

    res = jax.lax.while_loop(
        body_fun=_body_fn,
        cond_fun=_cond_fn,
        init_val=a,
    )
    return res


def test_while_basic():
    a = Unitful(jnp.asarray(1.0), {meters: 1})
    b = Unitful(jnp.asarray(2.0), {meters: 1})
    c = Unitful(jnp.asarray(10.0), {meters: 1})
    res = quax.quaxify(_outer_fn)(a, b, c)
    assert res.array == 11
    assert res.units == {meters: 1}


def test_while_different_units():
    a = Unitful(jnp.asarray([1.0]), {meters: 1})
    b = Unitful(jnp.asarray([2.0]), {meters: 1})
    c = Unitful(jnp.asarray([10.0]), {kilograms: 1})
    with pytest.raises(Exception):
        quax.quaxify(_outer_fn)(a, b, c)


def test_while_jit():
    a = Unitful(jnp.asarray(1.0), {meters: 1})
    b = Unitful(jnp.asarray(2.0), {meters: 1})
    c = Unitful(jnp.asarray(10.0), {meters: 1})
    res = quax.quaxify(jax.jit(_outer_fn))(a, b, c)
    assert res.array == 11
    assert res.units == {meters: 1}


def test_while_vmap():
    a = Unitful(jnp.arange(1), {meters: 1})
    b = Unitful(jnp.asarray(2), {meters: 1})
    c = Unitful(jnp.arange(2, 13, 2), {meters: 1})
    vmap_fn = jax.vmap(_outer_fn, in_axes=(None, None, 0))
    res = quax.quaxify(vmap_fn)(a, b, c)
    for i in range(len(c.array)):  # type: ignore
        assert res.array[i] == c.array[i]  # type: ignore
    assert res.units == {meters: 1}


def test_while_grad_closure():
    x = Unitful(jnp.asarray(2.0), {meters: 1})
    c = Unitful(jnp.asarray(10.0), {meters: 1})
    dummy = Unitful(jnp.asarray(1.0), {meters: 1})

    def outer_fn(outer_var: jax.Array, c: jax.Array, dummy: jax.Array):
        def _body_fn_grad(a: jax.Array):
            return a + outer_var

        def _cond_fn_grad(a: jax.Array):
            return (a < c).squeeze()

        def _outer_fn_grad(a: jax.Array):
            return jax.lax.while_loop(
                body_fun=_body_fn_grad,
                cond_fun=_cond_fn_grad,
                init_val=a,
            )

        primals = (outer_var,)
        tangents = (dummy,)
        p_out, t_out = jax.jvp(_outer_fn_grad, primals, tangents)
        return p_out, t_out

    p, t = quax.quaxify(outer_fn)(x, c, dummy)
    assert p.array == 10
    assert p.units == {meters: 1}
    assert t.array == 1
    assert t.units == {meters: 1}
