import jax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import kilograms, meters, Unitful

from ..unit.myarray import DenseArray


def _outer_fn(a: jax.Array, b: jax.Array, c: jax.Array, pred: bool | jax.Array):
    def _true_fn(a: jax.Array):
        return a + b

    def _false_fn(a: jax.Array):
        return a + c

    res = jax.lax.cond(pred, _true_fn, _false_fn, a)
    return res


@pytest.mark.parametrize("wrap", [lambda f: f, jax.jit], ids=["eager", "jit"])
def test_cond_basic(wrap):
    a = Unitful(jnp.asarray(1.0), {meters: 1})
    b = Unitful(jnp.asarray(2.0), {meters: 1})
    c = Unitful(jnp.asarray(10.0), {meters: 1})

    res = quax.quaxify(wrap(_outer_fn))(a, b, c, False)
    assert res.array == 11
    assert res.units == {meters: 1}

    res = quax.quaxify(wrap(_outer_fn))(a, b, c, True)
    assert res.array == 3
    assert res.units == {meters: 1}


def test_cond_different_units():
    a = Unitful(jnp.asarray([1.0]), {meters: 1})
    b = Unitful(jnp.asarray([2.0]), {meters: 1})
    c = Unitful(jnp.asarray([10.0]), {kilograms: 1})

    with pytest.raises(Exception):
        quax.quaxify(_outer_fn)(a, b, c, False)


def test_cond_different_out_trees():
    def _outer_fn(a: jax.Array, b: jax.Array, c: jax.Array, pred: bool | jax.Array):
        def _true_fn(a: jax.Array):
            return a + b

        def _false_fn(a: jax.Array):
            return a * c

        res = jax.lax.cond(pred, _true_fn, _false_fn, a)
        return res

    a = Unitful(jnp.asarray([1.0]), {meters: 1})
    b = Unitful(jnp.asarray([2.0]), {meters: 1})
    c = Unitful(jnp.asarray([10.0]), {meters: 1})

    with pytest.raises(Exception):
        quax.quaxify(_outer_fn)(a, b, c, False)


def test_cond_switch():
    def _outer_fn(index: int, a: jax.Array, b: jax.Array, c: jax.Array):
        def _fn0(a: jax.Array):
            return a + b

        def _fn1(a: jax.Array):
            return a + c

        def _fn2(a: jax.Array):
            return a + b + c

        res = jax.lax.switch(index, (_fn0, _fn1, _fn2), a)
        return res

    a = Unitful(jnp.asarray([1.0]), {meters: 1})
    b = Unitful(jnp.asarray([2.0]), {meters: 1})
    c = Unitful(jnp.asarray([10.0]), {meters: 1})

    res = quax.quaxify(_outer_fn)(0, a, b, c)
    assert res.array == 3
    assert res.units == {meters: 1}

    res = quax.quaxify(_outer_fn)(1, a, b, c)
    assert res.array == 11
    assert res.units == {meters: 1}

    res = quax.quaxify(_outer_fn)(2, a, b, c)
    assert res.array == 13
    assert res.units == {meters: 1}


def test_cond_vmap():
    a = Unitful(jnp.arange(1), {meters: 1})
    b = Unitful(jnp.asarray(2), {meters: 1})
    c = Unitful(jnp.arange(2, 13, 2), {meters: 1})
    vmap_fn = jax.vmap(_outer_fn, in_axes=(None, None, 0, None))

    res = quax.quaxify(vmap_fn)(a, b, c, True)
    assert (res.array == a.array + b.array).all()
    assert res.units == {meters: 1}

    res = quax.quaxify(vmap_fn)(a, b, c, False)
    assert (res.array.ravel() == a.array.ravel() + c.array.ravel()).all()  # type: ignore
    assert res.units == {meters: 1}


def test_cond_grad_closure():
    x = Unitful(jnp.asarray(2.0), {meters: 1})
    dummy = Unitful(jnp.asarray(1.0), {meters: 1})

    def outer_fn(
        outer_var: jax.Array,
        dummy: jax.Array,
        pred: bool | jax.Array,
    ):
        def _true_fn_grad(a: jax.Array):
            return a + outer_var

        def _false_fn_grad(a: jax.Array):
            return a + outer_var * 2

        def _outer_fn_grad(a: jax.Array):
            return jax.lax.cond(pred, _true_fn_grad, _false_fn_grad, a)

        primals = (outer_var,)
        tangents = (dummy,)
        p_out, t_out = jax.jvp(_outer_fn_grad, primals, tangents)
        return p_out, t_out

    p, t = quax.quaxify(outer_fn)(x, dummy, True)
    assert p.array == 4
    assert p.units == {meters: 1}
    assert t.array == 1
    assert t.units == {meters: 1}

    p, t = quax.quaxify(outer_fn)(x, dummy, False)
    assert p.array == 6
    assert p.units == {meters: 1}
    assert t.array == 1
    assert t.units == {meters: 1}


def _mixed_branches(x, pred):
    """One branch carries the `Value`, the other a plain array."""
    return jax.lax.cond(pred, lambda: x, lambda: jnp.zeros(3))


def test_cond_mismatched_branches_materialise():
    """Branches disagreeing on `Value`-ness fall back to `materialise`.

    This is the documented behaviour of `quax.Value.default` for any primitive
    with no applicable rule; `cond_p` should not be an exception to it.
    """
    x = jnp.arange(3.0)

    for pred in (True, False):
        got = quax.quaxify(_mixed_branches)(DenseArray(x), jnp.array(pred))

        assert not isinstance(got, quax.ArrayValue)
        assert jnp.array_equal(got, _mixed_branches(x, jnp.array(pred)))


def test_cond_mismatched_branches_reports_refusal():
    """A type that refuses to materialise says so, rather than "same pytree"."""
    x = Unitful(jnp.arange(3.0), meters)

    with pytest.raises(ValueError, match="Refusing to materialise"):
        quax.quaxify(_mixed_branches)(x, jnp.array(True))
