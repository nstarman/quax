import jax
import jax.core
import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import kilograms, meters, Unitful


class _Pair(quax.ArrayValue):
    """An `ArrayValue` that flattens to *two* array leaves.

    The shipped example types (`Unitful`, `MyArray`, ...) all flatten to a
    single array leaf, so they never exercised the scan path where quaxifying
    the body changes the flat operand count. `_Pair` does, with field-wise
    arithmetic so a scan body built from `+`/`*` has predictable per-field
    semantics.
    """

    a: jax.Array
    b: jax.Array

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.a.shape, self.a.dtype)

    def materialise(self) -> jax.Array:
        return self.a


@quax.register(jax.lax.add_p)
def add_pair_pair(x: _Pair, y: _Pair) -> _Pair:
    return _Pair(x.a + y.a, x.b + y.b)


@quax.register(jax.lax.mul_p)
def mul_pair_pair(x: _Pair, y: _Pair) -> _Pair:
    return _Pair(x.a * y.a, x.b * y.b)


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


def test_scan_multileaf_value():
    """A Value that flattens to multiple array leaves as const/carry/xs.

    Regression: the quaxified body has more flat operands than the original, so
    the (pre-0.11) re-bind forwarded a stale `linear` tuple sized to the old
    operand count, and JAX's scan rules raised `safe_zip(...)`.
    """
    const = _Pair(jnp.asarray(1.0), jnp.asarray(10.0))
    init = _Pair(jnp.asarray(0.0), jnp.asarray(0.0))
    xs = _Pair(jnp.arange(1.0, 4.0), jnp.arange(1.0, 4.0) * 10.0)

    def outer(const, init, xs):
        return jax.lax.scan(lambda c, x: (c + x + const, c), init, xs)

    final_carry, ys = quax.quaxify(jax.jit(outer))(const, init, xs)

    # Reference: two independent single-field scans.
    def ref(c0, cst, xarr):
        return jax.lax.scan(lambda c, x: (c + x + cst, c), c0, xarr)

    fa, ya = ref(jnp.asarray(0.0), jnp.asarray(1.0), jnp.arange(1.0, 4.0))
    fb, yb = ref(jnp.asarray(0.0), jnp.asarray(10.0), jnp.arange(1.0, 4.0) * 10.0)

    assert isinstance(final_carry, _Pair) and isinstance(ys, _Pair)
    assert jnp.allclose(final_carry.a, fa) and jnp.allclose(final_carry.b, fb)
    assert jnp.allclose(ys.a, ya) and jnp.allclose(ys.b, yb)


def test_scan_length_zero():
    """A scan over a length-0 leading axis. Regression: the body was traced by
    concretely indexing `xs[0]`, which is out of bounds when the length is 0.
    Plain `lax.scan` runs zero steps and returns the initial carry + empty ys.
    """
    init = _Pair(jnp.asarray(1.0), jnp.asarray(2.0))
    xs = _Pair(jnp.zeros((0,)), jnp.zeros((0,)))

    def outer(init, xs):
        return jax.lax.scan(lambda c, x: (c + x, c), init, xs)

    final_carry, ys = quax.quaxify(jax.jit(outer))(init, xs)

    assert isinstance(final_carry, _Pair) and isinstance(ys, _Pair)
    # Zero steps -> carry unchanged, ys empty.
    assert jnp.allclose(final_carry.a, 1.0) and jnp.allclose(final_carry.b, 2.0)
    assert ys.a.shape == (0,) and ys.b.shape == (0,)


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
