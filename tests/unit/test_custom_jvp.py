"""Tests for process_custom_jvp_call."""

import gc

import jax
import jax._src.core as core
import jax.numpy as jnp
import pytest
from jax.custom_derivatives import SymbolicZero as SZ

import quax
from quax._compat import typeof

from .myarray import MyArray


@pytest.fixture
def x_val():
    return jnp.array(1.0)


@pytest.fixture
def x2_val():
    return jnp.array(2.0)


@pytest.fixture
def y2_val():
    return jnp.array(3.0)


# A simple function decorated with @jax.custom_jvp for testing
@jax.custom_jvp
def f_custom_jvp(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@f_custom_jvp.defjvp
def f_custom_jvp_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    return jnp.sin(x), jnp.cos(x) * x_dot


# A custom_jvp function with two arguments
@jax.custom_jvp
def g_custom_jvp(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@g_custom_jvp.defjvp
def g_custom_jvp_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    return x * y, x * y_dot + y * x_dot


def test_custom_jvp_forward(x_val):
    """Forward pass through quaxified function with custom_jvp and MyArray input."""
    expected = f_custom_jvp(x_val)

    x = MyArray(x_val)
    got = quax.quaxify(f_custom_jvp)(x)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_jvp_forward_two_args(x2_val, y2_val):
    """Forward pass with two MyArray args through custom_jvp."""
    expected = g_custom_jvp(x2_val, y2_val)

    x = MyArray(x2_val)
    y = MyArray(y2_val)
    got = quax.quaxify(g_custom_jvp)(x, y)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_jvp_jvp(x_val):
    """jax.jvp through quaxified custom_jvp function with MyArray."""
    t_val = jnp.array(1.0)

    expected_p, expected_t = jax.jvp(f_custom_jvp, (x_val,), (t_val,))

    x = MyArray(x_val)
    t = MyArray(t_val)

    primal_out, tangent_out = jax.jvp(quax.quaxify(f_custom_jvp), (x,), (t,))

    assert isinstance(primal_out, MyArray)
    assert isinstance(tangent_out, MyArray)
    assert jnp.allclose(primal_out.array, expected_p)
    assert jnp.allclose(tangent_out.array, expected_t)


def test_custom_jvp_grad(x_val):
    """jax.grad through quaxified custom_jvp function with MyArray."""
    expected_grad = jax.grad(f_custom_jvp)(x_val)

    def scalar_fn(x: MyArray) -> jax.Array:
        return quax.quaxify(f_custom_jvp)(x).array

    got_grad = jax.grad(scalar_fn)(MyArray(x_val))

    assert isinstance(got_grad, MyArray)
    assert jnp.allclose(got_grad.array, expected_grad)


# ---------------------------------------------------------------------------
# symbolic_zeros=True fixtures
# ---------------------------------------------------------------------------


# A two-argument custom_jvp function whose JVP rule uses symbolic_zeros=True.
# When JAX differentiates w.r.t. only one argument (e.g. via jax.grad) it
# will pass a SymbolicZero for the other tangent.
@jax.custom_jvp
def h_sym(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


def _h_sym_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    result = x * y
    term1 = jnp.zeros_like(result) if type(x_dot) is SZ else x_dot * y
    term2 = jnp.zeros_like(result) if type(y_dot) is SZ else x * y_dot
    return result, term1 + term2


h_sym.defjvp(_h_sym_jvp, symbolic_zeros=True)


def test_custom_jvp_symbolic_zeros_jvp():
    """jax.jvp through quaxified symbolic_zeros custom_jvp with MyArray args."""
    x_val = jnp.array(2.0)
    y_val = jnp.array(3.0)
    xt_val = jnp.array(1.0)
    yt_val = jnp.array(0.0)

    expected_p, expected_t = jax.jvp(h_sym, (x_val, y_val), (xt_val, yt_val))

    x = MyArray(x_val)
    y = MyArray(y_val)
    xt = MyArray(xt_val)
    yt = MyArray(yt_val)

    primal_out, tangent_out = jax.jvp(quax.quaxify(h_sym), (x, y), (xt, yt))

    assert isinstance(primal_out, MyArray)
    assert isinstance(tangent_out, MyArray)
    assert jnp.allclose(primal_out.array, expected_p)
    assert jnp.allclose(tangent_out.array, expected_t)


def test_custom_jvp_symbolic_zeros_grad(x2_val, y2_val):
    """jax.grad through quaxified symbolic_zeros custom_jvp (SZ for y tangent)."""
    expected_grad = jax.grad(lambda x: h_sym(x, y2_val))(x2_val)

    def fn(x: MyArray) -> jax.Array:
        return quax.quaxify(h_sym)(x, MyArray(y2_val)).array

    got_grad = jax.grad(fn)(MyArray(x2_val))

    assert isinstance(got_grad, MyArray)
    assert jnp.allclose(got_grad.array, expected_grad)


# ---------------------------------------------------------------------------
# Mixed Value / plain-array operands, batching, and multi-argument gradients
# ---------------------------------------------------------------------------


def test_custom_jvp_mixed_value_and_array(x2_val, y2_val):
    """One MyArray operand and one plain array still dispatches through the
    custom_jvp rule (only the all-plain case short-circuits, see #58)."""
    expected = g_custom_jvp(x2_val, y2_val)

    got = quax.quaxify(g_custom_jvp)(MyArray(x2_val), y2_val)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_jvp_grad_both_argnums(x2_val, y2_val):
    """jax.grad w.r.t. both arguments of a two-argument custom_jvp."""

    def fn(x: MyArray, y: MyArray) -> jax.Array:
        return quax.quaxify(g_custom_jvp)(x, y).array

    gx, gy = jax.grad(fn, argnums=(0, 1))(MyArray(x2_val), MyArray(y2_val))

    # d(x*y)/dx = y, d(x*y)/dy = x
    assert isinstance(gx, MyArray) and isinstance(gy, MyArray)
    assert jnp.allclose(gx.array, y2_val)
    assert jnp.allclose(gy.array, x2_val)


def test_custom_jvp_vmap():
    """jax.vmap over a quaxified custom_jvp function batches correctly."""
    xs_val = jnp.arange(3.0)
    ys_val = jnp.arange(3.0) + 1.0
    expected = jax.vmap(g_custom_jvp)(xs_val, ys_val)

    got = jax.vmap(quax.quaxify(g_custom_jvp))(MyArray(xs_val), MyArray(ys_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


# A custom_jvp with a second output that is constant in `x`, so its JVP rule
# returns a SymbolicZero *output* tangent. This drives the output-symbolic-zero
# branch in `_custom_jvp_jvp_wrap` (the SZ is turned into a concrete zero).
@jax.custom_jvp
def q_two_out(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return x**2, jnp.ones(())


def _q_two_out_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = (x**2, jnp.ones(()))
    tangent_out = (2 * x * x_dot, SZ(typeof(primal_out[1])))
    return primal_out, tangent_out


q_two_out.defjvp(_q_two_out_jvp, symbolic_zeros=True)


def _raw(x):
    """Unwrap a MyArray to its array; pass plain arrays through."""
    return x.array if isinstance(x, MyArray) else x


def test_custom_jvp_symbolic_zero_output_tangent():
    """A JVP rule returning a SymbolicZero output tangent yields a concrete
    zero tangent after quaxification (exercises the output-SZ branch).

    The second output is constant in ``x``, so quax materialises a mismatched
    primal/tangent pair to plain arrays -- hence values are compared without
    assuming a MyArray wrapper.
    """
    x_val = jnp.array(3.0)
    t_val = jnp.array(1.0)

    expected_p, expected_t = jax.jvp(q_two_out, (x_val,), (t_val,))

    primal_out, tangent_out = jax.jvp(
        quax.quaxify(q_two_out), (MyArray(x_val),), (MyArray(t_val),)
    )

    for got, exp in zip(primal_out, expected_p, strict=True):
        assert jnp.allclose(_raw(got), exp)
    for got, exp in zip(tangent_out, expected_t, strict=True):
        assert jnp.allclose(_raw(got), exp)
    # The symbolic-zero output tangent becomes an exact concrete zero.
    assert jnp.array_equal(_raw(tangent_out[1]), jnp.zeros(()))


class NoMaterialise(quax.ArrayValue):
    """Refuses to materialise, like `lora.LoraArray(allow_materialise=False)`."""

    array: jax.Array

    def materialise(self):
        raise RuntimeError("refusing to materialise")

    def aval(self):
        return typeof(self.array)


def test_aborted_trace_does_not_clobber_trace_context():
    """An exception mid-trace must not leave JAX's trace context to be restored later.

    When quaxifying raises part-way through a `custom_jvp` function, JAX abandons the
    `lu.transformation` generator wrapping it rather than throwing into it. If that
    generator held the trace context open across its `yield`, Python would restore
    the stale trace whenever it got around to finalising the generator -- clobbering
    an unrelated, live trace, and surfacing as an `UnexpectedTracerError` far from
    the cause.
    """
    # `jax.nn.relu` is a custom_jvp function, and there is no rule for this value, so
    # the default process tries to materialise and the trace aborts part-way through.
    with pytest.raises(RuntimeError, match="refusing to materialise"):
        quax.quaxify(jax.nn.relu)(NoMaterialise(jnp.arange(3.0)))

    @jax.jit
    def f(x):
        before = core.trace_ctx.trace
        gc.collect()  # finalise the abandoned generator while this trace is live
        assert core.trace_ctx.trace is before
        return x + 1

    f(jnp.arange(3.0))
