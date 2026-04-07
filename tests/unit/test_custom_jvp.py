"""Tests for process_custom_jvp_call."""

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero as SZ

import quax

from .myarray import MyArray


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


def test_custom_jvp_forward():
    """Forward pass through quaxified function with custom_jvp and MyArray input."""
    x_val = jnp.array(1.0)
    expected = f_custom_jvp(x_val)

    x = MyArray(x_val)
    got = quax.quaxify(f_custom_jvp)(x)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_jvp_forward_two_args():
    """Forward pass with two MyArray args through custom_jvp."""
    x_val = jnp.array(2.0)
    y_val = jnp.array(3.0)
    expected = g_custom_jvp(x_val, y_val)

    x = MyArray(x_val)
    y = MyArray(y_val)
    got = quax.quaxify(g_custom_jvp)(x, y)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_jvp_jvp():
    """jax.jvp through quaxified custom_jvp function with MyArray."""
    x_val = jnp.array(1.0)
    t_val = jnp.array(1.0)

    expected_p, expected_t = jax.jvp(f_custom_jvp, (x_val,), (t_val,))

    x = MyArray(x_val)
    t = MyArray(t_val)

    primal_out, tangent_out = jax.jvp(quax.quaxify(f_custom_jvp), (x,), (t,))

    assert isinstance(primal_out, MyArray)
    assert isinstance(tangent_out, MyArray)
    assert jnp.allclose(primal_out.array, expected_p)
    assert jnp.allclose(tangent_out.array, expected_t)


def test_custom_jvp_grad():
    """jax.grad through quaxified custom_jvp function with MyArray."""
    x_val = jnp.array(1.0)
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


def test_custom_jvp_symbolic_zeros_grad():
    """jax.grad through quaxified symbolic_zeros custom_jvp (SZ for y tangent)."""
    x_val = jnp.array(2.0)
    y_val = jnp.array(3.0)

    expected_grad = jax.grad(lambda x: h_sym(x, y_val))(x_val)

    def fn(x: MyArray) -> jax.Array:
        return quax.quaxify(h_sym)(x, MyArray(y_val)).array

    got_grad = jax.grad(fn)(MyArray(x_val))

    assert isinstance(got_grad, MyArray)
    assert jnp.allclose(got_grad.array, expected_grad)
