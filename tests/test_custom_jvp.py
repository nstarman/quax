"""Tests for process_custom_jvp_call."""

import jax
import jax.numpy as jnp

import quax
from tests.myarray import MyArray


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
