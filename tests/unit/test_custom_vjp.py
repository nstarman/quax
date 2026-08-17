"""Tests for process_custom_vjp_call."""

import jax
import jax.numpy as jnp

import quax

from .myarray import MyArray


@jax.custom_vjp
def f_custom_vjp(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


def f_custom_vjp_fwd(x: jax.Array):
    return jnp.sin(x), jnp.cos(x)


def f_custom_vjp_bwd(res, ct):
    # deliberately scaled by 100 so a test can prove the *custom* rule ran
    return (100.0 * res * ct,)


f_custom_vjp.defvjp(f_custom_vjp_fwd, f_custom_vjp_bwd)


def test_custom_vjp_forward():
    """Forward pass through a quaxified custom_vjp function with MyArray input."""
    x_val = jnp.arange(1.0, 4.0)
    expected = f_custom_vjp(x_val)

    got = quax.quaxify(f_custom_vjp)(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_vjp_grad_uses_custom_rule():
    """`jax.grad` through a quaxified custom_vjp uses the user's bwd rule."""
    x_val = jnp.arange(1.0, 4.0)

    got = jax.grad(lambda a: quax.quaxify(f_custom_vjp)(a).array.sum())(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, 100.0 * jnp.cos(x_val))
