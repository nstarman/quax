"""Tests for process_custom_vjp_call."""

import jax
import jax.numpy as jnp
import pytest

import quax

from .myarray import MyArray


@pytest.fixture
def x_val():
    return jnp.arange(1.0, 4.0)


@pytest.fixture
def y_val():
    return jnp.arange(2.0, 5.0)


@jax.custom_vjp
def f_custom_vjp(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


def f_custom_vjp_fwd(x: jax.Array):
    return jnp.sin(x), jnp.cos(x)


def f_custom_vjp_bwd(res, ct):
    # deliberately scaled by 100 so a test can prove the *custom* rule ran
    return (100.0 * res * ct,)


f_custom_vjp.defvjp(f_custom_vjp_fwd, f_custom_vjp_bwd)


def test_custom_vjp_forward(x_val):
    """Forward pass through a quaxified custom_vjp function with MyArray input."""
    expected = f_custom_vjp(x_val)

    got = quax.quaxify(f_custom_vjp)(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected)


def test_custom_vjp_grad_uses_custom_rule(x_val):
    """`jax.grad` through a quaxified custom_vjp uses the user's bwd rule."""
    got = jax.grad(lambda a: quax.quaxify(f_custom_vjp)(a).array.sum())(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, 100.0 * jnp.cos(x_val))


@jax.custom_vjp
def g_custom_vjp(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


def g_custom_vjp_fwd(x: jax.Array, y: jax.Array):
    return x * y, (x, y)


def g_custom_vjp_bwd(res, ct):
    x, y = res
    return (y * ct, x * ct)


g_custom_vjp.defvjp(g_custom_vjp_fwd, g_custom_vjp_bwd)


def test_custom_vjp_two_args_forward(x_val, y_val):
    """Forward pass with two MyArray arguments."""
    got = quax.quaxify(g_custom_vjp)(MyArray(x_val), MyArray(y_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, x_val * y_val)


def test_custom_vjp_grad_both_argnums(x_val, y_val):
    """`jax.grad` w.r.t. both arguments of a two-argument custom_vjp."""
    got = jax.grad(
        lambda a, b: quax.quaxify(g_custom_vjp)(a, b).array.sum(), argnums=(0, 1)
    )(MyArray(x_val), MyArray(y_val))

    assert jnp.allclose(got[0].array, y_val)
    assert jnp.allclose(got[1].array, x_val)


def test_custom_vjp_mixed_value_and_array(x_val, y_val):
    """A quaxified custom_vjp accepts a mix of MyArray and plain arrays."""
    got = quax.quaxify(g_custom_vjp)(MyArray(x_val), y_val)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, x_val * y_val)


def test_custom_vjp_jit_grad(x_val):
    """`jax.jit` around `jax.grad` re-traces the fwd rule without store errors."""
    got = jax.jit(jax.grad(lambda a: quax.quaxify(f_custom_vjp)(a).array.sum()))(
        MyArray(x_val)
    )

    assert jnp.allclose(got.array, 100.0 * jnp.cos(x_val))


def test_custom_vjp_vmap(x_val, y_val):
    """`jax.vmap` over a quaxified custom_vjp batches correctly."""
    got = jax.vmap(quax.quaxify(g_custom_vjp))(MyArray(x_val), MyArray(y_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, x_val * y_val)


def test_custom_vjp_vmap_grad(x_val, y_val):
    """`jax.vmap` of `jax.grad` through a quaxified custom_vjp."""
    got = jax.vmap(jax.grad(lambda a, b: quax.quaxify(g_custom_vjp)(a, b).array.sum()))(
        MyArray(x_val), MyArray(y_val)
    )

    assert jnp.allclose(got.array, y_val)


# A custom_vjp declared with `symbolic_zeros=True`: the fwd rule receives
# `CustomVJPPrimal` wrappers rather than bare values.
@jax.custom_vjp
def h_custom_vjp(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


def h_custom_vjp_fwd(x, y):
    return x.value * y.value, (x.value, y.value)


def h_custom_vjp_bwd(res, ct):
    _, y = res
    return (y * ct, None)  # `None`: no cotangent for `y`


h_custom_vjp.defvjp(h_custom_vjp_fwd, h_custom_vjp_bwd, symbolic_zeros=True)


def test_custom_vjp_symbolic_zeros(x_val, y_val):
    """`symbolic_zeros=True`, including a `None` cotangent for one argument."""
    got = jax.grad(lambda a, b: quax.quaxify(h_custom_vjp)(a, b).array.sum())(
        MyArray(x_val), MyArray(y_val)
    )

    assert jnp.allclose(got.array, y_val)


@jax.custom_vjp
def i_custom_vjp(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


def i_custom_vjp_fwd(x: jax.Array):
    # The residual *is* the input itself. JAX detects this and prunes it from
    # the fwd rule's return value, recording the argument index in
    # `input_forwards` instead -- this exercises the splice in
    # `_custom_vjp_fwd_wrap` that reconstitutes forwarded residuals from
    # `in_tracers` before flattening.
    return jnp.sin(x), x


def i_custom_vjp_bwd(res, ct):
    # deliberately not `cos(res) * ct`, so this could not be produced by
    # ordinary autodiff -- proves the custom (and forwarded) rule ran.
    return (3.0 * res * ct,)


i_custom_vjp.defvjp(i_custom_vjp_fwd, i_custom_vjp_bwd)


def test_custom_vjp_fwd_forwards_input_as_residual(x_val):
    """`jax.grad` through a custom_vjp whose fwd rule returns an input as its
    own residual, exercising JAX's input-forwarding optimization."""
    got = jax.grad(lambda a: quax.quaxify(i_custom_vjp)(a).array.sum())(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, 3.0 * x_val)
