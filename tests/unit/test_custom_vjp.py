"""Tests for process_custom_vjp_call."""

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

import quax

from .myarray import DenseArray, MyArray


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


@jax.custom_vjp
def g_custom_vjp(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


def g_custom_vjp_fwd(x: jax.Array, y: jax.Array):
    return x * y, (x, y)


def g_custom_vjp_bwd(res, ct):
    x, y = res
    return (y * ct, x * ct)


g_custom_vjp.defvjp(g_custom_vjp_fwd, g_custom_vjp_bwd)


def test_custom_vjp_two_args_forward():
    """Forward pass with two MyArray arguments."""
    x_val, y_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

    got = quax.quaxify(g_custom_vjp)(MyArray(x_val), MyArray(y_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, x_val * y_val)


def test_custom_vjp_grad_both_argnums():
    """`jax.grad` w.r.t. both arguments of a two-argument custom_vjp."""
    x_val, y_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

    got = jax.grad(
        lambda a, b: quax.quaxify(g_custom_vjp)(a, b).array.sum(), argnums=(0, 1)
    )(MyArray(x_val), MyArray(y_val))

    assert jnp.allclose(got[0].array, y_val)
    assert jnp.allclose(got[1].array, x_val)


def test_custom_vjp_mixed_value_and_array():
    """A quaxified custom_vjp accepts a mix of MyArray and plain arrays."""
    x_val, y_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

    got = quax.quaxify(g_custom_vjp)(MyArray(x_val), y_val)

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, x_val * y_val)


def test_custom_vjp_jit_grad():
    """`jax.jit` around `jax.grad` re-traces the fwd rule without store errors."""
    x_val = jnp.arange(1.0, 4.0)

    got = jax.jit(jax.grad(lambda a: quax.quaxify(f_custom_vjp)(a).array.sum()))(
        MyArray(x_val)
    )

    assert jnp.allclose(got.array, 100.0 * jnp.cos(x_val))


def test_custom_vjp_vmap():
    """`jax.vmap` over a quaxified custom_vjp batches correctly."""
    xs_val, ys_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

    got = jax.vmap(quax.quaxify(g_custom_vjp))(MyArray(xs_val), MyArray(ys_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, xs_val * ys_val)


def test_custom_vjp_vmap_grad():
    """`jax.vmap` of `jax.grad` through a quaxified custom_vjp."""
    xs_val, ys_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

    got = jax.vmap(jax.grad(lambda a, b: quax.quaxify(g_custom_vjp)(a, b).array.sum()))(
        MyArray(xs_val), MyArray(ys_val)
    )

    assert jnp.allclose(got.array, ys_val)


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


def test_custom_vjp_symbolic_zeros():
    """`symbolic_zeros=True`, including a `None` cotangent for one argument."""
    x_val, y_val = jnp.arange(1.0, 4.0), jnp.arange(2.0, 5.0)

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


def test_custom_vjp_fwd_forwards_input_as_residual():
    """`jax.grad` through a custom_vjp whose fwd rule returns an input as its
    own residual, exercising JAX's input-forwarding optimization."""
    x_val = jnp.arange(1.0, 4.0)

    got = jax.grad(lambda a: quax.quaxify(i_custom_vjp)(a).array.sum())(MyArray(x_val))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, 3.0 * x_val)


@jax.custom_vjp
def k_custom_vjp(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


def k_custom_vjp_fwd(x: jax.Array):
    return jnp.sin(x), (jnp.cos(x), True)


def k_custom_vjp_bwd(res, ct):
    cos_x, flag = res
    if flag:  # a Python `bool`, not a traced array
        return (7.0 * cos_x * ct,)
    return (cos_x * ct,)


k_custom_vjp.defvjp(k_custom_vjp_fwd, k_custom_vjp_bwd)


def test_custom_vjp_python_scalar_residual_stays_static():
    """A Python scalar residual reaches the bwd rule as a Python scalar.

    Densifying it into a traced array would make the `if` in the bwd rule raise
    `TracerBoolConversionError`, and the `7.0` branch proves which path ran.
    """
    x_val = jnp.arange(1.0, 4.0)

    got = jax.grad(lambda a: quax.quaxify(k_custom_vjp)(a).array.sum())(MyArray(x_val))

    assert jnp.allclose(got.array, 7.0 * jnp.cos(x_val))


def _checkpointed_loop(y0):
    """`equinox.internal.while_loop`, which is built on `jax.custom_vjp`."""

    def cond(carry):
        return carry[0] < 3

    def body(carry):
        i, y = carry
        return i + 1, y * 0.5

    return eqxi.while_loop(cond, body, (0, y0), max_steps=4, kind="checkpointed")[1]


def test_custom_vjp_grad_through_equinox_checkpointed_loop():
    """Reverse-mode works through a real `custom_vjp`-based library."""
    y0 = jnp.array([1.0])
    expected = jax.grad(lambda y: _checkpointed_loop(y).sum())(y0)

    def _sum(y):
        # the loop's output materialises on the way out of the buffer
        out = quax.quaxify(_checkpointed_loop)(y)
        return (out.array if isinstance(out, quax.ArrayValue) else out).sum()

    got = jax.grad(_sum)(DenseArray(y0))

    assert jnp.allclose(got.array, expected)


def test_custom_vjp_vmap_grad_with_forwarded_residual():
    """Batching and input-forwarding compose.

    `batching.process_custom_vjp_call` rebuilds the bwd rule's input dimensions
    from `out_trees()`'s forwarding list, so the splice in
    `_custom_vjp_fwd_wrap` has to line up under `vmap` too.
    """
    xs_val = jnp.arange(1.0, 4.0)

    got = jax.vmap(jax.grad(lambda a: quax.quaxify(i_custom_vjp)(a).array.sum()))(
        MyArray(xs_val)
    )

    assert jnp.allclose(got.array, 3.0 * xs_val)
