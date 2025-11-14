"""Tests for custom VJP and JVP support in Quax."""

import importlib.metadata
from typing import cast, Final

import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import quax


JAX_VERSION = tuple(map(int, importlib.metadata.version("jax").split(".")))
JAX_VERSION_LT_7: Final = JAX_VERSION < (0, 7, 0)


class ScalarValue(quax.ArrayValue):
    """A simple Value wrapper for testing custom derivatives."""

    value: Float[Array, ""]

    def materialise(self):
        return self.value

    def aval(self):
        return cast(jax.core.ShapedArray, jax.core.get_aval(self.value))


def test_basic_custom_vjp():
    """Test that basic custom VJP works with Quax Values."""

    @jax.custom_vjp
    def f(x):
        return x**2

    def f_fwd(x):
        return f(x), x

    def f_bwd(x, g):
        return (2 * g * x,)  # Standard gradient: 2x

    f.defvjp(f_fwd, f_bwd)

    # Test with Quax Value
    x_val = ScalarValue(jnp.array(2.0))
    result, vjp_fn = jax.vjp(quax.quaxify(f), x_val)
    (grad,) = vjp_fn(jnp.array(1.0))

    assert jnp.allclose(result, 4.0)
    assert isinstance(grad, ScalarValue)
    assert jnp.allclose(grad.materialise(), 4.0)  # 2 * 2


def test_custom_vjp_with_grad():
    """Test custom VJP with jax.grad."""

    @jax.custom_vjp
    def square(x):
        return x**2

    def square_fwd(x):
        return square(x), x

    def square_bwd(x, g):
        return (2 * g * x,)

    square.defvjp(square_fwd, square_bwd)

    # Test with Quax Value
    x_val = ScalarValue(jnp.array(3.0))
    grad_val = jax.grad(quax.quaxify(square))(x_val)

    assert isinstance(grad_val, ScalarValue)
    assert jnp.allclose(grad_val.materialise(), 6.0)


def test_custom_vjp_with_multiple_args():
    """Test custom VJP with multiple arguments."""

    @jax.custom_vjp
    def multiply(x, y):
        return x * y

    def multiply_fwd(x, y):
        return multiply(x, y), (x, y)

    def multiply_bwd(res, g):
        x, y = res
        return (g * y, g * x)

    multiply.defvjp(multiply_fwd, multiply_bwd)

    x_val = ScalarValue(jnp.array(3.0))
    y_val = ScalarValue(jnp.array(4.0))

    result, vjp_fn = jax.vjp(quax.quaxify(multiply), x_val, y_val)
    grad_x, grad_y = vjp_fn(jnp.array(1.0))

    assert jnp.allclose(result, 12.0)
    assert isinstance(grad_x, ScalarValue)
    assert isinstance(grad_y, ScalarValue)
    assert jnp.allclose(grad_x.materialise(), 4.0)
    assert jnp.allclose(grad_y.materialise(), 3.0)


def test_custom_vjp_with_jit():
    """Test that custom VJP works with JIT compilation."""

    @jax.custom_vjp
    def f(x):
        return jnp.sin(x)

    def f_fwd(x):
        return f(x), jnp.cos(x)

    def f_bwd(cos_x, g):
        return (g * cos_x,)

    f.defvjp(f_fwd, f_bwd)

    @jax.jit
    @quax.quaxify
    def jitted_grad(x):
        return jax.grad(f)(x)

    x_val = ScalarValue(jnp.array(0.5))
    grad = jitted_grad(x_val)

    # JIT may materialize Values to arrays
    assert jnp.allclose(grad, jnp.cos(0.5))


def test_custom_vjp_symbolic_zero():
    """Test custom VJP with symbolic zeros."""

    @jax.custom_vjp
    def f(x, y):
        return x**2  # Only depends on x

    def f_fwd(x, y):
        return f(x, y), x

    def f_bwd(x, g):
        return (2 * g * x, jnp.zeros_like(g))  # Zero gradient wrt y

    f.defvjp(f_fwd, f_bwd)

    x_val = ScalarValue(jnp.array(3.0))
    y = jnp.array(5.0)

    result, vjp_fn = jax.vjp(quax.quaxify(f), x_val, y)
    grad_x, grad_y = vjp_fn(jnp.array(1.0))

    assert jnp.allclose(result, 9.0)
    assert isinstance(grad_x, ScalarValue)
    assert jnp.allclose(grad_x.materialise(), 6.0)
    assert jnp.allclose(grad_y, 0.0)


def test_custom_vjp_nondiff_args():
    """Test custom VJP with non-differentiable arguments."""

    @jax.custom_vjp
    def scale(x, factor):
        return x * factor

    def scale_fwd(x, factor):
        return scale(x, factor), factor

    def scale_bwd(factor, g):
        return (g * factor, None)  # None for non-differentiable args

    scale.defvjp(scale_fwd, scale_bwd)

    x_val = ScalarValue(jnp.array(2.0))
    result, vjp_fn = jax.vjp(quax.quaxify(scale), x_val, 5.0)
    grad_x, grad_factor = vjp_fn(jnp.array(1.0))

    assert jnp.allclose(result, 10.0)
    assert isinstance(grad_x, ScalarValue)
    assert jnp.allclose(grad_x.materialise(), 5.0)
    # Non-differentiable arg may get None or zero
    assert grad_factor is None or jnp.allclose(grad_factor, 0.0)


def test_custom_vjp_value_and_grad():
    """Test custom VJP with value_and_grad."""

    @jax.custom_vjp
    def loss(x):
        return jnp.sum(x**2)

    def loss_fwd(x):
        return loss(x), x

    def loss_bwd(x, g):
        return (2 * g * x,)

    loss.defvjp(loss_fwd, loss_bwd)

    x = jnp.array([1.0, 2.0, 3.0])
    value, grad = jax.value_and_grad(quax.quaxify(loss))(x)

    assert jnp.allclose(value, 14.0)  # 1 + 4 + 9
    assert jnp.allclose(grad, jnp.array([2.0, 4.0, 6.0]))


@pytest.mark.skipif(JAX_VERSION_LT_7, reason="Requires JAX >= 0.7.0")
def test_custom_vjp_with_registered_primitive():
    """Test that custom VJP works when Values have registered primitives."""

    class MyValue(quax.ArrayValue):
        array: Array

        def materialise(self):
            return self.array

        def aval(self):
            return cast(jax.core.ShapedArray, jax.core.get_aval(self.array))

    @quax.register(lax.mul_p)
    def _(x: MyValue, y: MyValue):
        return MyValue(x.array * y.array * 2)  # Custom multiply: 2xy

    @jax.custom_vjp
    def compute(x, y):
        return x * y

    def compute_fwd(x, y):
        return compute(x, y), (x, y)

    def compute_bwd(res, g):
        x, y = res
        return (g * y, g * x)

    compute.defvjp(compute_fwd, compute_bwd)

    x_val = MyValue(jnp.array(3.0))
    y_val = MyValue(jnp.array(4.0))

    result, vjp_fn = jax.vjp(quax.quaxify(compute), x_val, y_val)

    assert isinstance(result, MyValue)
    assert jnp.allclose(result.materialise(), 24.0)  # 2*3*4

    grad_x, grad_y = vjp_fn(MyValue(jnp.array(1.0)))
    assert isinstance(grad_x, MyValue)
    assert isinstance(grad_y, MyValue)


@pytest.mark.skipif(JAX_VERSION_LT_7, reason="Requires JAX >= 0.7.0")
def test_custom_vjp_with_residuals():
    """Test custom VJP that stores residuals for backward pass."""

    call_count = []

    @jax.custom_vjp
    def identity(x):
        return x

    def identity_fwd(x):
        call_count.append("fwd")
        return identity(x), None

    def identity_bwd(_, g):
        call_count.append("bwd")
        return (g,)

    identity.defvjp(identity_fwd, identity_bwd)

    x = ScalarValue(jnp.array(1.0))
    result, vjp_fn = jax.vjp(quax.quaxify(identity), x)
    (grad,) = vjp_fn(ScalarValue(jnp.array(1.0)))

    assert len(call_count) == 2
    assert call_count == ["fwd", "bwd"]


def test_nested_custom_vjp():
    """Test nested custom VJP calls."""

    @jax.custom_vjp
    def inner(x):
        return x**2

    def inner_fwd(x):
        return inner(x), x

    def inner_bwd(x, g):
        return (2 * g * x,)

    inner.defvjp(inner_fwd, inner_bwd)

    @jax.custom_vjp
    def outer(x):
        return inner(x) + 1

    def outer_fwd(x):
        return outer(x), x

    def outer_bwd(x, g):
        # Gradient of inner is handled by its custom VJP
        # Must return tuple with one element per arg
        return (jax.grad(lambda y: inner(y) + 1)(x) * g,)

    outer.defvjp(outer_fwd, outer_bwd)

    x_val = ScalarValue(jnp.array(3.0))
    grad_fn = jax.grad(quax.quaxify(outer))
    grad = grad_fn(x_val)

    assert isinstance(grad, ScalarValue)
    assert jnp.allclose(grad.materialise(), 6.0)  # 2 * 3
