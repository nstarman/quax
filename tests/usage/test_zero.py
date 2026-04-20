from typing import Final

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import pytest
from packaging.version import Version

import quax
import quax.examples.zero as zero
from quax._compat import JAX_VERSION

from ..helpers import tree_allclose


JAX_GE_0_10_0: Final = JAX_VERSION >= Version("0.10.0")


def test_broadcast():
    z = zero.Zero((3, 4), jnp.float32)
    true_out = zero.Zero((2, 1, 3, 4), jnp.float32)
    out = quax.quaxify(jnp.broadcast_to)(z, (2, 1, 3, 4))
    assert eqx.tree_equal(out, true_out)


def test_cast():
    z = zero.Zero((3, 4), jnp.int32)
    true_out = zero.Zero((3, 4), jnp.float32)
    out1 = quax.quaxify(lambda x: x.astype(jnp.float32))(z)
    out2 = quax.quaxify(lax.convert_element_type)(z, jnp.float32)
    assert eqx.tree_equal(out1, true_out)
    assert eqx.tree_equal(out2, true_out)


def test_materialise():
    z = zero.Zero((3, 4), jnp.float32)
    out = z.materialise()
    assert jnp.array_equal(out, jnp.zeros((3, 4), jnp.float32))


def test_add():
    scalar_zero = zero.Zero((), jnp.float32)
    vector_zero = zero.Zero((2,), jnp.bool_)
    tensor_zero = zero.Zero((3, 3, 3), jnp.int32)

    for add in (quax.quaxify(lambda x, y: x + y), quax.quaxify(jnp.add)):
        assert tree_allclose(add(scalar_zero, 1), jnp.array(1.0))
        assert eqx.tree_equal(add(scalar_zero, jnp.array(1)), jnp.array(1.0))
        assert tree_allclose(add(1, scalar_zero), jnp.array(1.0))
        assert eqx.tree_equal(add(jnp.array(1), scalar_zero), jnp.array(1.0))

        assert tree_allclose(add(vector_zero, 1), jnp.array([1, 1]))
        assert eqx.tree_equal(add(vector_zero, jnp.array(1)), jnp.array([1, 1]))
        assert tree_allclose(add(1, vector_zero), jnp.array([1, 1]))
        assert eqx.tree_equal(add(jnp.array(1), vector_zero), jnp.array([1, 1]))

        assert tree_allclose(add(tensor_zero, 1), jnp.ones((3, 3, 3), jnp.int32))
        assert eqx.tree_equal(
            add(tensor_zero, jnp.array(1)), jnp.ones((3, 3, 3), jnp.int32)
        )
        assert tree_allclose(add(1, tensor_zero), jnp.ones((3, 3, 3), jnp.int32))
        assert eqx.tree_equal(
            add(jnp.array(1), tensor_zero), jnp.ones((3, 3, 3), jnp.int32)
        )


def test_mul():
    # TODO: revisit this once we support weak types.
    scalar_zero = zero.Zero((), jnp.float32)
    in_vector_zero = zero.Zero((2,), jnp.bool_)
    out_vector_zero = zero.Zero((2,), jnp.int32)
    tensor_zero = zero.Zero((3, 3, 3), jnp.int32)

    for mul in (quax.quaxify(lambda x, y: x * y), quax.quaxify(jnp.multiply)):
        assert tree_allclose(mul(scalar_zero, 1), scalar_zero)
        assert eqx.tree_equal(mul(scalar_zero, jnp.array(1)), scalar_zero)
        assert tree_allclose(mul(1, scalar_zero), scalar_zero)
        assert eqx.tree_equal(mul(jnp.array(1), scalar_zero), scalar_zero)

        assert tree_allclose(mul(in_vector_zero, 1), out_vector_zero)
        assert eqx.tree_equal(mul(in_vector_zero, jnp.array(1)), out_vector_zero)
        assert tree_allclose(mul(1, in_vector_zero), out_vector_zero)
        assert eqx.tree_equal(mul(jnp.array(1), in_vector_zero), out_vector_zero)

        assert tree_allclose(mul(tensor_zero, 1), tensor_zero)
        assert eqx.tree_equal(mul(tensor_zero, jnp.array(1)), tensor_zero)
        assert tree_allclose(mul(1, tensor_zero), tensor_zero)
        assert eqx.tree_equal(mul(jnp.array(1), tensor_zero), tensor_zero)


@pytest.mark.skipif(not JAX_GE_0_10_0, reason="out_dtype kwarg requires JAX >= 0.10.0")
def test_mul_out_dtype():
    """Test that mul_p respects out_dtype kwarg (added in JAX v0.10.0)."""
    scalar_zero = zero.Zero((), jnp.float32)
    vector_zero = zero.Zero((2,), jnp.float32)
    out_dtype = jnp.float16

    mul_bind = quax.quaxify(lax.mul_p.bind)

    # Zero * ArrayLike with out_dtype
    result = mul_bind(scalar_zero, jnp.array(2.0, jnp.float32), out_dtype=out_dtype)
    assert isinstance(result, zero.Zero)
    assert result.dtype == out_dtype
    assert result.shape == ()

    # ArrayLike * Zero with out_dtype
    result = mul_bind(jnp.array(2.0, jnp.float32), scalar_zero, out_dtype=out_dtype)
    assert isinstance(result, zero.Zero)
    assert result.dtype == out_dtype
    assert result.shape == ()

    # Zero * Zero with out_dtype
    result = mul_bind(vector_zero, vector_zero, out_dtype=out_dtype)
    assert isinstance(result, zero.Zero)
    assert result.dtype == out_dtype
    assert result.shape == (2,)


def test_matmul(getkey):
    for use_bias in (False, True):
        linear = eqx.nn.Linear(2, 3, key=getkey(), use_bias=use_bias)
        linear = eqx.tree_at(
            lambda x: x.weight,
            linear,
            zero.Zero(linear.weight.shape, linear.weight.dtype),
        )
        out = quax.quaxify(linear)(jnp.ones(2))
        if use_bias:
            assert jnp.array_equal(out, linear.bias)
        else:
            assert eqx.tree_equal(out, zero.Zero((3,), jnp.float32))


def test_dynamic_update_slice(getkey):
    z = zero.Zero((3, 4), jnp.float32)
    x = jr.normal(getkey(), (2, 3))
    out = quax.quaxify(lax.dynamic_update_slice)(z, x, (1, 1))
    true_out = jnp.zeros((3, 4)).at[1:3, 1:4].set(x)
    assert jnp.array_equal(out, true_out)

    z = zero.Zero((3, 4), jnp.float32)
    x = zero.Zero((2, 3), jnp.float32)
    out = quax.quaxify(lax.dynamic_update_slice)(z, x, (1, 1))
    assert eqx.tree_equal(out, z)


def test_slice(getkey):
    z = zero.Zero((5, 32, 1024), jnp.float32)
    out = quax.quaxify(lambda x: x[1:, 3:4])(z)
    true_shape = (4, 1, 1024)
    assert eqx.tree_equal(out, zero.Zero(true_shape, jnp.float32))


@pytest.mark.skip("dynamic quaxify is disabled for now")
def test_creation():
    for maybe_jit in (jax.jit, lambda x: x):

        @maybe_jit
        @quax.quaxify
        def run(a):
            return jnp.zeros((3, 4))

        out = run(1)
        out2 = quax.quaxify(jnp.zeros)((3, 4))
        z = zero.Zero((3, 4), jnp.float32)
        assert eqx.tree_equal(out, z)
        assert eqx.tree_equal(out2, z)


def test_pow():
    z = zero.Zero((3, 4), jnp.float32)

    power = quax.quaxify(pow)

    # Standard power
    out = power(z, 3)
    assert isinstance(out, zero.Zero)
    assert out.shape == (3, 4)

    # Zero power. First check the JAX behaviour.
    assert jnp.array(0) ** 0 == 1
    assert jnp.array(0.0) ** 0 == 1
    assert 0 ** jnp.array(0) == 1
    assert 0.0 ** jnp.array(0) == 1
    assert jnp.array(0) ** jnp.array(0) == 1
    assert jnp.array(0.0) ** jnp.array(0) == 1

    out = power(z, 0)
    assert not isinstance(out, zero.Zero)
    ones = jnp.ones((3, 4), jnp.float32)
    assert jnp.array_equal(out, ones)

    # More complex zero powers
    out = power(z, jnp.array(0))
    assert not isinstance(out, zero.Zero)
    assert jnp.array_equal(out, ones)

    out = power(z, z)
    assert not isinstance(out, zero.Zero)
    assert jnp.array_equal(out, ones)

    # Reverse power
    out = power(3, z)
    assert not isinstance(out, zero.Zero)
    assert jnp.array_equal(out, ones)
