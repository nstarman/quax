"""Tests for the LoraArray class and related."""

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
from jaxtyping import TypeCheckError
from plum import NotFoundLookupError

import quax
import quax.examples.lora as lora


def test_linear(getkey):
    linear = eqx.nn.Linear(10, 12, key=getkey())
    lora_weight = lora.LoraArray(linear.weight, rank=2, key=getkey())
    linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)
    vector = jr.normal(getkey(), (10,))
    quax.quaxify(linear)(vector)
    eqx.filter_jit(quax.quaxify(linear))(vector)


def test_loraify(getkey):
    mlp = eqx.nn.MLP(2, 2, 64, 3, key=getkey())
    mlp = lora.loraify(mlp, rank=3, key=getkey())
    vector = jr.normal(getkey(), (2,))
    quax.quaxify(mlp)(vector)
    eqx.filter_jit(quax.quaxify(mlp))(vector)


def test_complicated_dot(getkey):
    lhs = jr.normal(getkey(), (3, 2, 1, 4, 5))
    rhs = jr.normal(getkey(), (8, 1, 5, 7, 3, 9, 10))
    lhs_contract = (4,)
    rhs_contract = (2,)
    lhs_batch = (2, 0)
    rhs_batch = (1, 4)
    dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
    out = lax.dot_general(lhs, rhs, dimension_numbers)
    lhs_lora = lora.LoraArray(lhs, rank=11, key=getkey())
    out_lora = quax.quaxify(lax.dot_general)(lhs_lora, rhs, dimension_numbers)
    assert out.shape == out_lora.shape

    dimension_numbers2 = ((rhs_contract, lhs_contract), (rhs_batch, lhs_batch))
    out2 = lax.dot_general(rhs, lhs, dimension_numbers2)
    out_lora2 = quax.quaxify(lax.dot_general)(rhs, lhs_lora, dimension_numbers2)
    assert out2.shape == out_lora2.shape


def test_stop_gradient(getkey):
    mlp = eqx.nn.MLP(2, 2, 64, 3, activation=jax.nn.softplus, key=getkey())
    mlp_true = lora.loraify(mlp, rank=3, stop_gradient=True, key=getkey())
    mlp_false = lora.loraify(mlp, rank=3, stop_gradient=False, key=getkey())
    vector = jr.normal(getkey(), (2,))

    @eqx.filter_jit
    @eqx.filter_grad
    @quax.quaxify
    def run1(mlp, vector):
        return jnp.sum(mlp(vector))

    grad_true = run1(mlp_true, vector)
    grad_false = run1(mlp_false, vector)

    assert (grad_true.layers[1].weight.w == 0).all()
    assert (grad_true.layers[1].weight.a == 0).all()  # becuase b==0 at init
    assert not (grad_true.layers[1].weight.b == 0).all()
    assert not (grad_true.layers[1].bias == 0).all()

    assert not (grad_false.layers[1].weight.w == 0).all()
    assert (grad_false.layers[1].weight.a == 0).all()  # because b==0 at init
    assert not (grad_false.layers[1].weight.b == 0).all()
    assert not (grad_false.layers[1].bias == 0).all()


def test_decorator_stack_runs(getkey):
    mlp = eqx.nn.MLP(2, 2, 64, 3, activation=jax.nn.softplus, key=getkey())
    mlp = lora.loraify(mlp, rank=3, key=getkey())
    vector = jr.normal(getkey(), (2,))

    @eqx.filter_jit
    @quax.quaxify
    @eqx.filter_grad
    def run2(mlp, vector):
        return jnp.sum(mlp(vector))

    # Not efficient!
    @quax.quaxify
    @eqx.filter_jit
    @eqx.filter_grad
    def run3(mlp, vector):
        return jnp.sum(mlp(vector))

    run2(mlp, vector)
    run3(mlp, vector)
    run2(mlp, vector)
    run3(mlp, vector)


def test_materialise():
    """`allow_materialise=False` refuses to materialise, and does so mid-trace.

    The `RuntimeError` escapes from inside a quaxified `custom_jvp` function, so this
    also exercises the abandoned-transformation path: see
    `tests/unit/test_custom_jvp.py::test_aborted_trace_does_not_clobber_trace_context`
    for what that used to do to the next test to run.
    """
    key = jr.key(0)

    key, *subkeys = jr.split(key, 3)
    x_false = lora.LoraArray(
        jr.normal(subkeys[0], (3, 3)), rank=2, allow_materialise=False, key=subkeys[1]
    )
    key, *subkeys = jr.split(key, 3)
    x_true = lora.LoraArray(
        jr.normal(subkeys[0], (3, 3)), rank=2, allow_materialise=True, key=subkeys[1]
    )

    _ = quax.quaxify(jax.nn.relu)(x_true)
    with pytest.raises(RuntimeError, match="Refusing to materialise"):
        _ = quax.quaxify(jax.nn.relu)(x_false)


def test_regression_38(getkey):
    """Regression test for PR 38 (stackless tracers)."""
    x = jnp.arange(4.0).reshape(2, 2)
    y = lora.LoraArray(x, rank=1, key=getkey())

    def f(x):
        return jax.lax.add_p.bind(x, y)

    func = quax.quaxify(f)

    # Binding a raw `LoraArray` must be an error rather than silently doing something
    # wrong. Which error depends on how far it gets. With stackless tracers, JAX
    # canonicalises `bind`'s arguments before consulting the current trace, so it
    # raises TypeError without ever reaching quax's dispatch. Reaching dispatch gives
    # TypeCheckError when jaxtyping is on, and NotFoundLookupError when it is off --
    # which kicks over to the default process, itself raising RuntimeError when
    # `allow_materialise` is False.
    with pytest.raises((TypeError, TypeCheckError, NotFoundLookupError, RuntimeError)):
        _ = func(y)


def _count(tree):
    return sum(x.size for x in jtu.tree_leaves(tree) if eqx.is_array(x))


def test_trainable_filter_counts_only_the_adaptation(getkey):
    linear = eqx.nn.Linear(64, 32, key=getkey())
    loraed = lora.loraify(linear, rank=4, key=getkey())

    trainable = eqx.filter(loraed, lora.trainable_filter(loraed))

    # a (32x4) + b (4x64) + bias (32), but not the 32x64 frozen weight.
    assert _count(trainable) == 32 * 4 + 4 * 64 + 32
    assert _count(loraed) == _count(trainable) + 32 * 64


def test_trainable_filter_keeps_everything_when_not_frozen(getkey):
    mlp = eqx.nn.MLP(2, 2, 8, 2, key=getkey())
    loraed = lora.loraify(mlp, rank=3, stop_gradient=False, key=getkey())

    trainable = eqx.filter(loraed, lora.trainable_filter(loraed))

    assert _count(trainable) == _count(loraed)


def test_trainable_filter_is_a_no_op_without_lora(getkey):
    mlp = eqx.nn.MLP(2, 2, 8, 2, key=getkey())

    trainable = eqx.filter(mlp, lora.trainable_filter(mlp))

    assert _count(trainable) == _count(mlp)


def test_trainable_filter_removes_the_zero_cotangent(getkey):
    """Partitioning gives the same gradients without allocating one for `w`."""
    linear = eqx.nn.Linear(64, 32, key=getkey())
    loraed = lora.loraify(linear, rank=4, key=getkey())
    vector = jr.normal(getkey(), (64,))

    @eqx.filter_grad
    def whole(model, x):
        return jnp.sum(quax.quaxify(model)(x))

    @eqx.filter_grad
    def split(trainable, frozen, x):
        return jnp.sum(quax.quaxify(eqx.combine(trainable, frozen))(x))

    trainable, frozen = eqx.partition(loraed, lora.trainable_filter(loraed))
    grad_whole = whole(loraed, vector)
    grad_split = split(trainable, frozen, vector)

    # Differentiating the whole model produces a full-size buffer of zeros.
    assert (grad_whole.weight._w == 0).all()
    assert grad_split.weight._w is None

    assert jnp.array_equal(grad_split.weight.a, grad_whole.weight.a)
    assert jnp.array_equal(grad_split.weight.b, grad_whole.weight.b)
    assert jnp.array_equal(grad_split.bias, grad_whole.bias)
