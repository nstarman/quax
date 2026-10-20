"""Tests for `quax.dispatchable`: whole-function dispatch via a quax-owned primitive.

Rules are registered globally, so every function here is private to this module.
Each rule computes something different from its function's body, so a test can
tell which ran, and every result is compared against the same JAX transform
applied to a plain-array reference.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import quax

from .myarray import DenseArray, MyArray


_body_traces = 0


@quax.dispatchable
def f(x, y):
    global _body_traces
    _body_traces += 1
    return jnp.sin(x) * y


@quax.register(f)
def f_myarray_myarray(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(x.array + 100.0 * y.array)


def body(x, y):
    return jnp.sin(x) * y


def ref(x, y):  # what the rule computes, on plain arrays
    return x + 100.0 * y


qf = quax.quaxify(f)
a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
b = jnp.array([[10.0, 20.0], [30.0, 40.0]])
A, B = MyArray(a), MyArray(b)


def test_plain_jax_unchanged():
    assert jnp.allclose(f(a, b), body(a, b))
    assert jnp.allclose(jax.jit(f)(a, b), body(a, b))
    grad = jax.grad(lambda x: f(x, b).sum())(a)
    assert jnp.allclose(grad, jax.grad(lambda x: body(x, b).sum())(a))
    expected = jax.vmap(body, in_axes=(0, 1))(a, b)
    assert jnp.allclose(jax.vmap(f, in_axes=(0, 1))(a, b), expected)


def test_rule_eager():
    out = qf(A, B)
    assert isinstance(out, MyArray)
    assert jnp.allclose(out.array, ref(a, b))


def test_rule_jit_of_quaxify():
    assert jnp.allclose(jax.jit(qf)(A, B).array, ref(a, b))


def test_rule_quaxify_of_jit():
    assert jnp.allclose(quax.quaxify(jax.jit(f))(A, B).array, ref(a, b))


def test_rule_vmap_of_quaxify():
    assert jnp.allclose(jax.vmap(qf)(A, B).array, ref(a, b))


IN_AXES = [(0, 0), (0, 1), (1, 0), (0, None)]


def _inputs(in_axes):
    if in_axes[1] is None:
        return (a, b[0]), (A, MyArray(b[0]))
    return (a, b), (A, B)


@pytest.mark.parametrize("in_axes", IN_AXES)
def test_rule_inner_vmap(in_axes):
    plain, values = _inputs(in_axes)
    expected = jax.vmap(ref, in_axes=in_axes)(*plain)
    out = quax.quaxify(jax.vmap(f, in_axes=in_axes))(*values)
    assert jnp.allclose(out.array, expected)


@pytest.mark.parametrize("in_axes", IN_AXES)
def test_rule_inner_vmap_of_jit(in_axes):
    plain, values = _inputs(in_axes)
    expected = jax.vmap(ref, in_axes=in_axes)(*plain)
    out = quax.quaxify(jax.vmap(jax.jit(f), in_axes=in_axes))(*values)
    assert jnp.allclose(out.array, expected)


@pytest.mark.parametrize("in_axes", IN_AXES)
def test_rule_inner_jit_of_vmap(in_axes):
    plain, values = _inputs(in_axes)
    expected = jax.vmap(ref, in_axes=in_axes)(*plain)
    out = quax.quaxify(jax.jit(jax.vmap(f, in_axes=in_axes)))(*values)
    assert jnp.allclose(out.array, expected)


def test_rule_nested_inner_vmap():
    expected = jax.vmap(jax.vmap(ref), in_axes=(1, 0))(a, b)
    out = quax.quaxify(jax.vmap(jax.vmap(f), in_axes=(1, 0)))(A, B)
    assert jnp.allclose(out.array, expected)


def test_rule_outer_grad():
    grad = jax.grad(lambda x: qf(MyArray(x), B).array.sum())(a)
    assert jnp.allclose(grad, jax.grad(lambda x: ref(x, b).sum())(a))


@quax.dispatchable
def g_tree(d):
    return d["x"], {"y": d["x"] * d["y"]}


@quax.register(g_tree)
def g_tree_dict(d: dict) -> Any:
    return d["x"], {"y": MyArray(d["x"].array + d["y"].array)}


def test_pytree_args_and_outputs():
    x, out = quax.quaxify(g_tree)({"x": A, "y": B})
    assert isinstance(out["y"], MyArray)
    assert jnp.allclose(x.array, a)
    assert jnp.allclose(out["y"].array, a + b)


@quax.dispatchable
def scale(x, n):
    return x * n


@quax.register(scale)
def scale_myarray_int(x: MyArray, n: int) -> MyArray:
    return MyArray(x.array + 1000.0 * n)


def test_static_python_scalar_arg():
    out = quax.quaxify(scale)(A, 3)
    assert jnp.allclose(out.array, a + 3000.0)
    assert jnp.allclose(scale(a, 3), a * 3)


def test_no_rule_falls_back_to_body():
    out = qf(DenseArray(a), DenseArray(b))
    out = out.materialise() if isinstance(out, quax.Value) else out
    assert jnp.allclose(out, body(a, b))


def test_plain_arrays_under_quaxify():
    assert jnp.allclose(qf(a, b), body(a, b))


def test_jit_traces_once():
    jitted = jax.jit(f)
    jitted(a + 1.0, b)
    before = _body_traces
    jitted(a + 2.0, b)
    assert _body_traces == before


@quax.dispatchable
def h(x):
    return jnp.sin(x)


@quax.register(h)
def h_myarray(x: MyArray) -> Any:
    if x.array.ndim == 0:  # scalar input: wrong treedef
        return x, x
    return MyArray(x.array[0])  # array input: wrong shape


def test_rule_output_mismatch_raises():
    with pytest.raises(ValueError, match="'h'"):
        quax.quaxify(h)(MyArray(jnp.array(1.0)))
    with pytest.raises(ValueError, match="'h'"):
        quax.quaxify(h)(A)


@quax.dispatchable
def k(x):
    return jnp.sin(x)


@quax.register(k)
def k_myarray(x: MyArray) -> jax.Array:
    return x.array + 100.0  # a plain array, as the body returns


def test_rule_returns_array_for_array_output():
    out = quax.quaxify(k)(A)
    out = out.materialise() if isinstance(out, quax.Value) else out
    assert jnp.allclose(out, a + 100.0)


def test_keyword_args_raise():
    with pytest.raises(TypeError, match="keyword"):
        f(a, y=b)


def test_register_rejects_other_objects():
    @jax.custom_vjp
    def vjp_fn(x):
        return x

    with pytest.raises(TypeError):
        quax.register(lambda x: x)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        quax.register(vjp_fn)  # pyright: ignore[reportArgumentType]


@quax.dispatchable
def p(x, y):
    return x * y


@quax.register(p, precedence=1)
def p_myarray_value(x: MyArray, y: quax.Value) -> MyArray:
    return MyArray(jnp.ones_like(x.array))


@quax.register(p)
def p_value_myarray(x: quax.Value, y: MyArray) -> MyArray:
    return MyArray(jnp.zeros_like(y.array))


def test_precedence_picks_rule():
    assert jnp.allclose(quax.quaxify(p)(A, B).array, 1.0)


@quax.dispatchable
def m(x, y):
    return jnp.sin(x) * y


@quax.register(m)
def m_myarray_array(x: MyArray, y: jax.Array) -> MyArray:
    return MyArray(x.array + 100.0 * y)


def test_rule_with_plain_array_arg():
    assert jnp.allclose(quax.quaxify(m)(A, b).array, ref(a, b))
    out = quax.quaxify(jax.vmap(m, in_axes=(0, 1)))(A, b)
    assert jnp.allclose(out.array, jax.vmap(ref, in_axes=(0, 1))(a, b))
