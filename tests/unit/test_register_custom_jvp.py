"""Tests for `quax.register` with a `jax.custom_jvp` target.

Rules are registered globally, so every `custom_jvp` here is private to this
module; the rules' results differ from the bodies so tests can tell them apart.
"""

import functools as ft
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import ArrayLike

import quax

from .myarray import MyArray


# A one-input custom_jvp whose body (`sin`) MyArray can run inline.
@jax.custom_jvp
def f_sin(x):
    return jnp.sin(x)


@f_sin.defjvp
def f_sin_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sin(x), jnp.cos(x) * dx


@quax.register(f_sin)
def f_sin_myarray(x: MyArray) -> list[MyArray]:
    return [MyArray(x.array + 100.0)]


def test_rule_replaces_body_eager():
    out = quax.quaxify(f_sin)(MyArray(1.0))
    assert isinstance(out, MyArray)
    assert out.array == 101.0


@jax.custom_jvp
def f_dict(x):
    return {"y": jnp.sin(x)}


@f_dict.defjvp
def f_dict_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return {"y": jnp.sin(x)}, {"y": jnp.cos(x) * dx}


@quax.register(f_dict)
def f_dict_myarray(x: MyArray) -> list[MyArray]:
    return [MyArray(x.array + 100.0)]


def test_rule_replaces_body_dict_output():
    """Eager call: JAX reads the output tree from a store only `fun` fills."""
    out = quax.quaxify(f_dict)(MyArray(1.0))
    assert isinstance(out["y"], MyArray)
    assert out["y"].array == 101.0


def test_rule_quaxify_of_jit():
    assert quax.quaxify(jax.jit(f_sin))(MyArray(1.0)).array == 101.0


def test_rule_jit_of_quaxify():
    assert jax.jit(quax.quaxify(f_sin))(MyArray(1.0)).array == 101.0


def test_rule_vmap():
    out = jax.vmap(quax.quaxify(f_sin))(MyArray(jnp.array([1.0, 2.0])))
    assert jnp.array_equal(out.array, jnp.array([101.0, 102.0]))


@jax.custom_jvp
def f_grad(x):
    return jnp.sin(x)


@f_grad.defjvp
def f_grad_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sin(x), jnp.cos(x) * dx


@quax.register(f_grad)
def f_grad_myarray(x: MyArray) -> list[MyArray]:
    return [MyArray(jnp.sin(x.array) * 3.0)]


def test_rule_outer_grad():
    """Under an outer grad the rule, not the custom JVP, is differentiated."""
    g = jax.grad(lambda a: quax.quaxify(f_grad)(MyArray(a)).array)(0.5)
    assert jnp.allclose(g, 3.0 * jnp.cos(0.5))


@jax.custom_jvp
def f_mul(x, y):
    return x * y


@f_mul.defjvp
def f_mul_jvp(primals, tangents):
    (x, y), (dx, dy) = primals, tangents
    return x * y, x * dy + y * dx


@quax.register(f_mul)
def f_mul_myarray_arraylike(x: MyArray, y: ArrayLike) -> list[MyArray]:
    return [MyArray(x.array + y + 100.0)]


def test_rule_with_python_scalar_arg():
    out = quax.quaxify(f_mul)(MyArray(1.0), 2.0)
    assert isinstance(out, MyArray)
    assert out.array == 103.0


@jax.custom_jvp
def f_pair(x, y):
    return x * y


@f_pair.defjvp
def f_pair_jvp(primals, tangents):
    (x, y), (dx, dy) = primals, tangents
    return x * y, x * dy + y * dx


@quax.register(f_pair)
def f_pair_myarray_myarray(x: MyArray, y: MyArray) -> list[MyArray]:
    return [MyArray(x.array + 100.0)]


def test_type_miss_falls_back_to_inline():
    out = quax.quaxify(f_pair)(MyArray(2.0), jnp.array(3.0))
    assert isinstance(out, MyArray)
    assert out.array == 6.0


def test_unregistered_custom_jvp_inlines():
    @jax.custom_jvp
    def f_fresh(x):
        return jnp.sin(x)

    f_fresh.defjvp(lambda p, t: (jnp.sin(p[0]), jnp.cos(p[0]) * t[0]))
    out = quax.quaxify(f_fresh)(MyArray(1.0))
    assert jnp.allclose(out.array, jnp.sin(1.0))


@ft.partial(jax.custom_jvp, nondiff_argnums=(1,))
def f_nondiff(x, n):
    return x * n


@f_nondiff.defjvp
def f_nondiff_jvp(n, primals, tangents):
    (x,), (dx,) = primals, tangents
    return x * n, dx * n


@quax.register(f_nondiff)
def f_nondiff_myarray(x: MyArray) -> list[MyArray]:
    return [MyArray(x.array + 100.0)]


def test_nondiff_argnums_not_passed():
    assert quax.quaxify(f_nondiff)(MyArray(1.0), 3).array == 101.0


@jax.custom_jvp
def f_prec(x, y):
    return x * y


@f_prec.defjvp
def f_prec_jvp(primals, tangents):
    (x, y), (dx, dy) = primals, tangents
    return x * y, x * dy + y * dx


@quax.register(f_prec, precedence=1)
def f_prec_myarray_value(x: MyArray, y: quax.Value) -> list[MyArray]:
    return [MyArray(jnp.array(1.0))]


@quax.register(f_prec)
def f_prec_value_myarray(x: quax.Value, y: MyArray) -> list[MyArray]:
    return [MyArray(jnp.array(2.0))]


def test_precedence_picks_rule():
    assert quax.quaxify(f_prec)(MyArray(1.0), MyArray(1.0)).array == 1.0


@jax.custom_jvp
def f_count(x):
    return jnp.sin(x)


f_count.defjvp(lambda p, t: (jnp.sin(p[0]), jnp.cos(p[0]) * t[0]))


@quax.register(f_count)
def f_count_myarray(x: MyArray) -> list[MyArray]:
    return [x, x]


def test_wrong_output_count_raises():
    with pytest.raises(ValueError, match="expected 1"):
        quax.quaxify(f_count)(MyArray(1.0))


@jax.custom_jvp
def f_bare(x):
    return jnp.sin(x)


f_bare.defjvp(lambda p, t: (jnp.sin(p[0]), jnp.cos(p[0]) * t[0]))


@quax.register(f_bare)
def f_bare_myarray(x: MyArray) -> Any:
    return MyArray(x.array)  # not wrapped in a list


def test_bare_value_return_raises():
    with pytest.raises(TypeError, match="list or tuple"):
        quax.quaxify(f_bare)(MyArray(1.0))


def test_register_rejects_custom_vjp():
    @jax.custom_vjp
    def f_vjp(x):
        return x

    with pytest.raises(TypeError, match="custom_jvp"):
        quax.register(f_vjp)  # pyright: ignore[reportArgumentType]


def test_register_rejects_other_objects():
    with pytest.raises(TypeError, match="custom_jvp"):
        quax.register(lambda x: x)  # pyright: ignore[reportArgumentType]
