"""Tests for `quax.register` on a `jax.custom_jvp`, which swaps in a dispatchable body.

Each `custom_jvp` here is private to this module (registration mutates it), and
declares a derivative of `10 * cos`, so a test can tell the declared rule from
the derivative of the body or of the quax rule.
"""

import functools as ft

import jax
import jax.numpy as jnp

import quax

from .myarray import DenseArray, MyArray


a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
b = jnp.array([[10.0, 20.0], [30.0, 40.0]])
A, B = MyArray(a), MyArray(b)


def _sin_with_10x_jvp():
    @jax.custom_jvp
    def fn(x):
        return jnp.sin(x)

    fn.defjvp(lambda p, t: (fn(p[0]), jnp.cos(p[0]) * t[0] * 10.0))
    return fn


c1 = _sin_with_10x_jvp()


@quax.register(c1)
def c1_myarray(x: MyArray) -> MyArray:
    return MyArray(x.array + 100.0)


@jax.custom_jvp
def c2(x, y):
    return jnp.sin(x) * y


c2.defjvp(lambda p, t: (c2(*p), jnp.cos(p[0]) * p[1] * t[0] + jnp.sin(p[0]) * t[1]))


@quax.register(c2)
def c2_myarray_myarray(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(x.array + 100.0 * y.array)


def test_custom_jvp_rule_applies():
    assert jnp.allclose(quax.quaxify(c1)(A).array, a + 100.0)
    assert jnp.allclose(jax.jit(quax.quaxify(c1))(A).array, a + 100.0)
    assert jnp.allclose(quax.quaxify(jax.jit(c1))(A).array, a + 100.0)
    out = quax.quaxify(jax.vmap(c2, in_axes=(0, 1)))(A, B)
    expected = jax.vmap(lambda x, y: x + 100.0 * y, in_axes=(0, 1))(a, b)
    assert jnp.allclose(out.array, expected)


def test_custom_jvp_keeps_its_jvp_rule():
    assert jnp.allclose(c1(1.0), jnp.sin(1.0))
    assert jnp.allclose(jax.grad(c1)(1.0), 10.0 * jnp.cos(1.0))


def test_custom_jvp_inner_grad():
    out = quax.quaxify(jax.grad(c1))(MyArray(jnp.array(1.0)))
    out = out.array if isinstance(out, MyArray) else out
    assert jnp.allclose(out, 10.0 * jnp.cos(1.0))


def test_custom_jvp_outer_grad():
    grad = jax.grad(lambda x: quax.quaxify(c1)(MyArray(x)).array)(1.0)
    assert jnp.allclose(grad, 10.0 * jnp.cos(1.0))


c3 = _sin_with_10x_jvp()


@quax.register(c3)
def c3_myarray(x: MyArray) -> MyArray:
    return MyArray(x.array + 100.0)


_c3_body = c3.fun


@quax.register(c3)
def c3_densearray(x: DenseArray) -> DenseArray:
    return DenseArray(x.array + 200.0)


def test_custom_jvp_registered_twice_swaps_once():
    assert c3.fun is _c3_body
    assert jnp.allclose(quax.quaxify(c3)(A).array, a + 100.0)
    assert jnp.allclose(quax.quaxify(c3)(DenseArray(a)).array, a + 200.0)


def test_custom_jvp_registered_after_jit_cache():
    c4 = _sin_with_10x_jvp()
    jitted = jax.jit(c4)
    jitted(a)  # cache a trace of the original body

    @quax.register(c4)
    def c4_myarray(x: MyArray) -> MyArray:
        return MyArray(x.array + 100.0)

    assert jnp.allclose(quax.quaxify(jitted)(A).array, a + 100.0)


@ft.partial(jax.custom_jvp, nondiff_argnums=(1,))
def c5(x, n):
    return x * n


c5.defjvp(lambda n, p, t: (c5(p[0], n), t[0] * n))


@quax.register(c5)
def c5_myarray_int(x: MyArray, n: int) -> MyArray:
    return MyArray(x.array + 1000.0 * n)


def test_custom_jvp_nondiff_argnums():
    assert jnp.allclose(quax.quaxify(c5)(A, 3).array, a + 3000.0)
    assert jnp.allclose(c5(a, 3), a * 3)
