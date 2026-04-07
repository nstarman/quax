from typing import cast

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

import quax
import quax.examples.named as named


def test_init(getkey):
    Foo = named.Axis(3)
    Bar = named.Axis(3)
    a = jr.normal(getkey(), (3, 3))
    a = named.NamedArray(a, (Foo, Bar))

    with pytest.raises(ValueError):
        a = jr.normal(getkey(), (3, 3))
        a = named.NamedArray(a, (Foo, Foo))

    b = jr.normal(getkey(), (3, 4))
    with pytest.raises(ValueError):
        b = named.NamedArray(b, (Foo, Bar))


def test_add(getkey):
    Foo = named.Axis(3)
    Bar = named.Axis(3)
    a = jr.normal(getkey(), (3, 3))
    a = named.NamedArray(a, (Foo, Bar))
    b = jr.normal(getkey(), (3, 3))
    b = named.NamedArray(b, (Foo, Bar))
    out1 = quax.quaxify(lambda x, y: x + y)(a, b)
    out2 = quax.quaxify(lax.add)(a, b)
    true_out = named.NamedArray(a.array + b.array, (Foo, Bar))
    assert out1 == true_out
    assert out2 == true_out


def test_matmul(getkey):
    Foo = named.Axis(3)
    Bar = named.Axis(3)
    Qux = named.Axis(None)
    a = jr.normal(getkey(), (3, 3))
    a = named.NamedArray(a, (Foo, Bar))
    b = jr.normal(getkey(), (3, 3))
    b = named.NamedArray(b, (Bar, Qux))

    quax.quaxify(lambda x, y: x @ y)(a, b)
    quax.quaxify(jnp.matmul)(a, b)

    with pytest.raises(TypeError, match="Cannot contract mismatched dimensions"):
        quax.quaxify(lambda x, y: x @ y)(b, a)
    with pytest.raises(TypeError, match="Cannot contract mismatched dimensions"):
        quax.quaxify(jnp.matmul)(b, a)


def test_existing_function(getkey):
    # We can use NamedArrays even in functions that weren't designed for it! The output
    # will be a NamedArray as well!
    # In this case, eqx.nn.Linear.__call__ was not written expecting a NamedArray. It
    # works anyway :)

    # Existing program
    linear = eqx.nn.Linear(3, 4, key=getkey())

    # Wrap our desired inputs into NamedArrays
    In = named.Axis(3)
    Out = named.Axis(4)
    named_bias = named.NamedArray(cast(Array, linear.bias), (Out,))
    named_weight = named.NamedArray(linear.weight, (Out, In))
    named_linear = eqx.tree_at(
        lambda l: (l.bias, l.weight), linear, (named_bias, named_weight)
    )
    vector = named.NamedArray(jr.normal(getkey(), (3,)), (In,))

    # Wrap function with quaxify.
    out = quax.quaxify(named_linear)(vector)
    # Output is a NamedArray!
    true_out = named.NamedArray(linear(vector.array), (Out,))
    assert out == true_out


def test_trace(getkey):
    A = named.Axis(None)
    B = named.Axis(None)
    C = named.Axis(None)
    x = jr.normal(getkey(), (2, 3, 4))
    named_x = named.NamedArray(x, (A, B, C))
    out = named.trace(named_x, axis1=A, axis2=C)
    true_out = jnp.trace(x, axis1=0, axis2=2)
    assert out.axes == (B,)
    assert jnp.array_equal(out.enable_materialise().materialise(), true_out)
