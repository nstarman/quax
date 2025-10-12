import functools as ft
from typing import Any, cast, get_args, TypeVar

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import quax


T = TypeVar("T")


class Zero(quax.ArrayValue):
    """Represents a symbolic zero value. Operations like `array + zero` will be
    reduced down to just `array` at trace time.

    This is essentially a generalised version of the symbolic zeros used by JAX in its
    autodifferentiation rules -- in our case, we can apply them at any time, not just
    during AD.
    """

    _shape: tuple[int, ...] = eqx.field(static=True)
    _dtype: Any = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], dtype: Any):
        """**Arguments:**

        - `shape`: the shape of the zero array.
        - `dtype`: the dtype of the zero array.
        """

        self._shape = shape
        self._dtype = dtype

    def aval(self):
        return jax.core.ShapedArray(self._shape, self._dtype)

    def materialise(self):
        return jnp.zeros(self._shape, self._dtype)


@quax.register(lax.broadcast_in_dim_p)
def _(
    value: ArrayLike, *, broadcast_dimensions, shape, sharding=None
) -> ArrayLike | quax.ArrayValue:
    return lax.broadcast_in_dim_p.bind(
        value,
        broadcast_dimensions=broadcast_dimensions,
        shape=shape,
        sharding=sharding,
    )


@quax.register(lax.broadcast_in_dim_p)
def _(value: Zero, *, broadcast_dimensions, shape, sharding=None) -> Zero:
    del broadcast_dimensions
    return Zero(shape, value.dtype)


@quax.register(lax.convert_element_type_p)
def convert_element_type_zero(
    value: Zero, *, new_dtype, weak_type, sharding=None
) -> Zero:
    # sharding was added around JAX 0.4.31, it seems.
    del weak_type, sharding
    return Zero(value.shape, new_dtype)


def _to_struct(x):
    if isinstance(x, quax.ArrayValue):
        aval = x.aval()
        return jax.ShapeDtypeStruct(aval.shape, aval.dtype)
    elif isinstance(x, get_args(ArrayLike)):
        return jax.ShapeDtypeStruct(jnp.shape(x), jnp.result_type(x))
    else:
        assert False


@quax.quaxify
def _shape_dtype(x, y, value: T) -> T:
    x = _to_struct(x)
    y = _to_struct(y)
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    dtype = jnp.result_type(x.dtype, y.dtype)
    return cast(T, jnp.broadcast_to(cast(ArrayLike, value), shape).astype(dtype))


@quax.register(lax.add_p)
def _(x: ArrayLike | quax.ArrayValue, y: Zero) -> ArrayLike | quax.ArrayValue:
    return _shape_dtype(x, y, value=x)


@quax.register(lax.add_p)
def _(x: Zero, y: ArrayLike | quax.ArrayValue) -> ArrayLike | quax.ArrayValue:
    return _shape_dtype(x, y, value=y)


@quax.register(lax.add_p)
def _(x: Zero, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=x)


@quax.register(lax.mul_p)
def _(x: ArrayLike | quax.ArrayValue, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=y)


@quax.register(lax.mul_p)
def _(x: Zero, y: ArrayLike | quax.ArrayValue) -> Zero:
    return _shape_dtype(x, y, value=x)


@quax.register(lax.mul_p)
def _(x: Zero, y: Zero) -> Zero:
    return _shape_dtype(x, y, value=x)


@quax.register(lax.dynamic_update_slice_p)
def _(operand: Zero, update: Zero, *indices) -> Zero:
    del update, indices
    return operand


@quax.register(lax.dynamic_slice_p)
def _(operand: Zero, *indices, slice_sizes) -> Zero:
    del indices
    return Zero(slice_sizes, operand.dtype)


@quax.register(lax.slice_p)
def _(operand: Zero, *, start_indices, limit_indices, strides) -> Zero:
    if strides is None:
        strides = [1 for _ in start_indices]
    shape = [
        (limit - start) // stride
        for start, limit, stride in zip(start_indices, limit_indices, strides)
    ]
    return Zero(tuple(shape), operand.dtype)


def _zero_matmul(lhs, rhs, kwargs) -> Zero:
    lhs = _to_struct(lhs)
    rhs = _to_struct(rhs)
    out_struct = jax.eval_shape(ft.partial(lax.dot_general_p.bind, **kwargs), lhs, rhs)
    return Zero(out_struct.shape, out_struct.dtype)


@quax.register(lax.dot_general_p)
def _(lhs: Zero, rhs: ArrayLike | quax.ArrayValue, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@quax.register(lax.dot_general_p)
def _(lhs: ArrayLike | quax.ArrayValue, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@quax.register(lax.dot_general_p)
def _(lhs: Zero, rhs: Zero, **kwargs) -> Zero:
    return _zero_matmul(lhs, rhs, kwargs)


@quax.register(lax.integer_pow_p)
def _integer_pow(x: Zero, *, y: int) -> Array | Zero:
    # Zero is a special case, because 0^0 = 1.
    if y == 0:
        return jnp.ones(x.shape, x.dtype)  # pyright: ignore

    # Otherwise, we can just return a zero.
    # Inf and NaN are not integers, so we don't need to worry about them.
    del y
    return x
