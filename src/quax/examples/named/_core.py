import dataclasses
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.core
import jax.extend as jex
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import ArrayLike

import quax
from quax._compat import typeof


@dataclasses.dataclass(frozen=True, eq=False)
class Axis:
    """Represents a named axis. Optionally can specify a fixed integer size for this
    axis.
    """

    size: int | None


Axis.__init__.__doc__ = """**Arguments:**

- `size`: either `None` (do not enforce that this axis take that particular size), or
    an integer (do enforce that this axis take that size -- when passed to `NamedArray`,
    throw an error if this is not the case).
"""


_Array = TypeVar("_Array", bound=ArrayLike)


class NamedArray(quax.ArrayValue, Generic[_Array]):
    """Represents an array, with each axis bound to a name."""

    array: _Array
    axes: tuple[Axis, ...] = eqx.field(static=True)
    allow_materialise: bool = eqx.field(default=False, static=True)

    def __check_init__(self):
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("Axis names for `NamedArray` must be unique.")
        if jnp.ndim(self.array) != len(self.axes):
            raise ValueError("`NamedArray` must have every axis be named.")
        for size, axis in zip(jnp.shape(self.array), self.axes):
            if axis.size is not None and size != axis.size:
                raise ValueError(f"Mismatched axis size for axis {axis}.")

    @property
    def shape(self):
        if self.allow_materialise:
            return super().shape
        else:
            raise RuntimeError(
                "Refusing to access the shape of a `NamedArray` with "
                "`allow_materialise=False`."
            )

    def materialise(self):
        if self.allow_materialise:
            return self.array
        else:
            raise RuntimeError(
                "Refusing to materialise `NamedArray` with `allow_materialise=False`."
            )

    def aval(self) -> jax.core.ShapedArray:
        return typeof(self.array)  # pyright: ignore

    def enable_materialise(self, allow_materialise: bool = True):
        return NamedArray(self.array, self.axes, allow_materialise)


NamedArray.__init__.__doc__ = """**Arguments:**

- `array`: the JAX array to wrap.
- `axes`: a tuple of `Axis`, that name each axis of `array`. It must be the case that
    `len(axes) == array.ndim`.
- `allow_materialise`: if Quax encounters an operation for which there has not been a
    specific override specified for named arrays, should it either (a) throw an error
    (`allow_materialise=False`, the default), or (b) silently convert the `NamedArray`
    back into an unnamed JAX array (`allow_materialise=True`).
"""


def _broadcast_axes(axes1, axes2):
    if len(axes1) == 0:
        return axes2
    elif len(axes2) == 0:
        return axes1
    elif set(axes1).issubset(set(axes2)):
        return axes2
    elif set(axes2).issubset(set(axes1)):
        return axes1
    else:
        raise ValueError(f"Cannot broadcast {axes1} against {axes2}")


def _register_elementwise_binop(
    op: Callable[[Any, Any], Any], prim: jex.core.Primitive
):
    quax_op = quax.quaxify(op)

    @quax.register(prim)
    def _(x: NamedArray, y: NamedArray) -> NamedArray:
        axes = _broadcast_axes(x.axes, y.axes)
        return NamedArray(quax_op(x.array, y.array), axes)

    @quax.register(prim)
    def _(x: ArrayLike | quax.ArrayValue, y: NamedArray) -> NamedArray:
        if quax.quaxify(jnp.shape)(x) == ():
            return NamedArray(quax_op(x, y.array), y.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")

    @quax.register(prim)
    def _(x: NamedArray, y: ArrayLike | quax.ArrayValue) -> NamedArray:
        if quax.quaxify(jnp.shape)(y) == ():
            return NamedArray(quax_op(x.array, y), x.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")


_register_elementwise_binop(lax.add, lax.add_p)
_register_elementwise_binop(lax.mul, lax.mul_p)
_register_elementwise_binop(lax.sub, lax.sub_p)


@quax.register(lax.dot_general_p)
def _(lhs: NamedArray, rhs: NamedArray, *, dimension_numbers, **kwargs) -> NamedArray:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    if {lhs.axes[i] for i in lhs_contract} != {rhs.axes[i] for i in rhs_contract}:
        raise TypeError("Cannot contract mismatched dimensions.")
    if {lhs.axes[i] for i in lhs_batch} != {rhs.axes[i] for i in rhs_batch}:
        raise TypeError("Cannot batch mismatched dimensions.")
    out = lax.dot_general(lhs.array, rhs.array, dimension_numbers, **kwargs)
    shared = tuple(lhs.axes[i] for i in lhs_batch)
    lhs_used = lhs_contract + lhs_batch
    rhs_used = rhs_contract + rhs_batch
    lhs_unused = tuple(axis for i, axis in enumerate(lhs.axes) if i not in lhs_used)
    rhs_unused = tuple(axis for i, axis in enumerate(rhs.axes) if i not in rhs_used)
    out_axes = shared + lhs_unused + rhs_unused
    return NamedArray(out, out_axes)


def trace(
    array: NamedArray,
    *,
    offset: int = 0,
    axis1: Axis,
    axis2: Axis,
    dtype=None,
) -> NamedArray:
    """As `jax.numpy.trace`, but supports specifying axes by name, not just by index.

    **Arguments:**

    - `array`: a `NamedArray`.
    - `offset`: Whether to offset above or below the main diagonal. Can be both positive
        and negative.
    - `axis1`: an `Axis` specifying the first axis to trace along. Must be a named axis
        of `array`.
    - `axis2`: an `Axis` specifying the second axis to trace along. Must be a named axis
        of `array`.
    - `dtype`: Determines the data-type of the returned array, and of the accmumulator
        when the elements are summed.

    **Returns:**

    An array without the `axis1` and `axis2` axes.
    """
    index1 = array.axes.index(axis1)
    index2 = array.axes.index(axis2)
    if index1 == index2:
        raise ValueError("Cannot trace along the same named axis.")
    inner = jnp.trace(
        array.array, offset=offset, axis1=index1, axis2=index2, dtype=dtype
    )
    axes = tuple(x for i, x in enumerate(array.axes) if i not in (index1, index2))
    return NamedArray(inner, axes)
