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
        return typeof(self.array)

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
    # By the time an elementwise `*_p` fires, JAX has already broadcast the
    # operands to the same rank, so two `NamedArray`s reaching here have
    # equal-length axis tuples (differing ranks fail earlier: `NamedArray` has
    # no `broadcast_in_dim` rule). The names must then match by position -- as
    # the `dot_general` rule validates its pairing, names check that operands
    # share matching semantics rather than reordering to align (see the module
    # README). A set-based check instead accepts reordered names (e.g. (A, B)
    # against (B, A)), which are then computed positionally and mislabelled.
    if len(axes1) == 0:
        return axes2
    if len(axes2) == 0:
        return axes1
    if axes1 != axes2:
        raise ValueError(f"Cannot broadcast named axes {axes1} against {axes2}.")
    return axes1


def _register_elementwise_binop(
    op: Callable[[Any, Any], Any], prim: jex.core.Primitive
):
    # Re-bind `prim` (rather than call the high-level `op`) so its parameters are
    # forwarded unchanged -- e.g. the `out_dtype` newer JAX threads through
    # `mul_p`, which the handlers must accept or dispatch fails with an
    # "unexpected keyword argument" error.
    bind = quax.quaxify(prim.bind)

    @quax.register(prim)
    def prim_named_array_named_array(
        x: NamedArray, y: NamedArray, **params: Any
    ) -> NamedArray:
        axes = _broadcast_axes(x.axes, y.axes)
        return NamedArray(bind(x.array, y.array, **params), axes)

    @quax.register(prim)
    def prim_array_like_named_array(
        x: ArrayLike | quax.ArrayValue, y: NamedArray, **params: Any
    ) -> NamedArray:
        if quax.quaxify(jnp.shape)(x) == ():
            return NamedArray(bind(x, y.array, **params), y.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")

    @quax.register(prim)
    def prim_named_array_array_like(
        x: NamedArray, y: ArrayLike | quax.ArrayValue, **params: Any
    ) -> NamedArray:
        if quax.quaxify(jnp.shape)(y) == ():
            return NamedArray(bind(x.array, y, **params), x.axes)
        else:
            raise ValueError(f"Cannot apply {op} to non-scalar array and named array.")


_register_elementwise_binop(lax.add, lax.add_p)
_register_elementwise_binop(lax.mul, lax.mul_p)
_register_elementwise_binop(lax.sub, lax.sub_p)


@quax.register(lax.dot_general_p)
def dot_general_named_array_named_array(
    lhs: NamedArray, rhs: NamedArray, *, dimension_numbers, **kwargs
) -> NamedArray:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    # `dot_general` pairs the contracted (and batched) axes positionally:
    # `lhs_contract[k]` is contracted with `rhs_contract[k]`. Compare the pairs,
    # not the axis-name *sets* -- a set comparison accepts a wrong pairing when
    # two dims share the same names in a different order (e.g. contracting
    # (A, B) against (B, A)), silently contracting mismatched axes. `strict=True`
    # additionally rejects a malformed `dimension_numbers` whose paired axis
    # tuples differ in length, rather than letting `zip` truncate and hide it.
    #
    # Only name-check when every axis index is in range; for an out-of-range
    # index (possible when binding `dot_general_p` directly) fall through to
    # `lax.dot_general` below, which raises a clear "dimension numbers ... less
    # than the number of axes" error rather than a bare `IndexError` from here.
    n_lhs, n_rhs = len(lhs.axes), len(rhs.axes)
    if all(0 <= i < n_lhs for i in (*lhs_contract, *lhs_batch)) and all(
        0 <= j < n_rhs for j in (*rhs_contract, *rhs_batch)
    ):
        if any(
            lhs.axes[i] != rhs.axes[j]
            for i, j in zip(lhs_contract, rhs_contract, strict=True)
        ):
            raise TypeError("Cannot contract mismatched dimensions.")
        if any(
            lhs.axes[i] != rhs.axes[j]
            for i, j in zip(lhs_batch, rhs_batch, strict=True)
        ):
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
