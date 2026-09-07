"""The hijax layer: a `UnitfulArray` value, its type, and its primitives.

Nothing here mentions Quax. This is an ordinary [hijax](https://docs.jax.dev/en/latest/301/hijax-types.html)
type: a value class that is *not* a pytree, a `HiType` describing it, and one
primitive per operation. [`_core.py`][] then wraps it in a `quax.ArrayValue` so
that unmodified `jnp` code can drive these primitives.

The reason to build units this way rather than as a plain `quax.ArrayValue`
(which [`quax.examples.unitful`][] already does) is `UnitfulArrayTy.to_ct_aval`: a
hijax type chooses its own cotangent type, so differentiating with respect to a
length can yield a per-length gradient. Quax alone cannot express that, because
a cotangent must reuse the primal's pytree structure -- static metadata
included.
"""

__all__ = (
    "UnitfulArray",
    "UnitfulArraySpec",
    "UnitfulArrayTy",
    "Units",
    "add",
    "broadcast_in_dim",
    "int_pow",
    "mul",
    "sum",
    "unwrap",
    "wrap",
)

import builtins
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import jax
import jax._src.core as jax_core
import jax.numpy as jnp
from jaxtyping import ArrayLike

from quax.experimental.hijax import (
    HiPrim,
    HiType,
    instantiate_zeros,
    MappingSpec,
    register_hitype,
    ShapedArray,
)

from ..unitful import Dimension


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

Units: TypeAlias = tuple[tuple[Dimension, int], ...]
"""Physical units, as `(dimension, exponent)` pairs sorted by dimension name.

A `HiType` must be hashable, so units are a sorted tuple rather than the `dict`
that [`quax.examples.unitful.Unitful`][] uses. `to_units` accepts either form.
"""

UnitsLike: TypeAlias = "Dimension | dict[Dimension, int] | Units"


def to_units(units: UnitsLike, /) -> Units:
    """Canonicalise a units specification into a `Units` tuple.

    Accepts a bare `Dimension` (exponent 1), a `{Dimension: exponent}` dict, or
    an already-canonical tuple. Zero exponents are dropped, so `m / m` is
    dimensionless rather than `m^0`.
    """
    if isinstance(units, Dimension):
        items: Any = ((units, 1),)
    elif isinstance(units, dict):
        items = units.items()
    else:
        items = units
    return tuple(sorted(((d, n) for d, n in items if n != 0), key=lambda p: p[0].name))


def mul_units(a: Units, b: Units, /) -> Units:
    """Multiply two units: add their exponents."""
    merged: dict[Dimension, int] = dict(a)
    for dim, exponent in b:
        merged[dim] = merged.get(dim, 0) + exponent
    return to_units(merged)


def pow_units(a: Units, n: int, /) -> Units:
    """Raise units to an integer power: multiply their exponents."""
    return to_units({d: e * n for d, e in a})


def inv_units(a: Units, /) -> Units:
    """Invert units: negate their exponents."""
    return pow_units(a, -1)


def str_units(a: Units, /) -> str:
    """Format units for a type's short string, e.g. `m s^-2`. Empty is `1`."""
    return " ".join(f"{d}^{e}" if e != 1 else f"{d}" for d, e in a) or "1"


# ---------------------------------------------------------------------------
# The value and its type
# ---------------------------------------------------------------------------


class _UnitfulArrayMeta(type):
    """Make `isinstance(x, UnitfulArray)` true for a tracer of a `UnitfulArray` too.

    Under a trace a unitful array is a `Tracer` of type `u[...]{...}`, not a
    `UnitfulArray` instance, so a plain `isinstance` check would answer "no" in
    exactly the code that has to work under tracing. Hijax's own `Box` and `Log`
    types use this same metaclass trick.
    """

    def __instancecheck__(cls, instance: Any) -> bool:
        return super().__instancecheck__(instance) or (
            isinstance(instance, jax_core.Tracer)
            and isinstance(jax.typeof(instance), UnitfulArrayTy)
        )


@dataclass(frozen=True)
class UnitfulArray(metaclass=_UnitfulArrayMeta):
    """An array with units, as hijax sees it.

    Deliberately *not* a pytree: JAX flattens a pytree into its leaves before it
    consults the hijax type registry, so a value that is a pytree can never
    carry a hijax type. It follows that this class is opaque to `jax.tree`
    utilities, to `equinox.filter_*`, and to anything else that walks pytrees --
    which is exactly why [`Unitful`][quax.examples.hijax.Unitful] exists to hold
    one.

    Construct these with [`wrap`][quax.examples.hijax.wrap] rather than by
    calling this class: under a trace, applying the primitive is what keeps the
    value in the jaxpr, whereas calling the constructor smuggles a tracer into a
    container JAX believes is a constant.
    """

    array: ArrayLike
    """The underlying array.

    `ArrayLike` rather than `Array`: while lowering, JAX hands `expand` the
    constants it folded out of the jaxpr, which are NumPy rather than JAX
    arrays.
    """

    units: Units

    def __repr__(self) -> str:
        return f"UnitfulArray({self.array}, {str_units(self.units)})"


@dataclass(frozen=True)
class UnitfulArraySpec(MappingSpec):
    """How a [`UnitfulArray`][quax.examples.hijax.UnitfulArray] is mapped by `vmap`.

    A batch of unitful arrays is one bigger unitful array with the same units,
    so the
    leading axis is the only mapping there is and this spec carries no data.
    Pass it as a `vmap` `in_axes`/`out_axes` entry, or use
    [`MAPPED`][quax.examples.hijax.MAPPED] for the `Unitful` that holds one.
    """


@dataclass(frozen=True)
class UnitfulArrayTy(HiType):
    """The hijax type of a [`UnitfulArray`][quax.examples.hijax.UnitfulArray].

    Prints in jaxprs as e.g. `u[3]{m s^-1}`. Equality includes the units, so
    JAX's own type checking rejects a rule that returns the wrong units.
    """

    shape: tuple[int, ...]
    dtype: Any
    units: Units

    # -- lowering: the arrays this type is made of, and how values convert
    def lo_ty(self) -> list[Any]:  # list[ShapedArray]
        return [ShapedArray(self.shape, self.dtype)]

    def lower_val(self, value: UnitfulArray) -> list[ArrayLike]:
        return [value.array]

    def raise_val(self, array: ArrayLike) -> UnitfulArray:
        return UnitfulArray(array, self.units)  # pyright: ignore[reportArgumentType]

    # -- autodiff.
    # A perturbation of a length is a length, so the tangent type is this type.
    # A cotangent is not: d(dimensionless)/d(x) is measured per unit of `x`, so
    # the cotangent type carries the *inverse* units. This asymmetry is the
    # whole reason for building units on hijax; see the module docstring.
    def to_tangent_aval(self) -> "UnitfulArrayTy":
        return self

    def to_ct_aval(self) -> "UnitfulArrayTy":
        return UnitfulArrayTy(self.shape, self.dtype, inv_units(self.units))

    # Required because the tangent type is itself a hi type: autodiff needs to
    # instantiate and accumulate values of it.
    def vspace_zero(self) -> UnitfulArray:
        return UnitfulArray(jnp.zeros(self.shape, self.dtype), self.units)

    def vspace_add(self, x: UnitfulArray, y: UnitfulArray) -> UnitfulArray:
        return add(x, y)

    # -- vmap, and through it scan
    def dec_rank(self, size: int | None, spec: Any) -> "UnitfulArrayTy":
        return UnitfulArrayTy(self.shape[1:], self.dtype, self.units)

    def inc_rank(self, size: int | None, spec: Any) -> "UnitfulArrayTy":
        return UnitfulArrayTy((size, *self.shape), self.dtype, self.units)  # pyright: ignore[reportArgumentType]

    def leading_axis_spec(self) -> UnitfulArraySpec:
        return UnitfulArraySpec()

    # -- printing
    def str_short(self, short_dtypes: bool = False, **kwargs: Any) -> str:
        dims = ",".join(map(str, self.shape))
        return f"u[{dims}]{{{str_units(self.units)}}}"

    __repr__ = str_short


register_hitype(
    UnitfulArray, lambda q: UnitfulArrayTy(q.array.shape, q.array.dtype, q.units)
)


def units_of(aval: Any, /) -> Units:
    """The units of a hijax aval; a plain array type is dimensionless."""
    return aval.units if isinstance(aval, UnitfulArrayTy) else ()


def _lower(aval: Any, value: Any, /) -> Any:
    """The array inside `value`, which is a `UnitfulArray` only if `aval` says so.

    Primitives here accept a mix of `UnitfulArrayTy` and plain array operands, so
    `expand` and the rules need to know which is which. The aval is fixed at
    primitive-construction time, which makes it the reliable place to ask.
    """
    return value.array if isinstance(aval, UnitfulArrayTy) else value


def _matching_ct(aval: Any, ct: UnitfulArray, /) -> Any:
    """Reshape and retype a cotangent to what the input type `aval` demands.

    Two corrections, both of which JAX type-checks. An operand that broadcast
    against a larger one gets a cotangent of the larger shape, which must be
    summed back down. And a plain array input has a plain array cotangent, so
    the units come off.
    """
    ct_shape = jax.typeof(ct).shape
    if aval.shape != ct_shape:
        if aval.shape != ():
            msg = (
                f"no transpose rule for an operand of shape {aval.shape} "
                f"broadcast to {ct_shape}. Only a scalar operand is supported."
            )
            raise NotImplementedError(msg)
        ct = sum(ct)
    return ct if isinstance(aval, UnitfulArrayTy) else unwrap(ct)


def _result_type(x_aval: Any, y_aval: Any, units: Units, /) -> UnitfulArrayTy:
    """The output type of a binary elementwise primitive.

    A scalar operand broadcasts against an array one, which is what `jnp` emits
    for `0.5 * x`. Anything more general is rejected here rather than failing
    later in a transpose rule that cannot undo it.
    """
    shape = jnp.broadcast_shapes(x_aval.shape, y_aval.shape)
    if () not in (x_aval.shape, y_aval.shape) and x_aval.shape != y_aval.shape:
        msg = (
            f"cannot broadcast {x_aval.shape} against {y_aval.shape}: only a "
            "scalar operand broadcasts. Use `broadcast_in_dim` explicitly."
        )
        raise TypeError(msg)
    dtype = jnp.promote_types(x_aval.dtype, y_aval.dtype)
    return UnitfulArrayTy(shape, dtype, units)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
#
# Every operation on a UnitfulArray is its own primitive: a hijax type gets no
# implementations for free, and `jnp.sin(unitful_array)` is an error rather than a
# silent unit loss. Each declares its input and output types, gives the
# implementation in `expand`, and carries the rules for the transforms it
# supports (`jvp` for forward mode, `vjp_fwd`/`vjp_bwd_retval` for reverse,
# `batch` for `vmap` and hence `scan`).


def _batched(size: int, args: Sequence[Any], dims: Sequence[Any]) -> tuple[Any, Any]:
    """Align a batch rule's arguments to a common leading batch axis.

    Returns the aligned arguments and the output mapping spec. A batch rule is
    invoked even when nothing is batched, in which case there is no batch axis
    to report and the result is `(args, None)`.

    An unbatched operand alongside batched ones is given a leading size-`size`
    axis. For a `UnitfulArray` that needs a primitive -- `q[None]` would read an
    attribute off a value that may be a tracer.
    """
    if all(d is None for d in dims):
        return tuple(args), None
    aligned = []
    for arg, dim in zip(args, dims, strict=True):
        if dim is None:
            aval = jax.typeof(arg)
            shape = (size, *aval.shape)
            bdims = tuple(range(1, len(aval.shape) + 1))
            arg = (
                broadcast_in_dim(arg, shape, bdims)
                if isinstance(aval, UnitfulArrayTy)
                else jnp.broadcast_to(arg, shape)
            )
        elif isinstance(dim, int):
            arg = jnp.moveaxis(arg, dim, 0)
        aligned.append(arg)
    return tuple(aligned), UnitfulArraySpec()


class Wrap(HiPrim):
    """`array -> UnitfulArray`, attaching static units."""

    def __init__(self, x_aval: Any, units: Units) -> None:
        self.in_avals = (x_aval,)
        self.out_aval = UnitfulArrayTy(x_aval.shape, x_aval.dtype, units)
        self.params = {"units": units}
        super().__init__()

    def expand(self, x: ArrayLike) -> UnitfulArray:
        return UnitfulArray(x, self.units)  # pyright: ignore[reportArgumentType]

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (x,), (t,) = primals, tangents
        return self(x), wrap(instantiate_zeros(t), self.units)

    def vjp_fwd(self, nzs_in: Any, x: Any) -> Any:
        return self(x), None

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        # `g` has the inverse units; the input is a plain array, so drop them.
        return (unwrap(instantiate_zeros(g)),)

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (x,), (d,) = args, dims
        if d is None:
            return self(x), None
        return wrap(jnp.moveaxis(x, d, 0), self.units), UnitfulArraySpec()


class Unwrap(HiPrim):
    """`UnitfulArray -> array`, discarding the units."""

    def __init__(self, q_aval: UnitfulArrayTy) -> None:
        self.in_avals = (q_aval,)
        self.out_aval = ShapedArray(q_aval.shape, q_aval.dtype)
        self.params = {}
        super().__init__()

    def expand(self, q: UnitfulArray) -> ArrayLike:
        return q.array

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (q,), (t,) = primals, tangents
        return self(q), unwrap(instantiate_zeros(t))

    def vjp_fwd(self, nzs_in: Any, q: Any) -> Any:
        return self(q), None

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        (q_aval,) = self.in_avals
        return (wrap(instantiate_zeros(g), inv_units(q_aval.units)),)

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (q,), (d,) = args, dims
        return self(q), (None if d is None else 0)


class Add(HiPrim):
    """Add two unitful arrays. Their units must match."""

    def __init__(self, x_aval: Any, y_aval: Any) -> None:
        x_units, y_units = units_of(x_aval), units_of(y_aval)
        if x_units != y_units:
            msg = (
                f"cannot add {str_units(x_units)!r} to {str_units(y_units)!r}: "
                "units differ."
            )
            raise TypeError(msg)
        self.in_avals = (x_aval, y_aval)
        self.out_aval = _result_type(x_aval, y_aval, x_units)
        self.params = {}
        super().__init__()

    def expand(self, x: Any, y: Any) -> UnitfulArray:
        x_aval, y_aval = self.in_avals
        array = _lower(x_aval, x) + _lower(y_aval, y)
        return UnitfulArray(array, self.out_aval.units)

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (x, y), (tx, ty) = primals, tangents
        tangent = add(instantiate_zeros(tx), instantiate_zeros(ty))
        return self(x, y), tangent

    def vjp_fwd(self, nzs_in: Any, x: Any, y: Any) -> Any:
        return self(x, y), None

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        g = instantiate_zeros(g)
        return tuple(_matching_ct(aval, g) for aval in self.in_avals)

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (x, y), spec = _batched(axis_data.size, args, dims)
        return add(x, y), spec


class Mul(HiPrim):
    """Multiply two unitful arrays, or one and a plain array."""

    def __init__(self, x_aval: Any, y_aval: Any) -> None:
        self.in_avals = (x_aval, y_aval)
        units = mul_units(units_of(x_aval), units_of(y_aval))
        self.out_aval = _result_type(x_aval, y_aval, units)
        self.params = {}
        super().__init__()

    def expand(self, x: Any, y: Any) -> UnitfulArray:
        x_aval, y_aval = self.in_avals
        array = _lower(x_aval, x) * _lower(y_aval, y)
        return UnitfulArray(array, self.out_aval.units)

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (x, y), (tx, ty) = primals, tangents
        x_aval, y_aval = self.in_avals
        tangent = add(
            mul(instantiate_zeros(tx), y),
            mul(x, instantiate_zeros(ty)),
        )
        return self(x, y), tangent

    def vjp_fwd(self, nzs_in: Any, x: Any, y: Any) -> Any:
        return self(x, y), (x, y)

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        x, y = res
        x_aval, y_aval = self.in_avals
        g = instantiate_zeros(g)
        # `g` carries 1/(xy), so `g * y` carries 1/x and `g * x` carries 1/y:
        # exactly the cotangent units each input's type asks for.
        return (_matching_ct(x_aval, mul(g, y)), _matching_ct(y_aval, mul(x, g)))

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (x, y), spec = _batched(axis_data.size, args, dims)
        return mul(x, y), spec


class IntPow(HiPrim):
    """Raise a unitful array to a static integer power, units included."""

    def __init__(self, q_aval: UnitfulArrayTy, y: int) -> None:
        self.in_avals = (q_aval,)
        self.out_aval = UnitfulArrayTy(
            q_aval.shape, q_aval.dtype, pow_units(q_aval.units, y)
        )
        self.params = {"y": y}
        super().__init__()

    def expand(self, q: UnitfulArray) -> UnitfulArray:
        return UnitfulArray(q.array**self.y, self.out_aval.units)

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (q,), (t,) = primals, tangents
        return self(q), self._derivative(q, instantiate_zeros(t))

    def vjp_fwd(self, nzs_in: Any, q: Any) -> Any:
        return self(q), q

    def vjp_bwd_retval(self, q: Any, g: Any) -> Any:
        return (self._derivative(q, instantiate_zeros(g)),)

    def _derivative(self, q: Any, other: Any) -> Any:
        """`y * q**(y - 1) * other`, which serves both the jvp and the vjp.

        As a tangent this carries `u^y`; as a cotangent, `u^(y-1) * u^-y` is
        `u^-1`. One expression, both units correct, because the chain rule and
        the units algebra are the same multiplication.
        """
        scale = wrap(jnp.asarray(self.y, jax.typeof(other).dtype), ())
        return mul(scale, mul(int_pow(q, self.y - 1), other))

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (q,), spec = _batched(axis_data.size, args, dims)
        return int_pow(q, self.y), spec


class Sum(HiPrim):
    """Sum a unitful array over `axes`. Units are unchanged."""

    def __init__(self, q_aval: UnitfulArrayTy, axes: tuple[int, ...]) -> None:
        self.in_avals = (q_aval,)
        shape = tuple(d for i, d in enumerate(q_aval.shape) if i not in axes)
        self.out_aval = UnitfulArrayTy(shape, q_aval.dtype, q_aval.units)
        self.params = {"axes": axes}
        super().__init__()

    def expand(self, q: UnitfulArray) -> UnitfulArray:
        return UnitfulArray(jnp.sum(q.array, axis=self.axes), q.units)

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (q,), (t,) = primals, tangents
        return self(q), sum(instantiate_zeros(t), self.axes)

    def vjp_fwd(self, nzs_in: Any, q: Any) -> Any:
        return self(q), None

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        # Transpose of a sum: broadcast back over the axes that were reduced.
        (q_aval,) = self.in_avals
        kept = tuple(i for i in range(len(q_aval.shape)) if i not in self.axes)
        return (broadcast_in_dim(instantiate_zeros(g), q_aval.shape, kept),)

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (q,), spec = _batched(axis_data.size, args, dims)
        if spec is None:
            return self(q), None
        # Axis 0 is now the batch; every axis this reduced has shifted up one.
        return sum(q, tuple(a + 1 for a in self.axes)), spec


class BroadcastInDim(HiPrim):
    """Broadcast a unitful array to `shape`, as `lax.broadcast_in_dim` does."""

    def __init__(
        self,
        q_aval: UnitfulArrayTy,
        shape: tuple[int, ...],
        broadcast_dimensions: tuple[int, ...],
    ) -> None:
        self.in_avals = (q_aval,)
        self.out_aval = UnitfulArrayTy(shape, q_aval.dtype, q_aval.units)
        self.params = {"shape": shape, "broadcast_dimensions": broadcast_dimensions}
        super().__init__()

    def expand(self, q: UnitfulArray) -> UnitfulArray:
        array = jax.lax.broadcast_in_dim(q.array, self.shape, self.broadcast_dimensions)
        return UnitfulArray(array, q.units)

    def jvp(self, primals: Any, tangents: Any) -> Any:
        (q,), (t,) = primals, tangents
        tangent = broadcast_in_dim(
            instantiate_zeros(t), self.shape, self.broadcast_dimensions
        )
        return self(q), tangent

    def vjp_fwd(self, nzs_in: Any, q: Any) -> Any:
        return self(q), None

    def vjp_bwd_retval(self, res: Any, g: Any) -> Any:
        # Transpose of a broadcast: sum over the axes it introduced, then undo
        # any size-1 axis it stretched.
        (q_aval,) = self.in_avals
        bdims = self.broadcast_dimensions
        summed = tuple(i for i in range(len(self.shape)) if i not in bdims)
        stretched = tuple(
            i for i, d in zip(bdims, q_aval.shape, strict=True) if d != self.shape[i]
        )
        if stretched:
            msg = (
                "no transpose rule for a broadcast that stretches a size-1 axis "
                f"(axes {stretched} of {q_aval.shape} -> {self.shape})."
            )
            raise NotImplementedError(msg)
        return (sum(instantiate_zeros(g), summed),)

    def batch(self, axis_data: Any, args: Any, dims: Any) -> Any:
        (q,), spec = _batched(axis_data.size, args, dims)
        if spec is None:
            return self(q), None
        shape = (axis_data.size, *self.shape)
        bdims = (0, *(d + 1 for d in self.broadcast_dimensions))
        return broadcast_in_dim(q, shape, bdims), spec


# ---------------------------------------------------------------------------
# Applying the primitives
# ---------------------------------------------------------------------------
#
# Each primitive's types are fixed when it is constructed, so the idiom is to
# build one from the argument types and immediately apply it.


def wrap(x: ArrayLike, units: UnitsLike = (), /) -> UnitfulArray:
    """Attach `units` to an array, giving a
    [`UnitfulArray`][quax.examples.hijax.UnitfulArray].

    ```python
    from quax.examples.hijax import wrap
    from quax.examples.unitful import meters
    import jax.numpy as jnp

    wrap(jnp.asarray(2.0), meters)  # UnitfulArray(2.0, m)
    ```
    """
    # A Python scalar reaches here as a typed literal, which has no `.shape`.
    x = jnp.asarray(x)
    return Wrap(jax.typeof(x), to_units(units))(x)


def unwrap(q: UnitfulArray, /) -> ArrayLike:
    """Drop the units from a [`UnitfulArray`][quax.examples.hijax.UnitfulArray]."""
    return Unwrap(jax.typeof(q))(q)


def add(x: Any, y: Any, /) -> UnitfulArray:
    """Add two unitful arrays of the same units and shape."""
    return Add(jax.typeof(x), jax.typeof(y))(x, y)


def mul(x: Any, y: Any, /) -> UnitfulArray:
    """Multiply two unitful arrays, or one and a plain array."""
    return Mul(jax.typeof(x), jax.typeof(y))(x, y)


def int_pow(q: UnitfulArray, y: int, /) -> UnitfulArray:
    """Raise a unitful array to a static integer power."""
    return IntPow(jax.typeof(q), y)(q)


def sum(q: UnitfulArray, axes: tuple[int, ...] | None = None, /) -> UnitfulArray:  # noqa: A001
    """Sum a unitful array over `axes`, or over every axis if `axes` is `None`."""
    if axes is None:
        axes = tuple(builtins.range(len(jax.typeof(q).shape)))
    return Sum(jax.typeof(q), tuple(axes))(q)


def broadcast_in_dim(
    q: UnitfulArray, shape: tuple[int, ...], broadcast_dimensions: tuple[int, ...], /
) -> UnitfulArray:
    """Broadcast a unitful array, as `jax.lax.broadcast_in_dim` does for arrays."""
    return BroadcastInDim(jax.typeof(q), tuple(shape), tuple(broadcast_dimensions))(q)
