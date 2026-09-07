"""Interval arithmetic as a `quax.ArrayValue`.

A demonstration, not a library -- see the class docstring for what that means
here, and `src/quax/examples/interval/README.md` for the shape of the idea.
"""

from typing import Any

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Shaped

import quax


class Interval(quax.ArrayValue):
    """An array carrying a lower and an upper bound on each of its elements.

    Every operation maps the bounds forward, so the result brackets the result
    of the same computation on any values inside the input bounds. Feed in the
    uncertainty on your inputs and read the width of the output.

    !!! warning "The bounds get pessimistic, because operands are assumed independent"

        Each operation treats its operands as varying independently, so a value
        used twice is treated as two unrelated values. On `x = [-1, 2]`:

        - `x * x` gives `[-2, 4]`, because the rule for `mul` cannot see that
            both operands are the same `x`;
        - `x ** 2` gives `[0, 4]`, the true range, because `integer_pow` sees
            one operand and knows it is squaring it.

        Both answers are *sound* -- the true range is inside both -- but the
        first is wider than it needs to be. This is the dependency problem, and
        it is inherent to interval arithmetic rather than a defect here. It
        compounds along a computation, so a long expression over correlated
        inputs can end up with bounds too wide to be useful. Affine arithmetic
        is the standard remedy, and is a different representation rather than a
        better rule set.

    !!! warning "The bounds are approximate, not certified"

        Sound interval arithmetic rounds the lower bound down and the upper
        bound up at every step. JAX exposes no control over floating-point
        rounding mode, so this cannot do that: each bound carries the ordinary
        rounding error of one float operation. Fine for estimating error, not
        for verified computation.

    Refuses to [`quax.Value.materialise`][]: an operation with no rule is an
    error rather than a silently collapsed bound, which would be a wrong answer
    presented as a right one.

    **Arguments:**

    - `lo`: the lower bound.
    - `hi`: the upper bound. Must have the same shape as `lo`. That `lo <= hi`
        is *not* checked -- it depends on the values, so no JAX type can enforce
        it while tracing.
    """

    lo: Shaped[Array, "..."] = eqx.field(converter=jnp.asarray)
    hi: Shaped[Array, "..."] = eqx.field(converter=jnp.asarray)

    def __check_init__(self) -> None:
        if jnp.shape(self.lo) != jnp.shape(self.hi):
            msg = (
                f"`Interval` bounds must have the same shape, got "
                f"{jnp.shape(self.lo)} and {jnp.shape(self.hi)}."
            )
            raise ValueError(msg)

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(
            jnp.shape(self.lo), jnp.result_type(self.lo, self.hi)
        )

    def materialise(self) -> Any:
        msg = (
            "Refusing to materialise an `Interval`: there is no single array "
            "that carries the bounds, so the uncertainty would be dropped "
            "silently. Register a rule for this primitive instead."
        )
        raise ValueError(msg)

    @property
    def width(self) -> Array:
        """`hi - lo`, the size of the bracket on each element."""
        return self.hi - self.lo

    @property
    def midpoint(self) -> Array:
        """The centre of the bracket on each element."""
        return self.lo + self.width / 2


def _bounds(x: "Interval | ArrayLike", /) -> tuple[Any, Any]:
    """The bounds of `x`, treating a plain array as a zero-width interval."""
    return (x.lo, x.hi) if isinstance(x, Interval) else (x, x)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
#
# One per `lax` primitive. A plain array operand is a degenerate interval, so
# the mixed cases fall out of the same arithmetic; they are registered
# separately because Quax dispatches on the operand types.


@quax.register(lax.add_p)
def add_interval_interval(x: Interval, y: Interval, **kw: Any) -> Interval:
    return Interval(x.lo + y.lo, x.hi + y.hi)


@quax.register(lax.add_p)
def add_interval_array_like(x: Interval, y: ArrayLike, **kw: Any) -> Interval:
    return Interval(x.lo + y, x.hi + y)


@quax.register(lax.add_p)
def add_array_like_interval(x: ArrayLike, y: Interval, **kw: Any) -> Interval:
    return Interval(y.lo + x, y.hi + x)


@quax.register(lax.sub_p)
def sub_interval_interval(x: Interval, y: Interval, **kw: Any) -> Interval:
    # The upper bound of a difference pairs x's upper with y's *lower*.
    return Interval(x.lo - y.hi, x.hi - y.lo)


@quax.register(lax.sub_p)
def sub_interval_array_like(x: Interval, y: ArrayLike, **kw: Any) -> Interval:
    return Interval(x.lo - y, x.hi - y)


@quax.register(lax.sub_p)
def sub_array_like_interval(x: ArrayLike, y: Interval, **kw: Any) -> Interval:
    return Interval(x - y.hi, x - y.lo)


@quax.register(lax.neg_p)
def neg_interval(x: Interval, **kw: Any) -> Interval:
    return Interval(-x.hi, -x.lo)


def _mul_bounds(x: Any, y: Any, /) -> Interval:
    """The product of two brackets: the extremes lie at the corners.

    Any of the four endpoint products can be the extreme, depending on the
    signs, so take the elementwise minimum and maximum of all four.
    """
    x_lo, x_hi = _bounds(x)
    y_lo, y_hi = _bounds(y)
    corners = jnp.stack(
        jnp.broadcast_arrays(x_lo * y_lo, x_lo * y_hi, x_hi * y_lo, x_hi * y_hi)
    )
    return Interval(jnp.min(corners, axis=0), jnp.max(corners, axis=0))


@quax.register(lax.mul_p)
def mul_interval_interval(x: Interval, y: Interval, **kw: Any) -> Interval:
    return _mul_bounds(x, y)


@quax.register(lax.mul_p)
def mul_interval_array_like(x: Interval, y: ArrayLike, **kw: Any) -> Interval:
    # Not `Interval(x.lo * y, x.hi * y)`: a negative `y` swaps the bounds.
    return _mul_bounds(x, y)


@quax.register(lax.mul_p)
def mul_array_like_interval(x: ArrayLike, y: Interval, **kw: Any) -> Interval:
    return _mul_bounds(x, y)


@quax.register(lax.integer_pow_p)
def integer_pow_interval(x: Interval, *, y: int, **kw: Any) -> Interval:
    if y < 0:
        msg = (
            f"`Interval ** {y}` is a division, which is unbounded when the "
            "interval contains zero. Not supported."
        )
        raise NotImplementedError(msg)
    lo, hi = x.lo**y, x.hi**y
    out_lo, out_hi = jnp.minimum(lo, hi), jnp.maximum(lo, hi)
    if y % 2 == 0:
        # An even power is non-negative, and dips to zero wherever the bracket
        # straddles zero -- which is what the endpoints alone cannot tell you.
        straddles = (x.lo < 0) & (x.hi > 0)
        out_lo = jnp.where(straddles, jnp.zeros_like(out_lo), out_lo)
    return Interval(out_lo, out_hi)


@quax.register(lax.broadcast_in_dim_p)
def broadcast_in_dim_interval(operand: Interval, **kw: Any) -> Interval:
    kw.pop("sharding", None)
    return Interval(
        lax.broadcast_in_dim(operand.lo, **kw),
        lax.broadcast_in_dim(operand.hi, **kw),
    )


@quax.register(lax.convert_element_type_p)
def convert_element_type_interval(operand: Interval, **kw: Any) -> Interval:
    return Interval(
        lax.convert_element_type_p.bind(operand.lo, **kw),
        lax.convert_element_type_p.bind(operand.hi, **kw),
    )


@quax.register(lax.reduce_sum_p)
def reduce_sum_interval(operand: Interval, *, axes: tuple[int, ...], **kw: Any):
    # Summation is monotonic in each term, so the bounds sum independently.
    return Interval(
        lax.reduce_sum_p.bind(operand.lo, axes=axes, **kw),
        lax.reduce_sum_p.bind(operand.hi, axes=axes, **kw),
    )
