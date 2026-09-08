"""Interval arithmetic as a `quax.ArrayValue`.

A demonstration, not a library -- see the class docstring for what that means
here, and `src/quax/examples/interval/README.md` for the shape of the idea.
"""

from collections.abc import Sequence
from typing import Any, Final

import equinox as eqx
import jax.core
import jax.extend.core as jexc
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Shaped

import quax
from quax._values import ValueLike


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

    Operations with no registered rule fall to
    [`default`][quax.examples.interval.Interval.default], which handles anything
    monotone in its operands — see [`MONOTONE`][quax.examples.interval.MONOTONE].
    Anything else is an error, and [`quax.Value.materialise`][] refuses too: a
    silently collapsed bound would be a wrong answer presented as a right one.

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

    @staticmethod
    def default(
        primitive: jexc.Primitive,
        values: Sequence[ValueLike],
        params: dict[str, Any],
    ) -> Any:
        """Handle any primitive that is monotone in each of its interval operands.

        For those, mapping the primitive over `lo` and over `hi` separately is
        not just sound but *exact*, so one implementation covers a whole class
        of operations without a rule each; `MONOTONE` is the list.

        It is a list rather than "everything" because the same rule is
        *unsound* for anything else, and fails silently. Applied to `sin` over
        `[0, 2*pi]` it would give `[sin(0), sin(2*pi)]`, which is `[0, 0]` -- a
        confident claim that the value is exactly zero, when it ranges over the
        whole of `[-1, 1]`. `abs` and `neg` fail the same way; `neg` even
        returns its bounds inverted. So an unrecognised primitive raises.
        """
        if primitive not in MONOTONE:
            covered = ", ".join(sorted(p.name for p in MONOTONE))
            msg = (
                f"`Interval` has no rule for `{primitive.name}`, and it is not "
                f"monotone in its operands as far as this example knows, so "
                f"there is no sound bound to give. Register a rule for it. "
                f"Handled by default: {covered}."
            )
            raise ValueError(msg)
        if primitive is lax.select_n_p and isinstance(values[0], Interval):
            msg = (
                "`select_n` is monotone in its cases but not in its predicate, "
                "so an `Interval` predicate has no sound bound."
            )
            raise ValueError(msg)
        lo = [v.lo if isinstance(v, Interval) else v for v in values]
        hi = [v.hi if isinstance(v, Interval) else v for v in values]
        return Interval(primitive.bind(*lo, **params), primitive.bind(*hi, **params))

    @property
    def width(self) -> Array:
        """`hi - lo`, the size of the bracket on each element."""
        return self.hi - self.lo


MONOTONE: Final = frozenset(
    {
        # Rearrange, replicate or select elements. Monotone because each
        # output element is some input element, or the largest/smallest of
        # several.
        lax.broadcast_in_dim_p,
        lax.concatenate_p,
        lax.copy_p,
        lax.dynamic_slice_p,
        lax.gather_p,
        lax.reduce_max_p,
        lax.reduce_min_p,
        lax.rev_p,
        lax.reshape_p,
        lax.select_n_p,
        lax.slice_p,
        lax.squeeze_p,
        lax.transpose_p,
        # Increasing elementwise functions, so the bounds map straight across.
        lax.asinh_p,
        lax.atan_p,
        lax.cbrt_p,
        lax.convert_element_type_p,
        lax.erf_p,
        lax.exp_p,
        lax.log_p,
        lax.logistic_p,
        lax.sqrt_p,
        lax.tanh_p,
        # Increasing in every operand they combine.
        lax.add_p,
        lax.reduce_sum_p,
    }
)
"""The primitives that [`Interval.default`][] handles.

Membership means the primitive is monotonically non-decreasing in each of its
interval operands, which is what makes "apply it to `lo`, apply it to `hi`"
exact. Anything missing from it raises instead --
[`Interval.default`][quax.examples.interval.Interval.default] says why there is
no blanket fallback.
"""


def _bounds(x: "Interval | ArrayLike", /) -> tuple[Any, Any]:
    """The bounds of `x`, treating a plain array as a zero-width interval."""
    return (x.lo, x.hi) if isinstance(x, Interval) else (x, x)


# Rules, for the primitives that are *not* monotone in every operand; everything
# else goes through `Interval.default`. A plain operand is a degenerate interval,
# so the mixed cases fall out of the same arithmetic -- but each registration
# must still name `Interval` on at least one side, or it would capture ordinary
# array arithmetic inside every other quaxified function.


@quax.register(lax.sub_p)
def sub_any_interval(x: Interval | ArrayLike, y: Interval, **kw: Any) -> Interval:
    # The upper bound of a difference pairs x's upper with y's *lower*.
    x_lo, x_hi = _bounds(x)
    return Interval(x_lo - y.hi, x_hi - y.lo)


@quax.register(lax.sub_p)
def sub_interval_array_like(x: Interval, y: ArrayLike, **kw: Any) -> Interval:
    return Interval(x.lo - y, x.hi - y)


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
def mul_interval_any(x: Interval, y: Interval | ArrayLike, **kw: Any) -> Interval:
    # Not `Interval(x.lo * y, x.hi * y)`: a negative operand swaps the bounds.
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
    if y >= 2 and y % 2 == 0:
        # An even power is non-negative, and dips to zero wherever the bracket
        # straddles zero -- which is what the endpoints alone cannot tell you.
        # `y == 0` is even but constant at 1, so it must not take this branch.
        straddles = (x.lo < 0) & (x.hi > 0)
        out_lo = jnp.where(straddles, jnp.zeros_like(out_lo), out_lo)
    return Interval(out_lo, out_hi)
