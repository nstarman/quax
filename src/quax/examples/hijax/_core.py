"""The Quax layer: an `ArrayValue` whose leaf is a hijax `UnitfulArray`.

The hijax type in [`_unitful_array.py`][] is complete but unfriendly: hijax
types get no operations for free, so `jnp.sum(unitful_array)` is an error and
every unit-aware program has to be written against the primitives by hand.

Quax fixes exactly that, and [`quax.experimental.hijax`][] supplies the
plumbing. `Unitful` is a `HiValue` -- a `quax.ArrayValue`, so a pytree, holding
one `UnitfulArray` as its single leaf -- and `register_rules` maps the `lax`
primitives that `jnp` emits onto the hijax primitives. Unmodified array code
then runs, while the units, the typed jaxpr, and the inverted cotangent type all
live in the leaf's hijax type.
"""

__all__ = ("MAPPED", "Unitful")

from typing import Any

import jax
import jax.lax as lax

from quax.experimental.hijax import HiValue, register_rules

from . import _unitful_array as hi
from ._unitful_array import UnitfulArray, UnitfulArraySpec, Units, UnitsLike


class Unitful(HiValue):
    """An array with physical units, backed by a hijax type.

    Same idea as [`quax.examples.unitful.Unitful`][], different construction:
    the units live in the JAX *type* of the leaf rather than in a static pytree
    field. Two things follow that the pure-Quax version cannot do. The units
    appear in jaxprs, so a traced program is checked and readable. And a
    cotangent carries the inverse units, because the hijax type says so --
    differentiating a dimensionless loss with respect to a length gives a
    per-length gradient, where Quax alone must return the primal's own metadata
    (see [Autodiff](../autodiff.md#metadata-on-a-cotangent-is-the-primals)).

    Refuses to [`quax.Value.materialise`][], so a primitive with no registered
    rule is an error rather than a silent unit loss.

    **Arguments:**

    - `array`: the array to attach units to. An existing
        [`UnitfulArray`][quax.examples.hijax.UnitfulArray] is taken as-is.
    - `units`: either a single `Dimension`, or a dict from `Dimension` to
        integer exponent -- e.g. `{meters: 1, seconds: -2}` for an acceleration.
    """

    def __init__(self, array: Any, units: UnitsLike = (), /) -> None:
        # A hi value is held as-is; anything else is an array to attach units
        # to. The `isinstance` is true for a tracer of a unitful array as well
        # as for a `UnitfulArray` instance, which matters because the dispatch
        # rules construct `Unitful`s under a trace. Do not test
        # `eqx.is_array_like` here: such a tracer forwards `.shape` and
        # `.dtype` from its type, so it passes that check and would be wrapped
        # a second time.
        self.leaf = array if isinstance(array, UnitfulArray) else hi.wrap(array, units)

    @property
    def units(self) -> Units:
        """The units of the wrapped array, as a `Units` tuple."""
        return jax.typeof(self.leaf).units

    @property
    def array(self) -> Any:
        """The wrapped array, with the units dropped.

        Applies the `unwrap` primitive, so this is usable under a trace as well
        as eagerly.
        """
        return hi.unwrap(self.leaf)


MAPPED = UnitfulArraySpec()
"""The `vmap` axis entry for a mapped [`Unitful`][quax.examples.hijax.Unitful].

`vmap` will not guess how a hijax type is batched, so it takes a mapping spec
in place of an axis index, and it cannot infer the axis size from one either:

```python
import jax
import jax.numpy as jnp
import quax
from quax.examples.hijax import MAPPED, Unitful
from quax.examples.unitful import meters

xs = Unitful(jnp.arange(6.0).reshape(3, 2), meters)
out = jax.vmap(
    quax.quaxify(lambda a: a * a), in_axes=MAPPED, out_axes=MAPPED, axis_size=3
)(xs)
```

`axis_size` is only needed when every mapped argument is a `Unitful`; with a
plain array argument as well, `vmap` infers the size from it as usual. Putting
`vmap` inside the `quaxify` instead -- `quax.quaxify(jax.vmap(f))` -- needs
none of this, because `vmap` then sees plain arrays.
"""


# One entry per `lax` primitive that `jnp` emits for the supported operations.
# `register_rules` generates the dispatch rules from these -- including the
# mixed `Unitful`/array operand combinations for `mul` -- and each entry adapts
# the primitive's parameters to the hijax function's positional arguments. All
# the unit algebra, and every autodiff and batching rule, lives in the hijax
# layer, which is the point: that is where JAX can act on it.
register_rules(
    Unitful,
    {
        lax.mul_p: hi.mul,
        lax.add_p: hi.add,
        lax.integer_pow_p: lambda q, *, y, **kw: hi.int_pow(q, y),
        lax.reduce_sum_p: lambda q, *, axes, **kw: hi.sum(q, axes),
        lax.broadcast_in_dim_p: (
            lambda q, *, shape, broadcast_dimensions, **kw: hi.broadcast_in_dim(
                q, shape, broadcast_dimensions
            )
        ),
    },
)
