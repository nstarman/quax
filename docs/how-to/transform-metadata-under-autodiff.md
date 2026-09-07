# Transform metadata under autodiff

Quax gives a cotangent the primal's pytree structure, static fields included, so
metadata comes back through `jax.grad` unchanged. Usually that is what you want.
Sometimes it is wrong: a length differentiates to a per-length gradient, a
probability to a log-odds, a quantity in one frame to a covector in the dual.

To make metadata transform, move it out of the pytree and into the JAX *type*,
using [hijax](../hijax.md). The recipe below is the structure of
[`quax.examples.hijax`](../api/hijax.md); read that module for a complete
implementation, and adapt it. Its first three steps are excerpts from that
module rather than a program you can run as it stands.

## Put the metadata in a hijax type

Write a `HiType` for your value. It needs the three lowering methods — `lo_ty`,
`lower_val` and `raise_val`, which say which arrays the type is made of and how
values convert — plus `to_tangent_aval` and `to_ct_aval`. Those last two are the
point of the exercise: they are separate methods, so the cotangent type does not
have to be the primal type.

```py
def to_tangent_aval(self):
    return self  # a perturbation of a length is a length


def to_ct_aval(self):
    return QuantityTy(self.shape, self.dtype, inv_units(self.units))
```

If your tangent type is itself a hijax type, as here, also implement
`vspace_zero` and `vspace_add`, which autodiff uses to instantiate and
accumulate cotangents.

## Hold the hi value as a single leaf

A hijax value must not be a pytree, so it cannot be a `quax.Value` itself. Put
one inside a `quax.ArrayValue` as its only leaf. Subclass
[`HiValue`][quax.experimental.hijax.HiValue] and you get that for free: it
stores the leaf, and reports an `aval()` built from the leaf's hijax type.

```py
from quax.experimental.hijax import HiValue

class Unitful(HiValue):
    def __init__(self, array, units=()):
        self.leaf = array if isinstance(array, Quantity) else wrap(array, units)

    @property
    def units(self):
        return jax.typeof(self.leaf).units
```

Two things `HiValue` gets right that are easy to get wrong by hand. The aval
stays a plain `ShapedArray`: returning the `HiType` instead makes every `jnp`
function reject your tracers, because `jax.Array`'s `isinstance` check reads the
aval (see [Quax and hijax](../hijax.md#hijax-cannot-be-quaxs-base-class)). And
the leaf is never re-wrapped: a hijax tracer forwards `.shape` and `.dtype` from
its type, so testing `equinox.is_array_like` in `__init__` wraps a traced value
twice. Test for your hi value class, as above.

## Map `lax` primitives onto your hijax primitives

[`register_rules`][quax.experimental.hijax.register_rules] takes a mapping and
generates the dispatch rules, including the mixed operand combinations that a
binary primitive needs. All the metadata algebra and every autodiff rule live in
the hijax primitives, which is what lets JAX act on them.

```py
from quax.experimental.hijax import register_rules

register_rules(Unitful, {
    lax.mul_p: hi.mul,
    lax.add_p: hi.add,
    lax.integer_pow_p: lambda q, *, y, **kw: hi.int_pow(q, y),
    lax.reduce_sum_p: lambda q, *, axes, **kw: hi.sum(q, axes),
})
```

The number of operands is read from the function; the primitive's parameters go
to whichever keyword-only arguments it declares. Use a small lambda wherever the
two do not line up, as three of the four entries above do.

## Check that the metadata transformed

<!--- skip: start if(not have_hijax, 'hijax needs a newer JAX than quax does') -->
```python
import jax
import jax.numpy as jnp
import quax
from quax.examples.hijax import Unitful
from quax.examples.unitful import meters

length = Unitful(jnp.asarray(3.0), meters)


def squared(value):
    return quax.quaxify(lambda a: a * a)(value).array


print(squared(length))  # 9.0
print(jax.grad(squared)(length).units)  # ((m, -1),)
```

The gradient is per metre, not in metres. Compare
[`quax.examples.unitful`](../api/unitful.md), which is the same physics as a
pytree and returns `{m: 1}`.

## Add an operation

Registering a rule for an operation your hijax primitives already cover is
cheap. Negation is multiplication by `-1`:

```python
from typing import Any

import quax.examples.hijax as hijax_units


@quax.register(jax.lax.neg_p)
def neg_unitful(x: hijax_units.Unitful, **kw: Any) -> hijax_units.Unitful:
    return hijax_units.Unitful(hijax_units.mul(x.leaf, jnp.asarray(-1.0)))


print(quax.quaxify(lambda a: -a)(length).units)  # ((m, 1),)
```

<!--- skip: end -->

An operation your primitives do *not* cover is the expensive case: it needs a
new `HiPrim`, with a typing rule, an `expand`, and a rule for each transform you
intend to apply to it. Budget for that when deciding whether the type is worth
building this way — [Quax and hijax](../hijax.md#what-it-costs) weighs it up.

## Two things to avoid

**Do not construct the value class under a trace.** `Quantity(array, units)`
works eagerly, but under `jit` it hides a tracer inside a container JAX treats as
a constant, and the error surfaces much later somewhere unrelated. Apply a
primitive instead — that is what `wrap` is for.

**Do not read the value class's attributes under a trace either.** Traced code
holds a tracer of your hijax type, not an instance of it, so `q.array` raises
`AttributeError`. Attribute access is legal only inside `expand` and the type's
own methods; everywhere else, including your autodiff rules, go through
primitives.
