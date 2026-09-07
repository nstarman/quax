# Transform metadata under autodiff

Quax gives a cotangent the primal's pytree structure, static fields included, so
your metadata comes back through `jax.grad` unchanged. Usually that is what you
want. Sometimes it is wrong: a length differentiates to a per-length gradient, a
probability to a log-odds, a quantity in one frame to a covector in the dual.

To make metadata transform, move it off the pytree and into the JAX *type*. This
guide assumes you have read [Quax and hijax](../hijax.md) and decided the cost is
worth it.

## Check whether you need to build anything

If units are your case, the type already exists. Use it:

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
[`quax.examples.unitful`](../api/unitful.md), the same physics as a pytree, which
returns `{m: 1}`.

## Add an operation to a type that already exists

If your type is there but an operation is missing, add an entry to its
`register_rules` mapping rather than writing a rule by hand. Negation is
multiplication by `-1`, which the existing primitives already cover:

```python
import quax.examples.hijax as hijax_units
from quax.experimental.hijax import register_rules

register_rules(
    hijax_units.Unitful,
    {jax.lax.neg_p: lambda q, **kw: hijax_units.mul(q, jnp.asarray(-1.0))},
)

print(quax.quaxify(lambda a: -a)(length).units)  # ((m, 1),)
```

If the operation is *not* expressible with the existing primitives, you need a
new `HiPrim` first — a typing rule, an implementation, and a rule for each
transformation you will apply to it. Weigh that against
[what hijax costs](../hijax.md#what-it-costs) before committing.

<!--- skip: end -->

## Build a type of your own

Three steps. Copy the structure from
[`quax.examples.hijax`](../api/hijax.md), which is the reference implementation
for all of this.

**Write the hijax type.** Subclass `HiType` with the three lowering methods
`lo_ty`, `lower_val` and `raise_val` — which say which arrays your type is made
of, and how values convert — then add `to_tangent_aval` and `to_ct_aval`. Those
last two are the point of the exercise: they are separate methods, so your
cotangent type need not be your primal type. If the tangent type is itself a
hijax type, also implement `vspace_zero` and `vspace_add`, which autodiff uses to
instantiate and accumulate cotangents.

**Wrap it in a `HiValue`.** A hijax value cannot be a `quax.Value` itself, so
hold one as the single leaf of a [`HiValue`][quax.experimental.hijax.HiValue].
Give the subclass a friendly constructor and whatever read-only properties your
users need; the base class supplies the rest. Do not test
`equinox.is_array_like` in that constructor to decide whether to wrap — a hijax
tracer forwards `.shape` and `.dtype` from its type, so it passes that check and
gets wrapped twice. Test for your own hi value class instead.

**Map the primitives.** Pass
[`register_rules`][quax.experimental.hijax.register_rules] a
`{primitive: function}` mapping and it generates the dispatch rules, including
the mixed operand combinations a binary primitive needs. Where a primitive's
parameters do not line up with the hijax function's arguments, adapt with a
lambda, as the negation entry above does.

## Avoid two traps

**Do not construct your hi value class under a trace.** Calling it directly works
eagerly, but under `jit` it hides a tracer inside a container JAX treats as a
constant, and the error surfaces much later somewhere unrelated. Apply a
primitive instead — a `wrap`-style constructor primitive is the usual way.

**Do not read its attributes under a trace either.** Traced code holds a tracer
of your hijax type, not an instance of it, so attribute access raises
`AttributeError`. It is legal only inside a primitive's implementation and the
type's own methods. Everywhere else, including your autodiff rules, go through
primitives.
