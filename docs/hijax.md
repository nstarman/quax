# Quax and hijax

JAX has its own extension API for custom types,
[hijax](https://docs.jax.dev/en/latest/301/hijax-types.html). If you are choosing
between it and Quax, or wondering whether you need both, this page is the answer.

They are not competitors. Quax runs *existing* code on your type: you do not own
`jnp.mean`, or the solver inside Diffrax, or the model someone else wrote, and
Quax makes them work with your array-ish object. Hijax gives a *new* type its own
operations, and nothing existing applies to it — `jnp.sin` on a hijax value is an
error rather than a silent loss of whatever the value was carrying.

So the question is not which is better. It is which problem you have.

## Which should you use?

Two questions settle it, and the first usually settles it alone.

**Who writes the code that touches your type?** If the answer includes anyone but
you, you need Quax. A hijax type is invisible to code that was not written for
it. This is the common case.

**Does the type itself have to do something Quax cannot?** Quax carries your
metadata on a pytree, and JAX never acts on metadata. Four things follow that
Quax cannot reach:

- a cotangent whose type differs from the primal's, so metadata transforms under
  differentiation instead of riding along unchanged;
- invariants checked while tracing, and printed in the jaxpr;
- batching semantics of your own, rather than a per-leaf axis index;
- sharding carried in the type, and a `shard_map` boundary that understands it.

That gives three answers.

**Quax alone** is the default. It works on every JAX from 0.7.2, your value stays
a pytree so `eqx.filter_*`, Optax and `jax.tree` keep working, and an operation
you have not written a rule for falls back to `materialise` rather than failing.
Metadata that only rides along — a name, a provenance tag, units you check but
never differentiate — belongs here.

**Hijax alone** suits a type you own end to end that is not array-shaped anyway:
a mutable cell, a log, a handle. JAX ships `Box` and `Log` as exactly this. It is
also right when an operation needs a lowering of its own, such as a fused kernel.

**Both** — a hijax value held as the single leaf of a `quax.ArrayValue` — when
existing code has to run *and* you need one of those four capabilities.
[`quax.experimental.hijax`](api/experimental.md) is the plumbing, and
[`quax.examples.hijax`](api/hijax.md) is a worked type built on it.

If you are unsure, start with Quax alone. Moving to the combination later changes
how your rules are implemented, not what your users write.

## What the combination buys

Quax cannot change metadata under differentiation. A cotangent must reuse the
primal's pytree structure, static fields included, so a length in metres
differentiates to a cotangent in metres — see
[Autodiff](autodiff.md#metadata-on-a-cotangent-is-the-primals). That is JAX's
contract, and no rule you write escapes it.

A hijax type is not metadata on a pytree; it is a *type*, and a type says what
its cotangent type is. Compare the two units libraries shipped here, which model
the same physics and differ only in where the units live:

<!--- skip: start if(not have_hijax, 'hijax needs a newer JAX than quax does') -->
```python
import jax
import jax.numpy as jnp
import quax
import quax.examples.hijax as hijax_units
import quax.examples.unitful as pytree_units
from quax.examples.unitful import meters


def squared(value):
    """A dimensionless number from a length, for differentiating."""
    return quax.quaxify(lambda a: a * a)(value).array


pytree_length = pytree_units.Unitful(jnp.asarray(3.0), meters)
hijax_length = hijax_units.Unitful(jnp.asarray(3.0), meters)

print(jax.grad(squared)(pytree_length).units)
# {m: 1}
print(jax.grad(squared)(hijax_length).units)
# ((m, -1),)
```

Both gradients hold the same numbers. Only the second is measured in the units
the derivative is actually in, per metre rather than metres.

The units are in the jaxpr too, which is what a pytree can never show:

```python
mass = hijax_units.Unitful(jnp.asarray(2.0), pytree_units.kilograms)
velocity = hijax_units.Unitful(
    jnp.asarray(3.0), {meters: 1, pytree_units.seconds: -1}
)

print(jax.jit(quax.quaxify(lambda m, v: 0.5 * m * v**2)).trace(mass, velocity).jaxpr)
# { lambda ; a:u[]{kg} b:u[]{m s^-1}. let
#     c:u[]{kg} = call_hi_primitive[_prim=Mul[{}]] 0.5:f32[] a
#     d:u[]{m^2 s^-2} = call_hi_primitive[_prim=IntPow[{'y': 2}]] b
#     e:u[]{kg m^2 s^-2} = call_hi_primitive[_prim=Mul[{}]] c d
#   in (e,) }
```

Nothing in the traced function mentions units. They were derived, checked at
trace time, and printed in the type of every intermediate.

<!--- skip: end -->

## What it costs

Everything in the hijax layer is written by hand. A hijax type gets no operations
for free, so each one you want is a primitive with its own typing rule, its own
implementation, and its own rules for whichever transformations you intend to
apply. The example here needs seven primitives to cover multiplication, addition,
powers, sums and broadcasting, where the pure-Quax version needs a handful of
short functions and gets its autodiff structurally.

That cost scales with how many operations you need, not with how hard the type
is. A dozen operations is a day's work; the whole of `jnp` is not a realistic
target. It is the number to weigh when
[choosing between the three](#which-should-you-use).

Two limits are worth knowing before you commit. `jax.jacfwd`, `jax.hessian` and
`jax.jacrev` do not work on a hijax-typed argument at all, though `jax.grad` and
`jax.jvp` do. And hijax is experimental: it has renamed things in most recent
releases, which is why [`quax.experimental.hijax`](api/experimental.md) resolves
those names for you.

## Next

- [Transform metadata under autodiff](how-to/transform-metadata-under-autodiff.md)
  builds one.
- [`quax.examples.hijax`](api/hijax.md) is the worked type, with the full list of
  what it supports.
- [Why Quax is not built on hijax](about/why-not-built-on-hijax.md) is the design
  reasoning, if you want it.
