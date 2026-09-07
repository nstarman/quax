# Quax and hijax

JAX has grown its own extension API for custom types:
[hijax](https://docs.jax.dev/en/latest/301/hijax-types.html). You subclass
`HiType` to describe a type, register a value class against it, and write
`HiPrim` primitives that consume and produce it. The type then appears in
jaxprs as one value of one type, it may choose its own tangent and cotangent
types, and it can say how it batches and how it shards.

That overlaps enough with Quax to raise a fair question: now that JAX ships
this, what is Quax for?

The short answer is that the two solve opposite halves of the same problem, and
the half hijax deliberately leaves out is the whole of Quax.

## They point in opposite directions

Quax runs *existing programs* on a new type. You do not own `jnp.mean`, or the
solver inside Diffrax, or the model someone else wrote; Quax intercepts the
primitives those programs already emit and dispatches them on your type.

Hijax defines a *new type with its own operations*. Nothing existing applies to
it — that is the design, not an omission. A value of a hijax type is produced
and consumed only by the primitives you write for it, so `jnp.sin` on one is an
error rather than a silent loss of whatever the type was carrying.

Put those together and the picture is not a competition:

| | Quax | Hijax |
|---|---|---|
| Answers | "run this code on my type" | "give my type its own semantics" |
| The value is | an `equinox.Module` pytree | an opaque leaf, never a pytree |
| A jaxpr shows | the array leaves; your type is gone | one variable, typed |
| Existing `jnp`/`lax` ops | dispatch onto your rules | reject the type |
| A cotangent | reuses the primal's pytree, metadata included | has whatever type `to_ct_aval` says |
| Unhandled operation | falls back to `materialise` | is an error |
| Stability | stable | experimental, and renaming |

The last row is not a slight. Hijax is young: `HiType` and `register_hitype`
arrived in JAX 0.8.2, `HiPspec` in 0.9.2, `MappingSpec` became public in 0.11.0,
and the primitive base class `VJPHiPrimitive` is renamed `HiPrim` after 0.11.1.

## Which should you use?

Two questions settle it, and the first usually settles it alone.

**Who writes the code that touches your type?** If the answer includes anyone
but you — `jnp`, a solver from Diffrax, a model someone else wrote — then you
need Quax, because a hijax type is invisible to all of it. That is the common
case, and it is the reason Quax exists.

**Does the type itself have to do something Quax cannot?** Quax's blind spot is
that your type is metadata on a pytree, and JAX does not act on metadata. Four
capabilities follow from that, and Quax can reach none of them:

- a cotangent whose type differs from the primal's, so metadata transforms under
  differentiation rather than riding along unchanged;
- invariants checked while tracing and printed in the jaxpr, because the
  information is in the type rather than beside it;
- batching semantics of your own, instead of a per-leaf axis index;
- sharding carried in the type, and a `shard_map` boundary that understands it.

That gives three answers.

**Reach for Quax alone** unless something below overrides it. It is the smaller
thing, it works on every JAX from 0.7.2, your value stays a pytree so
`eqx.filter_*`, Optax and `jax.tree` keep working, and an operation you have not
written a rule for still has a `materialise` fallback rather than being an
error. Metadata that only needs to ride along — a name, a provenance tag, units
you check but never differentiate — belongs here.

**Reach for hijax alone** when you own every call site and the type is not
array-shaped anyway. A mutable cell, a log, a handle, a tuple-like container: JAX
ships `Box` and `Log` as exactly this. It is also the right tool when an
operation needs a lowering of its own — a fused or hand-written kernel that
`vmap`-ing the ordinary implementation would not produce. One convenience comes
free here that the hybrid does not get: a hijax tracer forwards attributes and
methods from its type, so `q.units` works inside traced code, where a Quax
tracer raises `AttributeError`.

**Reach for both** when existing code has to run *and* you need one of the four
capabilities. Put the hijax value in the leaf of a `quax.ArrayValue`: Quax stays
on the outside dispatching unmodified `jnp` code, hijax owns the inside where the
type lives. [`quax.experimental.hijax`](api/experimental.md) supplies the
plumbing for it. The cost is one hijax primitive per operation you support, so
this suits a small operation surface — a dozen ops, not all of `jnp`.

If you are unsure, start with Quax alone. Moving to the hybrid later changes how
your rules are implemented but not what your users write, because the
`ArrayValue` on the outside is the same either way.

## Hijax cannot be Quax's base class

The tempting design is to rebuild Quax on hijax — make `quax.Value` inherit from
something in `jax.experimental.hijax` and let JAX carry the weight. Three
separate things stop it, and each is worth knowing on its own.

**`HiType` is a type, not a value.** Its counterpart in Quax is not `Value` but
the object that [`quax.Value.aval`][] returns. So the question is not whether
`Value` can subclass `HiType`; it is whether `aval()` can return one.

**It cannot, because `jnp` would stop accepting your tracers.** `jax.Array`'s
metaclass answers `isinstance` by looking at the tracer's aval, and every `jnp`
function gates its arguments on that check. Give a Quax tracer a `HiType` aval
and `jnp.multiply(a, a)` refuses it, even though `jax.lax.mul(a, a)` — which
binds the primitive directly — still works. Since running unmodified `jnp` code
is the entire point of Quax, an `ArrayValue`'s aval has to stay a
`ShapedArray`. (Subclassing `ShapedArray` to smuggle metadata in does not work
either: it interns its instances, so a subclass comes back as the base class.)

**And a hijax value must not be a pytree.** Every JAX transform flattens a
pytree into its leaves *before* it consults the hijax type registry, so a value
that is a pytree can never carry a hijax type — the leaves get typed instead.
But `quax.Value` is an `equinox.Module`, and being a pytree is exactly what
makes it flow through `jit`, `grad`, `eqx.filter_*`, Optax, and everything else
in the ecosystem. The two representations are mutually exclusive for the same
object.

## They compose anyway

None of that stops the two working together, because a hijax primitive applied
inside a `quaxify` is simply expanded on the spot. JAX asks each trace whether
it needs primitives lowered before it sees them; Quax's does, so the primitive's
`expand` runs under the Quax trace and the ordinary `lax` operations inside it
dispatch on Quax's rules as usual. `jit`, `grad` and `vmap` all work in either
order.

The more interesting direction is the other one: a hijax value as the *leaf* of
a Quax value. Quax stays the outside — a pytree, dispatching unmodified `jnp`
code — and hijax owns the inside, where the type lives.
[`quax.examples.hijax`](api/hijax.md) does exactly this, and it buys the one
thing Quax has never been able to offer.

## What the combination buys

Quax cannot change metadata under differentiation. A cotangent must reuse the
primal's pytree structure, static fields included, so a length in metres
differentiates to a cotangent in metres — see
[Autodiff](autodiff.md#metadata-on-a-cotangent-is-the-primals). That is JAX's
contract, and no rule you write can escape it.

A hijax type is not metadata on a pytree; it is a *type*, and a type gets to say
what its cotangent type is. Compare the two units libraries in this repository,
which model the same physics and differ only in where the units live:

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
the derivative is actually in, per metre rather than metres, and it is the
hijax type's `to_ct_aval` that says so.

The units are in the jaxpr too, which is worth seeing because it is what a
pytree can never show:

```python
mass = hijax_units.Unitful(jnp.asarray(2.0), pytree_units.kilograms)
velocity = hijax_units.Unitful(
    jnp.asarray(3.0), {meters: 1, pytree_units.seconds: -1}
)

print(jax.jit(quax.quaxify(lambda m, v: 0.5 * m * v**2)).trace(mass, velocity).jaxpr)
# { lambda ; a:q[]{kg} b:q[]{m s^-1}. let
#     c:q[]{kg} = call_hi_primitive[_prim=Mul[{}]] 0.5:f32[] a
#     d:q[]{m^2 s^-2} = call_hi_primitive[_prim=IntPow[{'y': 2}]] b
#     e:q[]{kg m^2 s^-2} = call_hi_primitive[_prim=Mul[{}]] c d
#   in (e,) }
```

Nothing in the traced function mentions units. They were derived, checked at
trace time, and printed in the type of every intermediate.

## What it costs

Everything in the hijax layer is written by hand. A hijax type gets no
operations for free, so each one you want is a primitive with its own typing
rule, its own `expand`, and its own rules for whichever of `jvp`, `vjp`, `batch`
and transposition you intend to use. The example in this repository needs seven
primitives to cover multiplication, addition, powers, sums and broadcasting —
where the pure-Quax version needs a handful of short functions and gets its
autodiff structurally.

That cost scales with the operation surface, not with the difficulty of the
type, which is what makes it the deciding number when you are
[weighing the three options](#which-should-you-use). A dozen operations is a
day's work; the whole of `jnp` is not a realistic target.

## Where this leaves Quax

Hijax does not replace Quax and Quax should not be rebuilt on it. The question
that decides how the two eventually fit is upstream, not here: will JAX ever let
a hijax type say how *existing* primitives act on it?

If it does, that hook is essentially Quax's interpreter, and Quax becomes the
ergonomic layer above it — multiple dispatch, the `materialise` fallback, the
control-flow re-tracing. If it does not, Quax remains the only way to run
unmodified code on a custom type, and hijax is the tool for a type's own
semantics. Either way the leaf pattern above is the bridge, and Quax is in a
better position than before: cotangent metadata, typed jaxprs and
sharding-in-types are all things it could not do alone and can now reach.

To build one, see
[How to make metadata transform under autodiff](how-to/transform-metadata-under-autodiff.md).

<!--- skip: end -->
