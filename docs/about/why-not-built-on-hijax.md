# Why Quax is not built on hijax

*A design note. Nothing here is needed to use Quax — see
[Quax and hijax](../hijax.md) for that. This records why the two libraries are
arranged as they are, for anyone evaluating Quax's direction or proposing to
change it.*

When JAX grew [hijax](https://docs.jax.dev/en/latest/301/hijax-types.html), an
extension API for custom types, the obvious question was whether Quax should be
rebuilt on top of it, or had been made redundant by it. Neither, as it turns out,
and the reasons are specific enough to be worth writing down.

## The two libraries answer opposite questions

| | Quax | Hijax |
|---|---|---|
| Answers | "run this code on my type" | "give my type its own semantics" |
| The value is | an `equinox.Module` pytree | an opaque leaf, never a pytree |
| A jaxpr shows | the array leaves; your type is gone | one variable, typed |
| Existing `jnp`/`lax` ops | dispatch onto your rules | reject the type |
| A cotangent | reuses the primal's pytree, metadata included | has whatever type `to_ct_aval` says |
| An unhandled operation | falls back to `materialise` | is an error |
| Stability | stable | experimental, and renaming |

Hijax deliberately provides no way to say what an *existing* primitive means for
your type. That sentence is the whole of Quax, so the overlap is much smaller
than it first appears.

## Three reasons Quax cannot inherit from hijax

Each of these is independent, and each is pinned by a test in
`tests/unit/test_hijax_interop.py`.

**`HiType` is a type, not a value.** Its counterpart in Quax is not `Value` but
the object [`quax.Value.aval`][] returns. So the question is not whether `Value`
can subclass `HiType`; it is whether `aval()` can return one.

**It cannot, because `jnp` would stop accepting the tracers.** `jax.Array`'s
metaclass answers `isinstance` by looking at the tracer's aval, and every `jnp`
function gates its arguments on that check. Give a Quax tracer a `HiType` aval
and `jnp.multiply(a, a)` refuses it, even though `jax.lax.mul(a, a)` — which
binds the primitive directly — still works. Since running unmodified `jnp` code
is the entire point of Quax, an `ArrayValue`'s aval has to stay a `ShapedArray`.
Subclassing `ShapedArray` to smuggle metadata in does not work either: it interns
its instances, so a subclass comes back as the base class.

**And a hijax value must not be a pytree.** Every JAX transform flattens a pytree
into its leaves *before* it consults the hijax type registry, so a pytree value
can never carry a hijax type — the leaves get typed instead. But `quax.Value` is
an `equinox.Module`, and being a pytree is exactly what makes it flow through
`jit`, `grad`, `eqx.filter_*`, Optax and the rest of the ecosystem. The two
representations are mutually exclusive for the same object.

## Why they compose anyway

None of that stops the two working together. A hijax primitive applied inside a
`quaxify` is expanded on the spot: JAX asks each trace whether it needs
primitives lowered before it sees them, Quax's does, so the primitive's
implementation runs under the Quax trace and the ordinary `lax` operations inside
it dispatch on Quax's rules as usual. `jit`, `grad` and `vmap` work in either
order.

The useful direction is the other one — a hijax value as the *leaf* of a Quax
value — because it puts each library where it is strong. That is what
`quax.experimental.hijax` supports and what `quax.examples.hijax` demonstrates.

## Where this leaves Quax

Hijax does not replace Quax, and Quax should not be rebuilt on it. The question
that decides how the two eventually fit is upstream: will JAX ever let a hijax
type say how *existing* primitives act on it?

If it does, that hook is essentially Quax's interpreter, and Quax becomes the
ergonomic layer above it — multiple dispatch, the `materialise` fallback, the
control-flow re-tracing. If it does not, Quax remains the only way to run
unmodified code on a custom type, and hijax is the tool for a type's own
semantics.

Either way the leaf arrangement is the bridge, and Quax is better placed than
before: cotangent metadata, typed jaxprs and sharding-in-types are all things it
could not reach alone.
