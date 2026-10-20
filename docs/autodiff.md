# Autodiff

Quax is a JAX transform, so it composes with the others. `jax.grad`,
`jax.jvp`, `jax.vjp`, `jax.jit` and `jax.vmap` all work through a
`quax.quaxify`, in either order, and nest.

The part worth knowing is what comes *back*.

## Cotangents keep your type

Differentiate a quaxified function and the cotangent is an instance of your type,
not a plain array:

```python
import jax
import jax.numpy as jnp
import quax
from quax.examples.unitful import kilograms, meters, seconds, Unitful


def kinetic_energy(m, v):
    return 0.5 * m * v**2


mass = Unitful(jnp.asarray(2.0), kilograms)
velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

print(quax.quaxify(kinetic_energy)(mass, velocity).units)  # {kg: 1, m: 2, s: -2}

grad = jax.grad(lambda v: quax.quaxify(kinetic_energy)(mass, v).array)(velocity)
print(type(grad).__name__)  # Unitful
print(grad.units)  # {m: 1, s: -1}
```

The forward units are derived: nothing in `kinetic_energy` mentions units, and
joules came out. The gradient is a `Unitful` too — so rules keep firing on the
backward pass, and a `Value` that refuses to materialise still refuses there.

But read those gradient units again. They are the *velocity's*, not the
`{kg: 1, m: 1, s: -1}` that `d(½mv²)/dv` is measured in.

## Metadata on a cotangent is the primal's

This is worth stating plainly, because the first example anyone writes hides it.
Differentiate `x²` where `x` is in metres and the gradient is in metres — which
looks like the derivative's units, but is only the primal's units coinciding with
them.

A cotangent must have the same pytree structure as the primal it belongs to; that
is JAX's contract, not a Quax choice. For an `equinox.Module` the static fields
*are* part of that structure, so your metadata rides along unchanged. Quax
enforces this on the way out of a bwd rule: it takes the leaves of whatever the
rule returned and rebuilds them against the primal's treedef, so even
`jax.custom_vjp` cannot hand back a cotangent carrying different metadata.

What you get is that your type survives differentiation intact. What you do not
get is metadata recomputed for the derivative — Quax has no way to know what your
metadata means, or how the chain rule ought to act on it. If that transformation
matters to you, either track it outside the gradient, or move it out of the
pytree and into the JAX type, where a cotangent type can differ from its primal:
see [Transform metadata under autodiff](how-to/transform-metadata-under-autodiff.md).

## Custom derivative rules

A function carrying its own derivative rule via `jax.custom_jvp` or
`jax.custom_vjp` works under `quaxify`, forwards and backwards:

```python
length = Unitful(jnp.asarray([2.0, 3.0]), meters)


@jax.custom_jvp
def square(x):
    return x * x

@square.defjvp
def square_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return x * x, 2 * x * dx

print(quax.quaxify(square)(length).units)  # {m: 2}

grad = jax.grad(lambda x: quax.quaxify(square)(x).array.sum())(length)
print(grad.units)  # {m: 1}
```

`jax.custom_vjp` behaves the same way. This matters more than it first looks:
custom derivative rules are how libraries express derivatives they know better
than autodiff does, so supporting them is what lets Quax reach into those
libraries at all. `equinox.internal.while_loop` is built on `custom_vjp`, and
so, transitively, is every `diffrax` solve.

Two constraints follow from the rules being *yours*:

- The fwd and bwd rules run under the Quax trace, so they dispatch on your
  type like any other code. A rule that only handles plain arrays will
  materialise.
- Structural decisions a fwd rule stashes for its bwd rule — a Python `bool`
  used to pick a branch, say — stay Python values rather than becoming traced
  arrays, so `if` on them still works.

## What is not supported

Forward-over-reverse and reverse-over-reverse compose as JAX does; nothing in
Quax special-cases them. Where Quax stops being transparent is at a library's
own scratch buffers, and inside loops whose carry changes shape or metadata —
both covered in [Sharp bits](sharp-bits.md).
