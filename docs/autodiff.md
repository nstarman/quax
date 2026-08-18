# Autodiff

Quax is a JAX transform, so it composes with the others. `jax.grad`,
`jax.jvp`, `jax.vjp`, `jax.jit` and `jax.vmap` all work through a
`quax.quaxify`, in either order, and nest.

The part worth knowing is what comes *back*.

## Cotangents keep your type

Differentiate a quaxified function and the cotangent is an instance of your
type, not a plain array. For a type that carries units, that means the
gradient carries the derivative's units:

```python
import jax
import jax.numpy as jnp
import quax
from quax.examples.unitful import meters, Unitful

def area(x):
    return x * x

length = Unitful(jnp.asarray([2.0, 3.0]), meters)

out = quax.quaxify(area)(length)
print(out.units)  # {m: 2}

grad = jax.grad(lambda x: quax.quaxify(area)(x).array.sum())(length)
print(grad.units)  # {m: 1}
```

`d(m²)/dm` is in metres, and that is what arrives. Nothing in `area` mentions
units; the rules registered on `Unitful` did the work in both directions.

This falls out of how JAX cotangents work — they match the structure of the
primal input — rather than being something Quax arranges specially. It does
mean a rule author gets it for free, and that a `Value` with an invariant to
enforce keeps enforcing it on the backward pass.

## Custom derivative rules

A function carrying its own derivative rule via `jax.custom_jvp` or
`jax.custom_vjp` works under `quaxify`, forwards and backwards:

```python
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
