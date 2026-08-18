# Control flow

`jax.jit`, `lax.cond`, `lax.while_loop`, `lax.fori_loop` and `lax.scan` all
work under `quax.quaxify`, and your type survives them:

```python
import jax
import jax.numpy as jnp
import quax
from jax import lax
from quax.examples.unitful import meters, Unitful

length = Unitful(jnp.asarray([2.0, 3.0]), meters)

square = quax.quaxify(jax.jit(lambda x: x * x))
print(square(length).units)  # {m: 2}

branch = quax.quaxify(lambda x: lax.cond(True, lambda: x * x, lambda: x * x))
print(branch(length).units)  # {m: 2}
```

## How it works

These primitives take a *jaxpr* rather than a callable — the body has already
been traced by the time Quax sees it. So Quax cannot simply dispatch on it the
way it dispatches on `mul_p`.

Instead it re-traces: it runs the body through `quaxify` again, producing a new
jaxpr that operates on your type's leaves, and rebinds the primitive with that.
`lax.cond` gets this treatment per branch, `lax.scan` once for the body.

Two consequences fall out of re-tracing, and both are visible from the outside.

A `Value` that flattens to several arrays changes the number of operands the
primitive takes, so Quax rebuilds the primitive's parameters to match. This is
why a type with two array fields works in a `scan` carry at all.

And the body is traced **once**, not once per iteration. That is where the
sharp edge is.

## The loop-carry trap

A loop's carry must be stable — JAX enforces this, and rejects a body whose
output type differs from its input. But a `Value`'s Python-level metadata is
*static*: it does not appear in the JAX type at all. So a body that changes it
looks perfectly stable to JAX, and the change is silently lost:

```python
def square_thrice(x):
    return lax.fori_loop(0, 3, lambda i, c: c * c, x)

out = quax.quaxify(square_thrice)(Unitful(jnp.asarray([2.0]), meters))
print(out.array)  # [256.] -- correct, 2**8
print(out.units)  # {m: 2}  -- wrong, should be {m: 8}
```

The number is right. The units are not: squaring three times should give
`{m: 8}`, and the result claims `{m: 2}` — what one pass through the body
produced. `lax.while_loop` and `lax.scan` mislabel the same way, differing only
in which pass they happen to keep.

`lax.cond` does not have this problem: its branches are traced separately and
compared, so a disagreement is caught rather than absorbed.

This is a real trap rather than a limitation to work around, so it is written
up in full — with what to do about it — in
[Sharp bits](sharp-bits.md#a-loop-carry-silently-keeps-the-metadata-it-started-with).
