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

And the body is traced for its *structure*, not once per iteration. Usually
that is a single pass. A body whose first pass changes the carry's wrapper —
a slot pre-allocated plain and first written to inside the loop, or one
materialised by a rule that does not match it — is traced again against that
new structure, until it settles. What settles is what every iteration after
the first actually sees, and it is how the result is labelled.

## The loop-carry constraint

A loop's carry must be stable — JAX enforces this, and rejects a body whose
output type differs from its input. But a `Value`'s Python-level metadata is
*static*: it does not appear in the JAX type at all, so a body that changes it
slips past that check.

A wrapper change settles: trace the body again against it and you reach a fixed
point. Metadata does not. Squaring takes `{m: 1}` to `{m: 2}` to `{m: 4}`, and
no number of passes reaches a structure the loop could keep. Quax refuses,
rather than returning an array labelled with one iteration's units:

```python
def square_thrice(x):
    return lax.fori_loop(0, 3, lambda i, c: c * c, x)

try:
    quax.quaxify(square_thrice)(Unitful(jnp.asarray([2.0]), meters))
except TypeError as e:
    print(str(e).splitlines()[0])  # `lax.scan` carry changed structure: the body was given
```

`lax.cond` does not need this: its branches are traced separately and compared,
so a disagreement is caught rather than absorbed.

Keeping the carry stable is the fix, and
[Sharp bits](sharp-bits.md#a-loop-carry-cannot-change-its-metadata) covers what
to do when the metadata genuinely has to change.
