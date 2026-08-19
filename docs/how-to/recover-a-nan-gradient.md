# Recover a gradient JAX computes as NaN

Some functions have a finite derivative that the chain rule cannot reach, because
an intermediate step is singular and the singularity cancels only afterwards:

```python
import jax
import jax.numpy as jnp


def g(a, b):
    return jnp.exp(jnp.log(a - b))


print(g(1.0, 1.0))  # 0.0
print(jax.grad(g)(1.0, 1.0))  # nan
```

`g` is `a - b`, so the derivative is `1.0`. But autodiff does not simplify. It
differentiates `log` at zero, gets `inf`, multiplies by `exp(-inf) = 0`, and
returns `nan`.

## Defer the singular step

Give the singular operation a `Value` that stands for its result without
computing it, and register the cancelling operation to unwrap it. Then the
`inf` never forms:

```python
from typing import Any

import equinox as eqx
import quax


class Tracked(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def aval(self):
        return jax.typeof(self.array)

    def materialise(self):
        return self.array


class Log(quax.ArrayValue):
    """Stands for `log(arg)`, left unevaluated."""

    arg: jax.Array = eqx.field(converter=jnp.asarray)

    def aval(self):
        return jax.typeof(self.arg)

    def materialise(self):
        return jnp.log(self.arg)


@quax.register(jax.lax.sub_p)
def sub_tracked(a: Tracked, b: Tracked, **kw: Any) -> Tracked:
    return Tracked(jax.lax.sub_p.bind(a.array, b.array, **kw))


@quax.register(jax.lax.log_p)
def log_tracked(a: Tracked, **kw: Any) -> Log:
    return Log(a.array)


@quax.register(jax.lax.exp_p)
def exp_log(a: Log, **kw: Any) -> Tracked:
    return Tracked(a.arg)  # exp(log(y)) is y, exactly


def quax_g(a, b):
    return quax.quaxify(g)(Tracked(a), Tracked(b)).array


print(jax.grad(quax_g)(1.0, 1.0))  # 1.0
```

Nothing here is an approximation or a clipped gradient. `exp(log(y))` *is* `y`,
and the rule says so, so the program being differentiated no longer contains a
singular step. Away from the singularity the answers are unchanged:

```python
print(quax_g(3.0, 1.0), jax.grad(quax_g)(3.0, 1.0))  # 2.0 1.0
```

## Cover the whole path

Every primitive between your inputs and the cancellation needs a rule. Miss one
and that operation falls back to `materialise`, your `Tracked` becomes a plain
array, and `log` is evaluated the ordinary way -- so you get the same `nan` back,
with nothing to say the rules did not fire.

Indexing is the usual way to trip on this: `theta[0]` is not one primitive but
`slice_p` followed by `squeeze_p`, and neither is `sub_p`. Register those too, or
keep the singular stretch free of indexing.

[Sharp bits](../sharp-bits.md) covers the general shape of this failure. If you
would rather it be loud, raise from `materialise` instead of returning the array
-- an unhandled primitive is then an error rather than a silent `nan`.

## When this applies

The pattern needs an *exact* inverse pair, one that cancels symbolically:
`exp`/`log` here, and equally `square`/`sqrt`, or a custom pair of your own. It
does not help where the limit is finite but the expression does not cancel; that
is what [`jax.custom_jvp`](../autodiff.md#custom-derivative-rules) is for.
