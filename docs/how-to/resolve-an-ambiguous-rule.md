# Resolve an ambiguous rule

You called a quaxified function and plum could not decide which of your rules
to use:

```
AmbiguousLookupError: `mul_dispatcher(Meters(array=f32[3]), Meters(array=f32[3]))`
is ambiguous.
```

Two rules match the call and neither is more specific than the other. The usual
shape is a pair written for mixed operands, meeting a call where *both* operands
are yours:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax
from jaxtyping import ArrayLike
from typing import Any


class Meters(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self):
        return self.array

    def aval(self):
        return jax.typeof(self.array)


@quax.register(jax.lax.mul_p)
def mul_meters_any(x: Meters, y: Meters | ArrayLike, **kw: Any) -> Meters:
    return Meters(jax.lax.mul_p.bind(x.array, getattr(y, "array", y), **kw))


@quax.register(jax.lax.mul_p)
def mul_any_meters(x: Meters | ArrayLike, y: Meters, **kw: Any) -> Meters:
    return Meters(jax.lax.mul_p.bind(getattr(x, "array", x), y.array, **kw))
```

`Meters * Meters` matches both, equally well.

## If you own both rules

Add the specific rule and give it a higher precedence. Dispatch then has a
single best answer rather than a tie:

```python
@quax.register(jax.lax.mul_p, precedence=1)
def mul_meters_meters(x: Meters, y: Meters, **kw: Any) -> Meters:
    return Meters(jax.lax.mul_p.bind(x.array, y.array, **kw))


out = quax.quaxify(lambda a, b: a * b)(Meters(jnp.full(3, 2.0)), Meters(jnp.full(3, 3.0)))
print(out.array)  # [6. 6. 6.]
```

Write the rule you actually want for that combination — the precedence only
breaks the tie, it does not choose behaviour for you.

## If the clash is between two libraries

Do not register a rule for the combination. You would be deciding, on behalf of
two projects you do not own, what their types mean together — and the answer
would apply process-wide to everyone who imports you.

Reach for [nested quaxifies](quaxify-part-of-a-function.md) instead: quaxify the
outer call for one type, and let the other reach its own `quax.quaxify` further
in.

## See also

The [advanced tutorial](../examples/redispatch.ipynb) works through why the
ambiguity arises, if you want the reasoning rather than the fix.
