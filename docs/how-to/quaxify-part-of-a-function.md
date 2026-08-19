# Quaxify only some arguments

By default `quax.quaxify` takes every `Value` you pass and handles it. Sometimes
you want one to travel *through* untouched — most often so it reaches a nested
`quax.quaxify` that knows what to do with it, rather than being resolved by the
outer one.

Pass a `filter_spec`. It partitions `(fn, args, kwargs)`, so the spec has that
shape: one entry for the function, one tuple for the positional arguments, one
dict for the keyword arguments.

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


class Seconds(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self):
        return self.array

    def aval(self):
        return jax.typeof(self.array)


@quax.register(jax.lax.mul_p)
def mul_meters_arraylike(x: Meters, y: ArrayLike, **kw: Any) -> Meters:
    return Meters(jax.lax.mul_p.bind(x.array, y, **kw))


@quax.register(jax.lax.mul_p)
def mul_seconds_arraylike(x: Seconds, y: ArrayLike, **kw: Any) -> Seconds:
    return Seconds(jax.lax.mul_p.bind(x.array, y, **kw))
```

Now quaxify the outer function for `Meters` only, and let `Seconds` reach the
inner one:

```python
def inner(s):
    return s * 2.0


def outer(m, s):
    # `s` arrives here untouched, so the nested quaxify is what handles it
    return m * 3.0, quax.quaxify(inner)(s)


filter_spec = (False, (True, False), {})  # (fn, args, kwargs)

m_out, s_out = quax.quaxify(outer, filter_spec=filter_spec)(
    Meters(jnp.ones(2)), Seconds(jnp.ones(2))
)
print(type(m_out).__name__, m_out.array)  # Meters [3. 3.]
print(type(s_out).__name__, s_out.array)  # Seconds [2. 2.]
```

`False` at a position means "leave this alone"; `True` means "quaxify it". The
function itself is usually `False` — set it `True` only if the callable carries
`Value`s you want handled, such as a model whose weights are your type.

## When to reach for this

Mostly when two libraries' types would otherwise meet in one dispatch and you
would be forced to
[write a rule for the combination](resolve-an-ambiguous-rule.md). Splitting the
quaxifies keeps each type's rules owned by the project that defines them.

If you only want a plain array left alone, you do not need `filter_spec` — pass
it as a plain array and it stays one.
