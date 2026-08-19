# Quaxify only some arguments

By default `quax.quaxify` takes every `Value` you pass and handles it. Sometimes
you want one to travel *through* untouched — most often so it reaches a nested
`quax.quaxify` that knows what to do with it, rather than being resolved by the
outer one.

Pass a `filter_spec`. It partitions `(fn, args, kwargs)`, so the spec has that
shape: one entry for the function, one tuple for the positional arguments, one
dict for the keyword arguments.

```python
import jax.numpy as jnp
import quax
from quax.examples.unitful import Unitful, meters, seconds


def inner(s):
    return s * 2.0


def outer(m, s):
    # `s` arrives here untouched, so the nested quaxify is what handles it
    return m * 3.0, quax.quaxify(inner)(s)


filter_spec = (False, (True, False), {})  # (fn, args, kwargs)

m_out, s_out = quax.quaxify(outer, filter_spec=filter_spec)(
    Unitful(jnp.ones(2), meters), Unitful(jnp.ones(2), seconds)
)
print(m_out.array, m_out.units)  # [3. 3.] {m: 1}
print(s_out.array, s_out.units)  # [2. 2.] {s: 1}
```

`False` at a position means "leave this alone"; `True` means "quaxify it".

Reach for this when two libraries' types would otherwise meet in one dispatch,
and you would be forced to
[write a rule for the combination](resolve-an-ambiguous-rule.md). Splitting the
quaxifies keeps each type's rules owned by the project that defines them.
