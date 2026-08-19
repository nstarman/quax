# Combine `quaxify` with `jit` and `vmap`

## Put `jit` outermost

`quaxify` costs Python time on every call: it installs a trace, walks the program
primitive by primitive, and allocates as it goes. That cost is per-call and
roughly independent of array size, so it dominates small operations.

`jax.jit` on the outside pays it once, at compile time:

```python
import jax
import jax.numpy as jnp
import quax

# Preferred: jit sees the whole quaxified computation.
fast = jax.jit(quax.quaxify(jnp.add))

# Also correct, but the dispatch runs on every call.
slow = quax.quaxify(jax.jit(jnp.add))
```

Both give the same answer. The first compiles the dispatch away; the second
re-enters the Quax trace on every call. Adding two two-element arrays, the first
measured about 15× faster on the machine these docs were written on. Treat the
shape of that rather than the number: it is fixed overhead, so the gap widens as
arrays get smaller and closes as they get larger.

## `vmap` in either order

Here the ordering does not matter: `quaxify(vmap(f))` and `vmap(quaxify(f))` do
the same thing. Pick whichever describes what you mean.

`grad` is the exception — there the ordering changes what type comes back. See
[the FAQ](../faq.md#should-i-write-gradquaxifyf-or-quaxifygradf).
