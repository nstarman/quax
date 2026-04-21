# FAQ

## Best practices with `jit` and `vmap`

### JIT

When using `jax.jit` together with `quax.quaxify`, the recommended pattern is to apply `jax.jit` at the **outermost** level:

```python
import jax
import quax

# Preferred: jit wraps the entire quaxified computation.
jit_fn = jax.jit(quax.quaxify(fn))

# Also works, but less optimal: quaxify wraps an already-jitted function.
jit_fn = quax.quaxify(jax.jit(fn))
```

Placing `jit` at the outermost level follows the same best practice that applies to other JAX transforms: it allows JAX to compile the entire computation, including the dispatch logic introduced by `quaxify`, as a single unit. The reversed ordering (`quax.quaxify(jax.jit(fn))`) does work correctly, but may compile less efficiently because each inner jitted call is compiled in isolation before `quaxify` sees it.

### vmap

When using `jax.vmap` together with `quax.quaxify`, the ordering does not affect correctness — place the transforms in whichever order best describes the operations you want to perform:

```python
import jax
import quax

# Both orderings are valid; choose whichever matches your intent.

# Batch over inputs, then quaxify:
vmap_fn = quax.quaxify(jax.vmap(fn))

# Or equivalently:
vmap_fn = jax.vmap(quax.quaxify(fn))
```

---

## Implementing `aval()` correctly

`aval()` must be a **pure method**: called on the same instance it must always return the same `jax.core.AbstractValue`. Quax caches the result at tracer-construction time for performance — if `aval()` could return different values over time, the cached result would become stale.

In practice this is guaranteed automatically when every field that affects the shape or dtype is declared with `eqx.field(static=True)`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax

class MyArray(quax.ArrayValue):
    array: jax.Array
    # shape and dtype are derived from `array`, which is static in the sense
    # that JAX arrays are immutable — no eqx.field annotation needed here.

    def aval(self):
        return jax.core.ShapedArray(self.array.shape, self.array.dtype)

    def materialise(self):
        return self.array
```

If you store the shape separately (e.g. to support lazy or symbolic shapes), mark it static:

```python
class MyArray(quax.ArrayValue):
    data: jax.Array
    _shape: tuple[int, ...] = eqx.field(static=True)   # REQUIRED

    def aval(self):
        return jax.core.ShapedArray(self._shape, self.data.dtype)

    def materialise(self):
        return self.data
```

Forgetting `static=True` on a shape field causes JAX to attempt tracing through the integer — which raises an error long before any caching issue arises.
