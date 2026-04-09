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
