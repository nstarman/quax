# FAQ

## Best practices with `jit` and `vmap`

### JIT

When using `jax.jit` together with `quax.quaxify`, apply `jax.jit` at the **outermost** level:

```python
import jax
import quax

# Do this:
jit_fn = jax.jit(quax.quaxify(fn))

# Not this:
jit_fn = quax.quaxify(jax.jit(fn))
```

This is the same best practice that applies to other JAX transforms: placing `jit` at the top level allows JAX to compile the entire computation, including the dispatch logic introduced by `quaxify`, as a single unit.

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
