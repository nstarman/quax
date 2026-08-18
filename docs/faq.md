# FAQ

## Best practices with `jit` and `vmap`

### JIT

!!! Warning "Pair quaxify with jax.jit"

    Calling `quax.quaxify(fn)(*args)` **without** an outer `jax.jit` is **50–100×
    slower** than the JIT path for small operations. Every call pays for Python-level
    trace setup, jaxpr interpretation, and equinox module overhead — roughly 1–2 µs
    of fixed cost regardless of array size. So you should wrap the quaxified computations with `jax.jit` for best performance. This can be done at the outermost level:

    ```python
    import jax
    import quax

    # Directly
    jit_fn = jax.jit(quax.quaxify(jax.numpy.add))


    # Outer function
    @jax.jit
    def some_computation(x, y):
        # ... some code; could have other quax'ed computations ...
        return quax.quaxify(jax.numpy.add)(x, y)
    ```
    

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
