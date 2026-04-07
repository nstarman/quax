# Pseudorandom number generators

JAX has a built in `jax.random.key` for creating PRNG keys. Here we demonstrate an alternative approach to these, by creating such array-ish objects in Quax.

## Example

```python
import jax.lax as lax
import jax.numpy as jnp
import quax
import quax.examples.prng as prng

# `key` is a PyTree wrapping a u32[2] array.
key = prng.ThreeFry(0)
prng.normal(key)

# Some primitives (lax.add_p) are disallowed.
def f(x, y):
    return x + y
quax.quaxify(f)(key, 1)  # TypeError!

# Some primitives (lax.select_n) are allowed.
# We're calling `jnp.where(..., pytree1, pytree2)` -- on pytrees, not arrays!
pred = jnp.array(True)
key2 = prng.ThreeFry(1)

@quax.quaxify
def run(pred, key1, key2):
    return jnp.where(pred, key1, key2)

run(pred, key, key2)
```

## API

```python
prng.PRNG      # Any custom PRNG type.
prng.ThreeFry  # Specifically a ThreeFry PRNG.
prng.uniform   # Sample from a uniform distribution.
prng.normal    # Sample from a normal distribution.
prng.split     # Split a PRNG key into multiple independent keys.
```

In addition, `jnp.where(..., key1, key2)` is supported.
