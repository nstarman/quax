# Symbolic zeros

This example library allows for the creation of symbolic zeros. These are equivalent to an array of zeros, like those created by `z = jnp.zeros(shape, dtype)`, except that we are able to resolve many operations, like `z * 5`, or `jnp.concatenate([z, z])`, at trace time -- as in these cases the result is again known to be a symbolic zero! -- and so we do not need to wait until runtime or hope that the compiler will figure it out.

As such, these are a more powerful version of the symbolic zeros JAX already uses inside its autodifferentiation rules (to skip computing gradients where none are required). However in this example library, we can use them anywhere in JAX.

## API

```python
zero.Zero
```

## Example

In this example, `quax.examples.zero` correctly identifies that (a) slicing an array of zeros again produces an array of zeros, and (b) that multiplying zero against nonzero still returns zero.

Thus the return value is again a symbolic zero.

```python
import jax.numpy as jnp
import quax
import quax.examples.zero as zero

z = zero.Zero((3, 4), jnp.float32)  # shape and dtype

def slice_and_multiply(a, b):
  return a[:, :2] * b

out = quax.quaxify(slice_and_multiply)(z, 3)
print(out)  # Zero(shape=(3, 2), dtype=dtype('float32'))
```
