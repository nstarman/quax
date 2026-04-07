# Sparse arrays

JAX already has sparse arrays, under `jax.experimental.sparse`. Here we demonstrate another way that they might be constructed, using Quax.

In this simple example, we only create `BCOO`-formatted arrays (not `BCSR` or any other sparse format), and only implement rules for addition and multiplication.

Incidentally, JAX's `jax.experimental.sparse.sparsify` transformation was inspiration for Quax! `sparsify` is basically a special case of `quaxify` -- checking only for JAX's sparse array types, and not any Quax type.
