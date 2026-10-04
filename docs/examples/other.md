# Symbolic zeros, structured matrices, ...

Quax also includes several other example libraries. Two are documented in their own right: [`quax.examples.lora`](../api/lora.md) and [`quax.examples.unitful`](../api/unitful.md).

The rest are listed below.

These are deliberately not documented further here, as we have no intention of turning these into officially-supported fully-fledged Quax libraries.

However if you want to write your own Quax library then they exist so that you can take a look at their source code -- as a useful demonstration, or as a starting point.

- `quax.examples.interval`: interval arithmetic — an array carrying a lower and an upper bound, so that uncertainty on the inputs comes out as a width on the output. Read its README first: the bounds are pessimistic where a value is used more than once, and are approximate rather than certified.
- `quax.examples.named`: arrays with named axes.
- `quax.examples.prng`: PRNGs as array-ish values. (Rather than the special-cased `jax.random.key` you normally use.)
- `quax.examples.sparse`: sparse arrays as array-ish values. (Rather than the `jax.experimental.sparse` implementation.)
- `quax.examples.structured_matrices`: a tridiagonal matrix with an efficient matmul implementation.
- `quax.examples.zero`: symbolic zeros, so that e.g. `a + zero` immediately returns `a` during tracing, or so that `zero[:5]` returns a zero of a different shape.
