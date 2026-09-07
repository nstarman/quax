# Example libaries

These examples demonstrate how to introduce your own types, and write your own multiple dispatch programs, in JAX.

The `quax.examples.{lora, zero}` libraries are fairly feature-complete. The other example libraries are much smaller -- possibly buggier -- and serve primarily as demonstrations. 

(Although if you'd be interested in growing them into fully-fledged libraries, then get in touch.)

`quax.examples.hijax` is a little different: rather than demonstrating quax alone, it shows quax combined with `jax.experimental.hijax`, so that a type's metadata can transform under autodiff. It needs a newer JAX than the rest.
