<h1 align="center">Quax</h1>
<h2 align="center">JAX + multiple dispatch + custom array-ish objects</h2>

For example, this can be mean overloading matrix multiplication to exploit sparsity or structure, or automatically rewriting a LoRA's matmul `(W + AB)v` into the more-efficient `Wv + ABv`.

Applications include:

- LoRA weight matrices
- symbolic zeros
- arrays with named dimensions
- structured (e.g. tridiagonal) matrices
- sparse arrays
- quantised arrays
- arrays with physical units attached
- etc! (See the built-in `quax.examples` library for most of the above!)

This works via a custom JAX transform. Take an existing JAX program, wrap it in a `quax.quaxify`, and then pass in the custom array-ish objects. This means it will work even with existing programs, that were not written to accept such array-ish objects!

_(Just like how `jax.vmap` takes a program, but reinterprets each operation as its batched version, so to will `quax.quaxify` take a program and reinterpret each operation according to what array-ish types are passed.)_

## Installation

```
pip install quax
```

## Documentation

Available at https://docs.kidger.site/quax.

## Example: LoRA

This example demonstrates everything you need to use the built-in `quax.examples.lora` library.

```python
import equinox as eqx
import jax.random as jr
import quax
import quax.examples.lora as lora

#
# Start off with any JAX program: here, the forward pass through a linear layer.
#

key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
linear = eqx.nn.Linear(10, 12, key=key1)
vector = jr.normal(key2, (10,))

def run(model, x):
  return model(x)

run(linear, vector)  # can call this as normal

#
# Now let's Lora-ify it.
#

# Step 1: make the weight be a LoraArray.
lora_weight = lora.LoraArray(linear.weight, rank=2, key=key3)
lora_linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)
# Step 2: quaxify and call the original function. The transform will call the
# original function, whilst looking up any multiple dispatch rules registered.
# (In this case for doing matmuls against LoraArrays.)
quax.quaxify(run)(lora_linear, vector)
# Appendix: Quax includes a helper to automatically apply Step 1 to all
# `eqx.nn.Linear` layers in a model.
lora_linear = lora.loraify(linear, rank=2, key=key3)
```

## Work in progress!

Right now, the following are not supported:

- Control flow primitives (e.g. `jax.lax.cond`).
- `jax.custom_vjp`

It should be fairly straightforward to add support for these; open an issue or pull request.

## See also: other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Built on Quax**  
[Quaxed](https://github.com/GalacticDynamics/quaxed): a namespace of already-wrapped `quaxify(jnp.foo)` operations.  
[unxt](https://github.com/GalacticDynamics/unxt): Unitful Quantities.
[coordinax](https://github.com/GalacticDynamics/coordinax): Coordinate representations; e.g. 3D Spherical.

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

## Acknowledgements

Significantly inspired by https://github.com/davisyoshida/qax, https://github.com/stanford-crfm/levanter, and `jax.experimental.sparse`.
