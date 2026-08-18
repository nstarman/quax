<h1 align="center">Quax</h1>
<h2 align="center">JAX + multiple dispatch + custom array-ish objects</h2>

<p align="center">
    <a href="https://pypi.org/project/quax/"><img alt="PyPI: quax" src="https://img.shields.io/pypi/v/quax?style=flat" /></a>
    <a href="https://pypi.org/project/quax/"><img alt="PyPI versions: quax" src="https://img.shields.io/pypi/pyversions/quax" /></a>
    <a href="https://nstarman.github.io/quax/"><img alt="Documentation" src="https://img.shields.io/badge/read_docs-here-orange" /></a>
    <a href="https://github.com/nstarman/quax/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/nstarman/quax" /></a>
</p>
<p align="center">
    <a href="https://docs.astral.sh/ruff/"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
    <a href="https://github.com/j178/prek"><img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /></a>
    <a href="https://app.codspeed.io/nstarman/quax"><img alt="CodSpeed" src="https://img.shields.io/endpoint?url=https://codspeed.io/badge.json" /></a>
</p>
<p align="center">
    <a href="https://github.com/nstarman/quax/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/nstarman/quax/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
    <a href="https://nstarman.github.io/quax/"><img alt="Docs status" src="https://github.com/nstarman/quax/actions/workflows/build_docs.yml/badge.svg?branch=main" /></a>
</p>

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

## What that looks like

A few examples of `quax` propagating physical units through JAX code.

**Your code doesn't change.** `kinetic_energy` is ordinary JAX with no notion
of units. Quax tracks them through it and works out the result's units:

```python
import jax.numpy as jnp
import quax
from quax.examples.unitful import kilograms, meters, seconds, Unitful

def kinetic_energy(m, v):
    return 0.5 * m * v**2

mass = Unitful(jnp.asarray(2.0), kilograms)
velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

energy = quax.quaxify(kinetic_energy)(mass, velocity)
print(energy.array, energy.units)  # 9.0 {kg: 1, m: 2, s: -2}
```

**Units are checked, not just carried along.** A `Unitful` knows that adding
metres to seconds is meaningless, and raises instead of silently returning a
wrong number:

```python
try:
    quax.quaxify(jnp.add)(
        Unitful(jnp.asarray(1.0), meters), Unitful(jnp.asarray(1.0), seconds)
    )
except ValueError as e:
    print(e)  # Cannot add two arrays with units {m: 1} and {s: 1}.
```

**It also works on code you didn't write.**
[Diffrax](https://github.com/patrick-kidger/diffrax) was written years before
Quax existed and knows nothing about units, but its solver carries them
through anyway — unmodified:

<!--- skip: next if(not have_diffrax, 'diffrax not installed') -->
```python
import diffrax
import jax.numpy as jnp
import quax
from quax.examples.unitful import meters, Unitful

term = diffrax.ODETerm(lambda t, y, args: -0.5 * y)
solver = diffrax.Euler()

def step(y0):
    state = solver.init(term, 0.0, 0.1, y0, None)
    y1, _, _, _, _ = solver.step(term, 0.0, 0.1, y0, None, state, made_jump=False)
    return y1

y1 = quax.quaxify(step)(Unitful(jnp.asarray([1.0]), meters))
print(y1.array, y1.units)  # [0.95] {m: 1}
```

The one place this breaks down: if a library allocates its own buffers
internally, Quax never sees the allocation and you get plain arrays back.
[Sharp bits](https://nstarman.github.io/quax/sharp-bits/) covers that and the
other boundaries.

## Documentation

<https://nstarman.github.io/quax>

- [Custom rules](https://nstarman.github.io/quax/examples/custom_rules/) —
  build your own array-ish type, start to finish
- [API reference](https://nstarman.github.io/quax/api/quax/)
- [Sharp bits](https://nstarman.github.io/quax/sharp-bits/) — where Quax stops
  being transparent, and why
- [FAQ](https://nstarman.github.io/quax/faq/) — `jit`, `vmap`, and writing
  `aval()`

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

**Built on Quax**  
[Quaxed](https://github.com/GalacticDynamics/quaxed): a namespace of already-wrapped `quaxify(jnp.foo)` operations.  
[quax-blocks](https://github.com/GalacticDynamics/quax-blocks): blocks for constructing `quax` classes.  
[unxt](https://github.com/GalacticDynamics/unxt): unitful quantities.  
[coordinax](https://github.com/GalacticDynamics/coordinax): coordinates in JAX.  
[galax](https://github.com/GalacticDynamics/galax): galactic and gravitational dynamics, with GPU and autodiff.  
[phasecurvefit](https://github.com/GalacticDynamics/phasecurvefit): construct paths through phase-space points.

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

## Acknowledgements

Significantly inspired by https://github.com/davisyoshida/qax, https://github.com/stanford-crfm/levanter, and `jax.experimental.sparse`.
