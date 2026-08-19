# Use a Quax type

The other tutorials teach you to *write* an array-ish type. This one is for using
one that already exists.

You will take a function that knows nothing about physical units and run it on
arrays that carry them — without editing the function. By the end you will have
caught a units bug inside code you did not write, and differentiated through it.

## 1. A function that knows nothing about you

```python
import jax.numpy as jnp


def kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity**2
```

Ordinary JAX. It multiplies and it squares; it has no opinion about units.

## 2. Give it values that do

`quax.examples.unitful` ships an array-ish type that carries units. Build two
values and wrap the function in `quax.quaxify`:

```python
import quax
from quax.examples.unitful import kilograms, meters, seconds, Unitful

mass = Unitful(jnp.asarray(2.0), kilograms)
velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

energy = quax.quaxify(kinetic_energy)(mass, velocity)
print(energy.array, energy.units)  # 9.0 {kg: 1, m: 2, s: -2}
```

Joules, derived rather than declared. `quaxify` reinterpreted each multiply and
each power according to what you passed in, the way `jax.vmap` reinterprets them
as their batched versions.

## 3. Get a unit error out of code that cannot raise one

Now the part a plain array cannot do for you. Add an energy to a velocity:

```python
try:
    quax.quaxify(lambda e, v: e + v)(energy, velocity)
except ValueError as e:
    print(e)  # Cannot add two arrays with units {kg: 1, m: 2, s: -2} and {m: 1, s: -1}.
```

Nothing in the lambda checks anything. The rule registered for `add` on `Unitful`
raised, at trace time, before a single number was added.

## 4. Compose with the rest of JAX

`quaxify` is a JAX transform, so the others still work — in either order, and
nested:

```python
import jax

fast = jax.jit(quax.quaxify(kinetic_energy))
print(fast(mass, velocity).units)  # {kg: 1, m: 2, s: -2}

grad = jax.grad(lambda v: quax.quaxify(kinetic_energy)(mass, v).array)(velocity)
print(type(grad).__name__, grad.units)  # Unitful {m: 1, s: -1}
```

The gradient came back as a `Unitful` rather than a bare array, so your type
survives the backward pass. Its units are the velocity's — a cotangent mirrors
the primal it belongs to, which is not the same thing as carrying the
derivative's units. [Autodiff](../autodiff.md) explains what that does and does
not buy you.

## What you did

You ran unmodified JAX code on a custom type, had that type reject an operation
that made no physical sense, and differentiated the result. That is the whole
user-facing surface: build values, wrap the function, read the result.

Where to go next:

- [Sharp bits](../sharp-bits.md) — where a type stops surviving, and why.
- [How-to guides](../how-to/resolve-an-ambiguous-rule.md) — for when you hit a
  specific problem.
- [Custom rules](../examples/custom_rules.ipynb) — if you now want to write a
  type of your own.
