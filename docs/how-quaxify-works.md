# How quaxify works

`quax.quaxify` is a JAX transform in the same sense `jax.vmap` is. `vmap` takes a
program and reinterprets each operation as its batched version; `quaxify` takes a
program and reinterprets each operation according to the types you passed in.

That is the whole idea. The rest of this page is what it means in practice.

## Programs bottom out in primitives

Whatever you write — `x * 2`, `jnp.sum`, a `diffrax` solve — decomposes into a few
hundred JAX *primitives*: `mul_p`, `add_p`, `reduce_sum_p`, and so on. A transform
does not need to understand your program, only those.

So Quax intercepts primitives. `quaxify` installs a trace, boxes each `Value` you
passed in, and every time a primitive is bound with one of those boxes as an
operand, Quax is asked what to do instead.

## What it does instead

It performs multiple dispatch on the unboxed operand types, and resolves in this
order:

1. A rule [`quax.register`][]'d for exactly those types wins.
2. Otherwise, if exactly one operand's type overrides [`quax.Value.default`][],
   that runs.
3. Otherwise every operand is [`quax.Value.materialise`][]d and ordinary JAX runs.
4. If more than one type overrides `default`, that is an error.

Case 3 is the one to keep in mind: **an unregistered primitive is not an error.**
It is a silent fallback to plain arrays.

```python
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import quax
from jaxtyping import ArrayLike


class Tagged(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def aval(self):
        return jax.typeof(self.array)

    def materialise(self):
        print("materialised")
        return self.array


x = Tagged(jnp.asarray([1.0, 2.0]))

# No rule for add_p, so case 3: materialise and run plain JAX.
print(type(quax.quaxify(lambda v: v + 1.0)(x)).__name__)  # materialised
#                                                         # ArrayImpl


@quax.register(jax.lax.mul_p)
def mul_tagged_array_like(a: Tagged, b: ArrayLike, **kw: Any) -> Tagged:
    return Tagged(jax.lax.mul_p.bind(a.array, b, **kw))


# Now case 1: a rule matches, and the type survives.
print(type(quax.quaxify(lambda v: v * 2.0)(x)).__name__)  # Tagged
```

Registering rules is therefore incremental. You are never required to cover the
whole primitive set — you cover what you care about, and everything else keeps
working by falling back. The cost is that "keeps working" can mean "quietly stopped
being your type", which is what [Sharp bits](sharp-bits.md) is about.

## The three methods

A `Value` answers three questions, and they map onto the three things a transform
needs to know.

`aval()` is what JAX sees. Shape and dtype, nothing else — so your metadata is
invisible at the JAX level, which is why it is not type-checked and why a loop
carry can change it without JAX noticing.

`materialise()` is how to stop being yourself. It is the escape hatch case 3 uses.
A type that has nothing to lose can return its array; a type whose whole point is
the metadata should raise, and turn a silent fallback into a loud one.

`default()` is optional and static: a catch-all for primitives you did not
register, letting you say "for anything else, do this" rather than enumerating.

## Where the model stops being this simple

Primitives that take a *jaxpr* rather than operands — `jit`, `cond`, `scan`,
`while_loop` — cannot be dispatched on, because the body was traced before Quax
saw it. Quax re-traces them instead; [Control flow](control-flow.md) covers what
follows from that.

And a transform can only intercept operations that actually reach it. A library
allocating its own buffers, or calling `jnp.zeros_like` on your value, never binds
a primitive against you at all.

## A quaxify inside a quaxified function

Code inside a quaxified function can call another quaxified function, often
without knowing it: a `Value`'s `__mul__` may itself be built on `quaxify`. When
the inner call is given one of the outer function's values, it joins the trace
that is already running, so it sees that value with its real type:

```python
import jax.numpy as jnp
import quax
from quax.examples.unitful import Unitful, meters

mul = quax.quaxify(jnp.multiply)


def area(side):
    return mul(side, Unitful(jnp.array(2.0), meters))


out = quax.quaxify(area)(Unitful(jnp.array(3.0), meters))
print(out.array, out.units)  # 6.0 {m: 2}
```

The joined call returns the result as part of the outer computation, like any
other operation on the outer values.

A call that gets none of the outer values runs in a trace of its own and returns
a `Value` you can inspect, as it would outside:

```python
def f(side):
    local = Unitful(jnp.array(2.0), meters)
    print(mul(local, local).units)  # {m: 2}
    return side


quax.quaxify(f)(Unitful(jnp.array(3.0), meters))
```

Some calls start a trace of their own even when given outer values, and see them
as plain arrays:

- A `quaxify` with a custom `filter_spec`, and a `quaxify` called directly by
  one. These leave values for a nested `quaxify` to handle on its own; see
  [Quaxify only some arguments](how-to/quaxify-part-of-a-function.md). Any
  `filter_spec` other than `True` counts, even one that selects everything.
- A `quaxify` called by a rule.
- A `quaxify` with another transform between it and the outer one: inside
  `jax.jit`, `jax.vmap`, the body of `lax.cond`, `lax.scan` or `lax.while_loop`,
  or the body of a `jax.custom_jvp` or `jax.custom_vjp` function.
