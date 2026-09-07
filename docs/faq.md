# FAQ

## Should I write `grad(quaxify(f))` or `quaxify(grad(f))`?

Both give the same numbers. They differ in what type the gradient comes back as,
because whichever side of the `quaxify` boundary the differentiation happens on
is the side that decides:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax
from jaxtyping import ArrayLike
from typing import Any


class Tracked(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self):
        return self.array

    def aval(self):
        return jax.typeof(self.array)


@quax.register(jax.lax.mul_p)
def mul_tracked(x: Tracked, y: Tracked | ArrayLike, **kw: Any) -> Tracked:
    return Tracked(jax.lax.mul_p.bind(x.array, getattr(y, "array", y), **kw))


def square_sum(z):
    return jnp.sum(z * z)


t = Tracked(jnp.asarray([2.0, 3.0]))

print(type(jax.grad(quax.quaxify(square_sum))(t)).__name__)  # Tracked
print(type(quax.quaxify(jax.grad(square_sum))(t)).__name__)  # ArrayImpl
```

Put `quaxify` on the inside when you want the gradient to keep your type — the
cotangent then matches the primal you passed in. See
[Autodiff](autodiff.md) for what that gives you.

`jit` is not like this: both orderings return the same thing, and the
recommendation above is about compilation, not types.

## Why does `jnp.zeros_like(my_value)` give me a plain array?

Because nothing was dispatched on. The `*_like` functions read a shape and a
dtype off their argument and then allocate a fresh array; your value is never an
operand of an operation, so there is no rule for Quax to find:

```python
print(type(quax.quaxify(jnp.zeros_like)(t)).__name__)  # ArrayImpl
```

This is the same mechanism as a library allocating its own scratch buffers, and
[Sharp bits](sharp-bits.md) covers what follows from it. If you need a zero of
your own type, construct it directly rather than deriving it.

## Can I call a method on my `Value`?

Only across a `quaxify` boundary. A method body is ordinary code, so it needs a
trace installed before its operations can dispatch — outside one, the operators
have nothing to hook into:

```python
class WithMethod(Tracked):
    def doubled(self):
        return self * 2.0


try:
    WithMethod(jnp.ones(2)).doubled()
except TypeError as e:
    print(e)  # unsupported operand type(s) for *: 'WithMethod' and 'float'

quax.quaxify(WithMethod.doubled)(WithMethod(jnp.ones(2)))  # works
```

Quaxify the method as you would any other function. `quax.quaxify(Type.method)`
takes the instance as its first argument, so it behaves like the unbound
function it is.

## Why does `jaxtyping` reject my `Value` inside a quaxified function?

Because inside the `quaxify` your function is not handed the `Value` — it is
handed a tracer, and the tracer presents as a plain array:

```python
import jax
import jax.numpy as jnp
import quax
from quax.examples.unitful import meters, Unitful


def peek(x):
    print(type(x).__name__)  # _QuaxTracer
    print(jax.typeof(x))  # float32[1]
    return x


quax.quaxify(peek)(Unitful(jnp.asarray([1.0]), meters))
```

So a runtime typechecker sees `f32[1]` where you wrote `Unitful`, and rejects it.
Annotate the array, not your type:

```python
import beartype
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
def double(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return x * 2.0


out = quax.quaxify(double)(Unitful(jnp.asarray([1.0]), meters))
print(type(out).__name__, out.units)  # Unitful {m: 1}
```

The annotation describes what the function body works with; your type is what
crosses the `quaxify` boundary, and it still comes back out. Annotate with your
`Value` only on functions that take it *outside* a quaxify — a constructor, or a
rule registered with [`quax.register`][].

## I get an `UnexpectedTracerError`. Is this a Quax bug?

Almost certainly not, though it once was: `jnp.linalg.inv` and `jnp.linalg.solve`
did leak tracers, and no longer do. Today the usual cause is the ordinary JAX one
— a value escaped the transform and was used after it finished:

```python
stash = []


def leaky(x):
    stash.append(x)  # keeps the tracer past the trace
    return x * 2.0


quax.quaxify(leaky)(Unitful(jnp.asarray([1.0, 2.0]), meters))

try:
    quax.quaxify(lambda y: y * stash[0])(Unitful(jnp.asarray([1.0]), meters))
except Exception as e:
    print(type(e).__name__)  # UnexpectedTracerError
```

`quaxify` is a JAX transform, so JAX's rule applies unchanged: do not let a value
outlive the call it was traced in. Return it instead of stashing it.

## JAX has hijax now. Should I use that instead?

Usually not, and they are not really alternatives. Hijax defines a new JAX type
with its own primitives; nothing existing applies to one, so `jnp.sin` on it is
an error and every operation you want is a primitive you write. Quax runs code
you do not own on your type, which is the opposite problem and the one most
people have.

Reach for hijax when the *type* has to do something Quax cannot — chiefly a
cotangent that carries different metadata than its primal. The two also compose,
with a hijax value in the leaf of a `quax.ArrayValue`.
[Which should you use?](hijax.md#which-should-you-use) sets out the three cases,
and [`quax.examples.hijax`](api/hijax.md) is a worked implementation of the
combination.
