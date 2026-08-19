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
