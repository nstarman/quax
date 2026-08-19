# FAQ

## Best practices with `jit` and `vmap`

### JIT

!!! Warning "Pair quaxify with jax.jit"

    Calling `quax.quaxify(fn)(*args)` **without** an outer `jax.jit` is **50–100×
    slower** than the JIT path for small operations. Every call pays for Python-level
    trace setup, jaxpr interpretation, and equinox module overhead — roughly 1–2 µs
    of fixed cost regardless of array size. So you should wrap the quaxified computations with `jax.jit` for best performance. This can be done at the outermost level:

    ```python
    import jax
    import quax

    # Directly
    jit_fn = jax.jit(quax.quaxify(jax.numpy.add))


    # Outer function
    @jax.jit
    def some_computation(x, y):
        # ... some code; could have other quax'ed computations ...
        return quax.quaxify(jax.numpy.add)(x, y)
    ```
    

When using `jax.jit` together with `quax.quaxify`, the recommended pattern is to apply `jax.jit` at the **outermost** level:

```python
import jax
import quax

fn = jax.numpy.add

# Preferred: jit wraps the entire quaxified computation.
jit_fn = jax.jit(quax.quaxify(fn))

# Also works, but less optimal: quaxify wraps an already-jitted function.
jit_fn = quax.quaxify(jax.jit(fn))
```

Placing `jit` at the outermost level follows the same best practice that applies to other JAX transforms: it allows JAX to compile the entire computation, including the dispatch logic introduced by `quaxify`, as a single unit. The reversed ordering (`quax.quaxify(jax.jit(fn))`) does work correctly, but may compile less efficiently because each inner jitted call is compiled in isolation before `quaxify` sees it.

### vmap

When using `jax.vmap` together with `quax.quaxify`, the ordering does not affect correctness — place the transforms in whichever order best describes the operations you want to perform:

```python
import jax
import quax

# Both orderings are valid; choose whichever matches your intent.

# Batch over inputs, then quaxify:
vmap_fn = quax.quaxify(jax.vmap(fn))

# Or equivalently:
vmap_fn = jax.vmap(quax.quaxify(fn))
```

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
