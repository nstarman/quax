# 🔪 Quax - The Sharp Bits 🔪

Quax runs on existing JAX programs unchanged. These are the places where that
stops being true, what they look like when you hit them, and what to do.

## 🔪 A `Value` went in and a plain array came out

Quax dispatches on **operations**. It sees `lax.mul_p(x, y)` and can pick a rule
for your type. It cannot see an array a library allocates from nothing:

```python
buffer = jnp.zeros((n, *shape))   # no operand for Quax to dispatch on
```

so `buffer` is a plain array. A `Value` written into it comes back out plain,
and the type is gone from that point on.

This is why `diffrax.diffeqsolve` returns plain arrays under `quaxify`: it
allocates its `SaveAt` output buffer with `jnp.full` before the solve begins.
The numbers are right, the type is not preserved.

There is no general fix, because there is no general way to reconstruct your
type from an array — Quax cannot invent the units, the sparsity pattern, or the
LoRA factors. If you control the library, allocate from an operation on the
value (`jnp.broadcast_to(y0, ...)`) rather than from its shape.

## 🔪 A branch mismatch materialises instead of failing

The buffer above often surfaces at a `lax.cond`: one branch returns your
`Value`, another returns a slice of the plain buffer, and the two disagree on
structure. `cond_p` needs one output structure, so Quax materialises the branch
outputs — the same fallback that applies to any primitive with no matching rule.

You get a plain array rather than an error. If you would rather know, give your
type a `materialise` that raises.

## 🔪 A type that refuses to materialise stops there

`quax.examples.unitful` does exactly that:

```
ValueError: Refusing to materialise Unitful array.
```

This is the feature, not a bug. A unit-carrying array has no meaningful
dimensionless form, so silently dropping to a plain array would hand you a
number whose units you can no longer check. The error names the boundary where
the information was lost.

If you want a type that survives these boundaries, implement `materialise` to
return the dense array. If you want the boundary flagged, raise.

## 🔪 A gradient carries the primal's metadata, not the derivative's

Differentiate a `Unitful` in metres and the cotangent is in metres. That looks
right for `x²` and is wrong for almost everything else: the units you get are
whatever the *input* had, because a cotangent must match the structure of its
primal, and metadata is part of that structure.

Quax cannot fix this for you, and neither can a `jax.custom_vjp` rule —
[Autodiff](autodiff.md#metadata-on-a-cotangent-is-the-primals) has the details.
If your metadata should transform under differentiation, track it yourself.

## 🔪 A loop carry cannot change its metadata

A `Value`'s Python-level metadata — units, a sparsity pattern, anything held in
an `eqx.field(static=True)` — is invisible to JAX's type system. JAX enforces
that a loop carry is type-stable, but it cannot see static metadata, so a body
that *changes* it slips past that check:

```python
def square_thrice(x):
    return lax.fori_loop(0, 3, lambda i, c: c * c, x)

quax.quaxify(square_thrice)(Unitful(jnp.asarray([2.0]), meters))
# TypeError: `lax.scan` carry changed structure ...
```

Squaring three times should give `{m: 8}`. Quax traces the body until its carry
structure settles — a wrapper change reaches a fixed point after one more pass —
but metadata like this never does: `{m: 1}`, `{m: 2}`, `{m: 4}`, on and on.
Rather than hand back an array wearing one iteration's units, Quax refuses.

`lax.while_loop` and `lax.scan` both check this; `lax.fori_loop` reports as
whichever it lowered to. `lax.cond` never needed it — its branches are traced
separately and compared already.

The check does not fire when the body *materialises* its carry away. That is
the ordinary fallback for a value with no matching rule: the result is honestly
a plain array rather than a `Value` wearing the wrong metadata, so there is
nothing to warn about.

Keep loop carries metadata-stable. If the metadata must change, do that outside
the loop, or hoist it into the value itself where JAX can see it — an exponent
stored as a traced array rather than a static field is checked like any other
carry.

## 🔪 Implementing `aval()` correctly

`aval()` must be a **pure method**: called on the same instance it must always return the same `jax.core.AbstractValue`. Quax caches the result at tracer-construction time for performance — if `aval()` could return different values over time, the cached result would become stale.

In practice, any **Python/static metadata** that `aval()` uses to determine shape or dtype (for example cached shapes, symbolic dimensions, or units) should be declared with `eqx.field(static=True)`. Array payload fields do not need `static=True` just because `aval()` reads their shape or dtype:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import quax


class MyArray(quax.ArrayValue):
    array: jax.Array
    # shape and dtype are derived from `array`, which is static in the sense
    # that JAX arrays are immutable — no eqx.field annotation needed here.

    def aval(self):
        return jax.core.ShapedArray(self.array.shape, self.array.dtype)

    def materialise(self):
        return self.array
```

If you store the shape separately (e.g. to support lazy or symbolic shapes), mark it static:

```python
class MyArray(quax.ArrayValue):
    data: jax.Array
    _shape: tuple[int, ...] = eqx.field(static=True)  # REQUIRED

    def aval(self):
        return jax.core.ShapedArray(self._shape, self.data.dtype)

    def materialise(self):
        return self.data
```

Forgetting `static=True` on a shape field causes JAX to attempt tracing through the integer — which raises an error long before any caching issue arises.
