---
name: quax
description: Use when writing, reviewing, or debugging JAX code that involves quax — custom array-ish objects (physical units, LoRA, sparse, symbolic zero, named axes), quax.quaxify, quax.register dispatch rules, or Value/ArrayValue subclasses. Also use when a custom array type is unexpectedly materialised into a plain array, a primitive raises a plum ambiguity or "multiple array-ish types" error, aval()/materialise() misbehave, or a quaxified program leaks tracers or runs far slower than expected.
---

# Using Quax Effectively

Quax runs a JAX program under a custom interpreter, reinterpreting each primitive
according to the types passed in. Just as `jax.vmap` takes a program and
reinterprets every operation as its batched version, `quax.quaxify` takes a
program and reinterprets every operation according to multiple-dispatch rules you
register. This means custom array-ish objects work with existing JAX code that
was never written to accept them.

Checked against quax v0.4.x (`jax` 0.7.2–0.11.x, `equinox>=0.13.3`,
`plum-dispatch>=2.5.7`, Python >=3.11). Docs: <https://nstarman.github.io/quax>.
Note that `patrick-kidger/quax` is at 0.2.x while PyPI `quax` now comes from
`nstarman/quax` — prior knowledge of this library is probably a version behind.
See "Version notes" at the end.

## Quick start

Take any existing JAX program and pass custom types through it:

```python
import equinox as eqx
import jax.random as jr
import quax
import quax.examples.lora as lora

key1, key2, key3 = jr.split(jr.key(0), 3)
linear = eqx.nn.Linear(10, 12, key=key1)
vector = jr.normal(key2, (10,))


def run(model, x):
    return model(x)


run(linear, vector)  # ordinary JAX, works as normal

# Swap the weight for a LoRA array, then quaxify the *unchanged* function.
lora_weight = lora.LoraArray(linear.weight, rank=2, key=key3)
lora_linear = eqx.tree_at(lambda l: l.weight, linear, lora_weight)
out = quax.quaxify(run)(lora_linear, vector)
```

`quax.examples` ships `lora`, `zero`, `unitful`, `named`, `sparse`, `prng`, and
`structured_matrices` — read their `_core.py` files as reference implementations.

## Pair quaxify with jax.jit

Bare `quax.quaxify(fn)(*args)` is **50–100x slower** than the JIT path for small
operations. Every call pays Python-level trace setup, jaxpr interpretation, and
equinox module overhead — roughly 1–2 µs of fixed cost regardless of array size.
This is the single most expensive mistake available here, and it is invisible:
the code is correct, just slow.

```python
import jax
import jax.numpy as jnp

jit_fn = jax.jit(quax.quaxify(jnp.add))  # preferred: jit outermost
```

Put `jax.jit` at the **outermost** level. `quax.quaxify(jax.jit(fn))` is correct
but compiles less efficiently, because each inner jitted call is compiled in
isolation before quaxify sees it. With `vmap`, ordering does not matter for
correctness or speed — choose whichever expresses your intent.

Quaxify short-circuits entirely when no argument is a `quax.Value`, so
quaxifying a function that might receive plain arrays costs nothing extra.

## How dispatch resolves

For `primitive.bind(x, y, ...)` under a quaxify, in order:

1. A rule registered for the types of `(x, y, ...)` matches → use it.
2. Exactly one argument's type overrides `Value.default` → use that.
3. No type overrides `default` → **materialise every operand** and call ordinary
   JAX.
4. Two or more types override `default` → `TypeError: Multiple array-ish types
   {...} are specifying default process rules.`

Rule 3 is the one that surprises people. A missing rule is not an error by
default; it silently degrades to plain JAX, and your custom type vanishes from
the output. If that is never acceptable for your type, make `materialise()`
raise — then rule 3 becomes a loud failure instead of a silent one.

An operand that is neither a `quax.Value` nor array-like (a `str`, `None`) raises
`TypeError: Primitive ... got an operand of type ... that is neither a quax.Value
nor array-like`.

## Creating a custom ArrayValue

1. **Subclass `quax.ArrayValue`** (not `Value` — `ArrayValue` is for array-ish
   things and gives you `.shape`/`.dtype`/`.ndim`/`.size` for free). It is an
   `equinox.Module`, so it is a frozen dataclass and a pytree.
2. **Mark metadata `eqx.field(static=True)`** — units, axis names, stored shapes,
   bool flags. Anything JAX must not trace through.
3. **Implement `aval()`** returning a `jax.core.ShapedArray`. It must be pure.
4. **Implement `materialise()`** — or raise, to forbid the rule-3 fallback.
5. **Register rules** for the primitives you care about, starting with the ones
   your users will actually hit (`add_p`, `mul_p`, `dot_general_p`).
6. **Add mixed-type rules** so your type interoperates with plain arrays and with
   other people's quax types.
7. **Test both paths**: the registered rule, and what happens when no rule
   matches.

```python
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import ArrayLike


class Meters(quax.ArrayValue):
    array: ArrayLike

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(jnp.shape(self.array), jnp.result_type(self.array))

    def materialise(self):
        raise ValueError("Refusing to materialise Meters: it would drop the unit.")


@quax.register(lax.add_p)
def add_meters_meters(x: Meters, y: Meters) -> Meters:
    return Meters(x.array + y.array)


quax.quaxify(jnp.add)(Meters(jnp.arange(3.0)), Meters(jnp.ones(3)))
```

## aval() and materialise()

`aval()` **must be pure**: the same instance must return the same
`AbstractValue` every time. Quax caches the result at tracer-construction time
and never observes later changes.

Pure does not mean array-free — `aval()` may bind JAX primitives, and quax
evaluates it under the parent trace so that works. Prefer not to: `aval` runs
once per tracer, so a bind there is on the hot path. Read the field directly
when the primitive would not change the shape or dtype. (Under jax ≤0.10 this held by luck; jax 0.11 leaves
no current trace at that point. Fixed on main after v0.4.3 — on v0.4.3 or
earlier with jax 0.11, an `aval()` that binds anything fails.)

Which fields need `static=True` follows from that. Shape and dtype *derived from
an array field* need nothing special — JAX arrays are immutable, so reading
`self.array.shape` is already stable. A separately stored shape, or units, or a
flag consulted by `aval()`, must be `eqx.field(static=True)`; forget it and JAX
tries to trace through the integer, which errors long before staleness matters.

```python
import equinox as eqx


class Zeros(quax.ArrayValue):
    _shape: tuple[int, ...] = eqx.field(static=True)  # REQUIRED
    _dtype: jnp.dtype = eqx.field(static=True)  # REQUIRED

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self._shape, self._dtype)

    def materialise(self):
        return jnp.zeros(self._shape, self._dtype)
```

`materialise()` raising is a legitimate, common design — `Unitful`, `LoraArray`,
and `NamedArray` all do it, because materialising would silently discard the very
information the type exists to carry. `LoraArray` and `NamedArray` expose an
`allow_materialise` flag so users can opt into the fallback. If you see a
`materialise()` that raises, do not "fix" it.

`eqx.field(converter=jnp.asarray)` on the payload is a cheap way to accept
lists/scalars without writing `__init__`.

## Writing dispatch rules

`@quax.register(primitive)` takes the `jax.extend.core.Primitive`, and dispatches
on the **type annotations** — plum reads them, so they are load-bearing, not
documentation.

**Name the rule; never `def _(...)`.** Dispatch does not read the name, but
everything you debug with does: tracebacks, `plum` ambiguity and redefinition
errors, and profiles all identify a rule by its function name, and a module of
rules all called `_` makes every one of them indistinguishable. Name it after
the primitive and the types it dispatches on — `add_meters_meters`,
`mul_meters_array_like`, `select_n_unitful` — matching the existing
`convert_element_type_zero` in `quax.examples.zero` and `cond_quax` in
`quax._primitives`.

Find the primitive behind a `jnp` function by tracing it:

```python
print(jax.make_jaxpr(jnp.square)(jnp.arange(3.0)))  # shows integer_pow
```

Positional arguments are the primitive's operands; keyword arguments are its
params (`bind(*args, **params)`). **Accept `**kw` and forward it** — params come
and go across JAX versions (`out_dtype` on `mul_p`, `out_sharding` on
`reduce_sum_p`, `sharding` on `broadcast_in_dim_p`), and a rule with a rigid
signature breaks on upgrade.

For mixed types, register a pair, using `ArrayLike | quax.ArrayValue` for the
operand you do not own:

```python
@quax.register(lax.mul_p)
def mul_meters_array_like(x: Meters, y: ArrayLike, /, **kw) -> Meters:
    return Meters(lax.mul_p.bind(x.array, y, **kw))


@quax.register(lax.mul_p)
def mul_array_like_meters(x: ArrayLike, y: Meters, /, **kw) -> Meters:
    return Meters(lax.mul_p.bind(x, y.array, **kw))
```

Annotating the other operand as `ArrayLike | quax.ArrayValue` and then
**redispatching** through a nested `quax.quaxify` is what lets your type
interoperate with types you have never heard of — the other author's rules handle
their half. This is the preferred pattern for library types.

A rule may return a plain array rather than your type when that is the honest
answer (comparisons returning bools, `integer_pow` with `y=0` returning ones).

If two rules can both match — `(Meters, ArrayLike | ArrayValue)` and
`(ArrayLike | ArrayValue, Meters)` both match a `(Meters, Meters)` call — plum
raises `AmbiguousLookupError`. Fix it by adding the specific rule with
`precedence=1`:

```python
@quax.register(lax.mul_p, precedence=1)
def mul_meters_meters(x: Meters, y: Meters, /, **kw) -> Meters:
    return Meters(lax.mul_p.bind(x.array, y.array, **kw))
```

Prefer resolving ambiguity with nested `quaxify(fn, filter_spec=...)` at the call
site when the clash is between two libraries' types: registering a cross-library
rule mutates a global dispatch table and commits you to an implementation for a
combination you may not understand. Order matters — the outer quaxify sees the
type it selects last.

## Overriding Value.default

`default` is a rule for *every* primitive at once, keyed only on your type. Use
it when the behaviour is uniform across primitives — a tag that propagates
through everything (detecting the backward pass, quantisation policy), rather
than per-primitive semantics.

```python
class Tagged(quax.ArrayValue):
    array: ArrayLike

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(jnp.shape(self.array), jnp.result_type(self.array))

    def materialise(self):
        return self.array

    @staticmethod
    def default(primitive, values, params):
        raw = [x.array if isinstance(x, Tagged) else x for x in values]
        out = primitive.bind(*raw, **params)
        return [Tagged(o) for o in out] if primitive.multiple_results else Tagged(out)
```

Only one type per expression may override `default` (dispatch rule 4). Two
`default`-overriding types meeting in one computation is a hard error, so this is
a strong commitment for a library type — per-primitive rules compose, `default`
does not.

## Boundaries

**Construct Values outside the quaxify boundary, not inside.** Constructing one
inside appears to work — it is combining it with values that *did* cross the
boundary that fails, with `TypeError: unsupported operand type(s) for +:
'_QuaxTracer' and 'YourType'`. A value built inside has no trace to associate
with (which of two nested quaxifies would own it?), so it never becomes a
tracer. Build your values, *then* cross the boundary.

**Inside the boundary your type looks like an array.** That is the entire point,
and it has a consequence for runtime typechecking: the jaxtyping/beartype import
hook installs *inside* the quaxify, so annotations on the wrapped function must
describe arrays, not your Value type. A function annotated
`q: Shaped[UnitfulArray, "3"]` will fail typechecking when quaxify hands it something
that presents as `f64[3]`.

**`__array__` must not silently strip.** If your type carries information a bare
array cannot (a unit, an axis name), returning a stripped array from `__array__`
converts a loud failure into a wrong number — `UnitfulArray(90, "deg")` becoming
`90.0` for a radian consumer is a 57x error nothing will catch. This is reached
implicitly: `np.asarray` always used it, and `jax.numpy.asarray` began using it in
jax 0.10 where it previously raised. Raise instead, and name the explicit
conversion in the message.

**Values nest inside pytrees.** Everything crossing the boundary is
`tree_map`ped, so Values inside `eqx.Module`s and containers are wrapped too. If
one is not being wrapped, suspect how the object was constructed rather than
quax's traversal.

**`filter_spec` passes things through unchanged.** `quax.quaxify(fn,
filter_spec)` runs `eqx.partition((fn, args, kwargs), filter_spec)` and quaxifies
only the dynamic half — the mechanism behind nested-quaxify redispatch. Accepts a
predicate or a nested bool pytree matching the arguments positionally.

## Transforms and control flow

Supported: `jit`, `vmap`, `grad`, `jax.custom_jvp`, and the control-flow
primitives `cond_p`, `while_p`, and `scan_p` (quax rebuilds the branch/body
jaxprs quaxified, and caches them). **Not supported: `jax.custom_vjp`.**

`quaxify` wraps a function *of arrays*, not a function of functions:

```python
def f(x):
    return (x * x).sum()


grad_f = quax.quaxify(jax.grad(f))  # correct
# grad_f = quax.quaxify(jax.grad)(f)        # wrong: grad is not a function of arrays
```

You should never need to touch JAX internals such as `pytype_aval_mappings` —
needing them means the transform is nested the wrong way round.

Operations whose *output shape depends on values* (`jnp.compress`,
`jnp.unique`, boolean masking) cannot be traced, in quax or in `jax.jit`. Pass
the `size=` argument, exactly as you would under `jit`.

## Don't hand-roll these

- **[quaxed](https://github.com/GalacticDynamics/quaxed)** — a whole pre-quaxified
  namespace: `import quaxed.numpy as jnp` instead of writing your own
  `quaxify(jnp.foo)` wrappers one at a time.
- **[quax-blocks](https://github.com/GalacticDynamics/quax-blocks)** — mixins for
  the ~40 dunder methods (`__add__`, `__radd__`, comparisons, bitwise) that an
  array-ish class needs. Writing them by hand is the single biggest source of
  boilerplate in a quax type.

Both are used heavily by [unxt](https://github.com/GalacticDynamics/unxt) (units)
and [coordinax](https://github.com/GalacticDynamics/coordinax) (coordinates),
which are the largest real-world quax types and worth reading before designing
your own.

## Quax, hijax, or both

`jax.experimental.hijax` is JAX's own extension API for custom *types*: you
write a `HiType` and a `HiPrim` per operation, and the type appears in jaxprs as
one typed value. It is not a replacement for quax and quax is not built on it.
They answer opposite questions — quax runs *existing* code on your type; hijax
gives a *new* type its own operations, and nothing existing applies to it.

Two questions decide it:

1. **Does code you do not own have to work with the type?** (`jnp`, diffrax,
   somebody's model.) If yes you need quax; a hijax type is invisible to all of
   it.
2. **Must the type do something quax cannot?** Quax's blind spot is that your
   metadata sits on a pytree and JAX never acts on it. Four things follow that
   quax cannot reach: a cotangent type that differs from the primal's; invariants
   checked at trace time and printed in the jaxpr; custom batching semantics;
   sharding carried in the type.

| Situation | Use |
|---|---|
| Existing code must run; metadata just rides along | **quax alone** — the default |
| You own every call site, and the type is not array-shaped (a box, a log, a handle) or needs its own lowering (a fused/Pallas kernel) | **hijax alone** |
| Existing code must run **and** you need one of the four capabilities above | **both**: a hijax value as the single leaf of a `quax.ArrayValue` |

Default to quax alone. It supports every JAX from 0.7.2, the value stays a
pytree (so `eqx.filter_*`, Optax and `jax.tree` keep working), and a primitive
with no rule falls back to `materialise` instead of erroring. Hijax costs one
primitive — typing rule, `expand`, and a rule per transform — for *every*
operation you want, so the hybrid suits a dozen ops, not all of `jnp`.

Hard constraints, if you do reach for hijax:

- **A hijax value must not be a pytree.** Every transform flattens a pytree
  before consulting the hijax registry, so a pytree value can never carry a hijax
  type. `quax.Value` is an `eqx.Module`, so it can never *be* a hijax value —
  only hold one.
- **`ArrayValue.aval()` must keep returning a `ShapedArray`.** `jax.Array`'s
  `isinstance` check reads the tracer's aval and every `jnp` function gates on
  it, so a `HiType` aval makes `jnp` reject your tracers. Build the `ShapedArray`
  *from* the leaf's hijax type instead.
- **Under a trace, do not construct the hi value class or read its attributes.**
  Both are legal only inside `expand` and the type's own methods; elsewhere apply
  a primitive. A quax tracer also will not forward attributes from a hijax leaf,
  though a bare hijax tracer does.
- **Do not hand-roll the wrapper.** `quax.experimental.hijax` has `HiValue`
  (supplies `aval`/`materialise` and the `leaf` field, correctly) and
  `register_rules(cls, {primitive: hijax_fn})`, which generates the dispatch
  rules including the mixed operand combinations. It also re-exports the
  hijax names, so a rename upstream never reaches your code.
- Working example: `quax.examples.hijax`. User-facing guidance:
  `docs/hijax.md`. Why quax is not built on hijax, with the three constraints
  above spelled out: `docs/why-not-built-on-hijax.md`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Quaxified code is ~100x slower than expected | No outer `jax.jit`. Wrap with `jax.jit(quax.quaxify(fn))`. |
| Custom type silently becomes a plain array | No rule matched and nothing overrode `default`, so everything materialised (rule 3). Register the primitive, or make `materialise()` raise to surface it. |
| `TypeError: Multiple array-ish types {...} are specifying default process rules` | Two types in one computation both override `Value.default`. Only one may; convert one to per-primitive rules. |
| `AmbiguousLookupError: <prim>_dispatcher(...) is ambiguous` | Two registered rules match the call equally well. Add the specific `(MyType, MyType)` rule with `precedence=1`. |
| `TypeError: Primitive ... operand ... neither a quax.Value nor array-like` | A genuine non-array operand (`None`, `str`) reached a primitive. Usually a bug in the calling code, not in the rule. |
| `Gradient only defined for scalar-output functions` from `quaxify(jax.grad)(f)` | Transform nested wrongly. Write `quaxify(jax.grad(f))`. Never register `pytype_aval_mappings`. |
| jaxtyping/beartype rejects your Value inside a quaxified function | Typechecking runs *inside* the boundary, where the Value presents as an array. Annotate arrays, not Value types. |
| `TypeError: unsupported operand type(s) for +: '_QuaxTracer' and 'MyType'` | A Value was constructed *inside* the quaxified function; only values that crossed the boundary are tracers. Construct it outside and pass it in. |
| `TypeError: got an unexpected keyword argument` in your rule | A JAX version added a primitive param. Accept `**kw` and forward it. |
| Rule never fires for `jnp.zeros_like`/`empty_like` | Those lower to a broadcast of a fresh scalar, so your type is never an operand. There is nothing of yours to dispatch on. |
| `jnp.compress`/boolean-mask style call fails under quaxify | Value-dependent output shape. Pass `size=`, as under `jit`. |
| `UnexpectedTracerError` / tracer leak | Historically real quax bugs (#42, #68), since fixed — update quax first. Tests run with `JAX_CHECK_TRACER_LEAKS=1`, which surfaces these early. |
| `KeyError: <primitive>` from inside `process_primitive` | No rule for the primitive and the fallback could not handle it. Register a rule for it, even one that just binds the materialised operands. |
| Rule works eagerly, breaks under `scan`/`while`/`cond` | Those re-trace the body with quaxified jaxprs; all branches must return the same pytree structure. |

## Version notes

Written against quax v0.4.x. `patrick-kidger/quax` (0.2.x) is the version most
prior knowledge describes; PyPI `quax` is now published from `nstarman/quax`,
which added JAX 0.9–0.11 support, `scan_p`, and large trace-path speedups.

| Old / wrong | Current |
|---|---|
| `quax._core` | Split into `_values.py`, `_quaxify.py`, `_dispatch.py`, `_primitives.py`, `_trace.py`, `_module.py`. Never cite `_core.py`. |
| `quax.lora`, `quax.zero` | `quax.examples.lora`, `quax.examples.zero` (old names still alias). |
| `jax.core.Primitive` in annotations | `jax.extend.core.Primitive`. |
| Registering `jax.core.pytype_aval_mappings` for your type | Never needed; indicates a wrongly-nested transform. |
| `_DenseArrayValue` | Internal and `@final`. Never instantiate or reference it. |

`jax.experimental.hijax` is experimental and renames things: `HiType` and
`register_hitype` arrived in JAX 0.8.2, `HiPspec` in 0.9.2, `MappingSpec` became
public in 0.11.0, and `VJPHiPrimitive` becomes `HiPrim` after 0.11.1. Do not
spell either name yourself: import them from `quax.experimental.hijax`, which
resolves them by probing (see `_resolve` in `src/quax/experimental/hijax.py`)
and carries the JAX floor as `HIJAX_FLOOR`. See "Quax, hijax, or both" above.

JAX-version-sensitive surfaces to expect churn in: primitive params (`sharding`,
`out_sharding`, `out_dtype`), primitives that only exist in newer versions
(`stack_p`, `tile_p`, `unstack_p`), and `scan_p`'s bind signature (changed in
0.11). Gate registrations on a version check when supporting a range.
