# Dispatchable functions and `custom_jvp` overrides — design

Revised 2026-10-20. Supersedes the first version of this spec (rules keyed by a
`custom_jvp`'s source location), which a whole-branch review showed to be
unsound under a `vmap` inside `quaxify`.

## Problem

On JAX 0.12, `jnp.hypot` is `jit(lambda x1, x2: _hypot(x1, x2)[0])`, where
`_hypot` is a `jax.custom_jvp`. Its body compares `max(|x1|, |x2|)` against bare
constants and mixes bare constants into `select_n`, which a unit-carrying type
(unxt `Quantity`, `quax.examples.unitful.Unitful`) cannot allow without becoming
unit-unsafe. A Value type needs to replace the whole function, not each primitive
in it.

## Root cause any design must respect

A rule that replaces a *function* is correct only if every transform between
`quaxify` and the call sees the replacement. Primitive rules get this for free:
an inner `vmap` rewrites `add` into another `add` with moved axes. A JAX
`custom_jvp` does not: an inner `vmap` rewrites its *body* but keeps its
*identity* (same `DebugInfo`, and in the staged path no recorded transform). So
no rule keyed on a JAX-owned identity can be sound — the first version of this
design returned `[[5, 10], [15, 20]]` instead of `[[5, 13.4], [12.0, 20]]` for
`quaxify(vmap(jnp.hypot, in_axes=(0, 1)))`.

The fix is for quax to own the call: a primitive whose batching rule turns an
inner `vmap` into a parameter quax can read.

## Goal and success criteria

- A per-example rule registered once for a function replaces it for the Value
  types it dispatches on, under any composition of `jit`, `vmap` (any `in_axes`,
  nested, either side of `quaxify`) and `grad` (either side of `quaxify`).
- `quaxify(jnp.hypot)` works for `Unitful` on JAX ≥ 0.12 in all of the above,
  given ordinary `Unitful` rules for `sub`, `div`, `transpose`,
  `broadcast_in_dim`.
- With no matching rule, behaviour is exactly as today.
- Plain JAX (no quax) is unaffected in value, derivative and lowering.
- Supported on every JAX version quax supports (0.7.2 onwards).

Out of scope: `custom_vjp` (the same body swap would work; not needed yet);
keyword arguments to dispatchable functions; bodies with side effects
(`jax.debug.print`, callbacks).

## Alternatives considered

| | Approach | Outcome |
|---|---|---|
| D | Teach the Value type every primitive in the body | Needs unit-unsafe rules; grad still fails. Rejected. |
| C | `jit_p` rule keyed on `name == "hypot"` | Captures every jitted call of that signature; collides; identity-keyed, so also wrong under inner `vmap`. Rejected. |
| A | Generic `custom_jvp_call_p` rule checking `func_name` | Private name key; identity-keyed. Rejected. |
| B | `register(custom_jvp)` keyed by `func_src_info` | Implemented first. Silently wrong under inner `vmap`; callables without source info collide. Rejected. |
| B+guard | B, plus fall back unless `fun`'s jaxpr equals the registered function's | Sound but conservative: inner `vmap` and inner `grad` fail loudly; brittle jaxpr comparison. Rejected in favour of the below. |
| Upstream | JAX marks transformed `custom_jvp`s | Outside our control. |
| **P** | **quax-owned dispatch primitive + `custom_jvp` body swap** | **Chosen.** Prototype passes every case above, including inner `vmap` with misaligned axes and inner `grad` on JAX 0.12's `jnp.hypot`. |

## Design

### API

```python
@quax.dispatchable                         # a function you own
def myhypot(x, y): ...

@quax.register(myhypot)                    # per-example rule
def myhypot_unitful_unitful(x: Unitful, y: Unitful) -> Unitful: ...

@quax.register(jax._src.numpy.ufuncs._hypot)   # a JAX-owned custom_jvp
def hypot_unitful_unitful(x1: Unitful, x2: Unitful) -> tuple[Unitful, Unitful, Unitful]: ...
```

- `quax.dispatchable(fn) -> DispatchableFunction`. Calling it binds
  `quax_dispatch_call`; outside quax its value, derivative and lowering are
  `fn`'s. Positional arguments only; keyword arguments raise `TypeError`.
- `quax.register(target, /, *, precedence=0)` accepts a `jexc.Primitive`
  (unchanged), a `DispatchableFunction`, or a `jax.custom_jvp`. Anything else
  raises `TypeError`.
- A rule has the function's signature: it takes the function's arguments as
  pytrees and returns the function's output pytree. Rules are **per-example**:
  quax applies any `vmap` between `quaxify` and the call, so a rule is never
  shown a batch axis. Rules must therefore be vmappable (JAX operations; Values
  are pytrees).
- Dispatch and `precedence` behave as for primitive rules; each
  `DispatchableFunction` has its own plum table.

### The primitive

`quax_dispatch_call_p` (multiple results). `bind(*operands, fn, in_tree, static,
n_batch)` means: rebuild the arguments from `in_tree`, the array operands and the
non-array `static` leaves, and apply `fn` vmapped `n_batch` times over leading
axes; outputs are the flattened output leaves.

- Arguments: array-like leaves are operands; other leaves (Python scalars,
  strings, other hashable data) go into `static` as a hashable tuple of
  `(position, value)`.
- impl and lowering (`mlir.lower_fun`): `vmap^n_batch` of the body.
- abstract eval: `jax.eval_shape` of the same, cached on
  `(fn, operand avals, static, n_batch)`; the output treedef is recorded with it.
- batching: move each operand's batch axis to the front, broadcast unbatched
  operands to the batch size, re-bind with `n_batch + 1`, report output dims 0.
  The new axis is the outermost vmap level.
- JVP: `jax.jvp` of `vmap^n_batch` of the body — the body's own derivative. For a
  swapped `custom_jvp`, JAX's `custom_jvp` rule sits outside the body and takes
  precedence, so the function keeps its own derivative rule.
- No transpose rule (the primitive is not linear).

### quax side

One generic rule is registered on `quax_dispatch_call_p`:

1. No operand is a `quax.Value`: bind on the parent trace (plain JAX).
2. Otherwise unflatten the per-example arguments' types and resolve the
   function's plum table. Miss: return `quaxify(vmap^n_batch(body))(*args)` —
   today's behaviour of running the body under quax.
3. Hit: run `vmap^n_batch(rule)` on the Values (unflattened via `in_tree` and
   `static`), flatten the output.
4. Check the rule's per-example output treedef, and each leaf's shape and dtype
   (a Value's via `aval()`), against the cached abstract eval. Mismatch:
   `ValueError` naming the function, the expected and the actual.

The first version's interpreter hook is removed: `_QuaxTrace._custom_jvp_rule`,
`_compat.fun_sourceinfo`, `str` keys in `_rules`, and the `eval_shape` step;
`process_custom_jvp_call` returns to its previous form.

### The `custom_jvp` body swap

On the first `register` of a given `jax.custom_jvp`, quax wraps its body:
`d = dispatchable(cjvp.fun)`, `cjvp.fun = d`, records `cjvp -> d`, and calls
`jax.clear_caches()` so no jit cache keeps the old body. Later registrations reuse
`d`. Rules are registered into `d`'s table.

Consequences, to document:

- Every later trace of the function contains `quax_dispatch_call`, also outside
  quax. Values, derivatives and lowering are unchanged; jaxpr-walking tools see a
  primitive they do not know.
- JAX's own JVP rule (`_hypot_jvp`) still applies, now on Values, so the type
  needs rules for the arithmetic in it (`sub`, `div`, `mul`, …).
- An inner `vmap` with misaligned axes binds `transpose` / `broadcast_in_dim` on
  Values, so the type needs those rules.
- Targets such as `ufuncs._hypot` are private and version-dependent; register
  them behind `getattr(module, name, None)`.

## Testing

`tests/unit/test_dispatchable.py`, with the `MyArray` fixture, each case comparing
against the same computation on plain arrays:

- rule applies: eager; `jit(quaxify)`; `quaxify(jit)`; `vmap(quaxify)`;
  `quaxify(vmap)` with `in_axes` `(0, 0)`, `(0, 1)`, `(1, 0)`, `(0, None)`; nested
  `vmap`; `quaxify(vmap(jit))`; `quaxify(jit(vmap))`; outer `grad`;
- plain JAX unaffected: eager, `jit`, `grad`, `vmap`;
- pytree input and output; a Python-scalar static argument;
- no rule for the types: falls back to the body under quax;
- output treedef / shape / dtype mismatch raises `ValueError`;
- `register` rejects other objects and `custom_vjp` with `TypeError`;
  keyword arguments raise `TypeError`; `precedence` picks between rules;
- `custom_jvp` swap: rule applies; JAX's derivative rule is still used (plain
  `grad` matches); inner `grad` (`quaxify(grad(f))`) works; registering twice
  swaps once.

`src/quax/examples/unitful`: add `sub_p`, `div_p` (Unitful/array combinations)
and `transpose_p` rules; `hypot_unitful_unitful` returns `_hypot`'s 3-tuple,
registered when `ufuncs._hypot` is a `jax.custom_jvp`. Usage tests (skipped on
JAX < 0.12): eager, both jit orders, `quaxify(vmap(jnp.hypot, in_axes=(0, 1)))`,
outer and inner `grad`, units mismatch eagerly and under `jit`.

## Docs

- `quax.dispatchable` and `quax.register` docstrings: per-example rule contract,
  the body swap and its consequences.
- `skills/quax/SKILL.md`: section on replacing a whole function, with a runnable
  `quax.dispatchable` example and the `custom_jvp` swap (run under Sybil).
- `docs/autodiff.md`: note that a replaced function keeps its own JVP rule and
  works with `grad` on either side of `quaxify` (run under Sybil).
- `AGENTS.md`: pitfall on the body swap (global, clears jit caches); keep the
  corrected Doctests bullet.
