# `quax.register` for `jax.custom_jvp` functions — design

## Problem

On JAX 0.12, `jnp.hypot` is `jit(lambda x1, x2: _hypot(x1, x2)[0])`, where
`_hypot` is a `jax.custom_jvp`. `_QuaxTrace.process_custom_jvp_call` never
consults the rule table: it re-binds `custom_jvp_call` on the parent trace with
the body wrapped, so each primitive in the body is dispatched individually. A
Value type has no way to say "this is `_hypot`; here is my implementation".

The body cannot be made to work instead. `_hypot` compares `max(|x1|, |x2|)`
against bare constants (`ge e 9.2e18`, `eq c inf`) and mixes bare constants into
`select_n`, which a unit-carrying type (unxt `Quantity`, `quax.examples.unitful.Unitful`)
cannot allow without becoming unit-unsafe. Its JVP rule needs further rules.

## Goal and success criteria

A rule registered once for a `custom_jvp` replaces it for the Value types it
dispatches on:

- `quaxify(jnp.hypot)(q1, q2)` works for `Unitful`, eagerly, under
  `quaxify(jit(...))` and `jit(quaxify(...))`, under `vmap`, and under an outer
  `jax.grad` (gradient of the rule, which uses JAX's own derivative when the rule
  calls JAX ops on plain arrays).
- A `custom_jvp` without a matching rule behaves exactly as today.
- Supported on every JAX version quax supports (0.7.2 onwards).

Out of scope: `custom_vjp`; `jax.jvp`/`grad` taken *inside* `quaxify` (JAX's
`JVPTrace` sits below quax and consumes the `custom_jvp` before quax sees it —
no quax-level hook can intercept it).

## Alternatives considered

Prototyped in throwaway scripts against a common harness:

| | Approach | Outcome |
|---|---|---|
| D | Teach the Value type every primitive in the body | Needs unit-unsafe rules (Quantity vs bare constant comparisons and `select_n`); grad still fails. Rejected. |
| C | Rule on `jit_p` keyed on `params["name"] == "hypot"` | Works with no quax change, but captures every jitted call with that type signature, needs a hand-written fallback to `jit_quax`, collides with any user jit named `hypot`, and becomes ambiguous when two libraries register `jit_p`. Only works because JAX happens to wrap `hypot` in `jit`. A possible downstream stopgap, not a design. |
| A | Dispatch `custom_jvp_call_p` through the rule table; rule checks `fun.debug_info.func_name` and returns `NotImplemented` to decline | Works, but keys on a private, collision-prone name and invents a decline convention. |
| F | Bind with materialised Values | Not viable; `Unitful`/`Quantity` refuse to materialise. |
| **B** | `quax.register(custom_jvp_obj)`, keyed by source location | **Chosen.** Precise key, plum miss falls back to today's behaviour, same API shape as primitive rules. |

## Design

### API

```python
@quax.register(jax._src.numpy.ufuncs._hypot)
def hypot_unitful_unitful(x1: Unitful, x2: Unitful):
    h = jnp.hypot(x1.array, x2.array)
    return [Unitful(h, x1.units), Unitful(x1.array / h, {}), Unitful(x2.array / h, {})]
```

- `register(primitive, /, *, precedence=0)` accepts a `jexc.Primitive`
  (unchanged) or a `jax.custom_jvp` instance. Anything else, including a
  `jax.custom_vjp`, raises `TypeError`.
- The rule's positional arguments are the `custom_jvp`'s inputs, flattened to
  leaves. `nondiff_argnums` arguments are closed over by JAX and never passed.
- No keyword arguments are passed.
- The rule returns a sequence: the `custom_jvp`'s output, flattened to leaves
  (`[out]` for a single array output).
- Dispatch, `precedence` and ambiguity behave as for primitive rules.

### Key

At registration, `fun_sourceinfo(cjvp.fun)` (from `jax._src.api_util`, imported
through `_compat.py`) gives a string such as
`"_hypot at …/jax/_src/numpy/ufuncs.py:3984"`. It is the key into the existing
`_rules` dict, whose key type widens to `jexc.Primitive | str`. No synthetic
primitive is created. If `fun_sourceinfo` returns `None`, registration raises
`ValueError`.

At call time the same string is `fun.debug_info.func_src_info`. Verified to be
present and identical on JAX 0.7.2 and 0.12 in all three cases: eager call,
`custom_jvp` already staged into a jaxpr (`quaxify(jit(f))`), and `jit(quaxify(f))`.
Because the key is computed from the live object, it tracks JAX's line numbers
across versions.

### Call path

`process_custom_jvp_call` (both JAX-version branches) first calls one helper,
`_custom_jvp_rule(self, fun, tracers)`, which returns output tracers or `None`:

1. `_rules.get(getattr(fun.debug_info, "func_src_info", None))`; no entry → `None`.
2. Convert tracers to values exactly as `process_primitive` does; resolve a plum
   method. `NotFoundLookupError` → `None`.
3. `jax.eval_shape(fun.call_wrapped, *ShapeDtypeStruct(aval) for each input)`,
   where the aval of a Value is `Value.aval()`. This is required, not just a
   check: on an eager `custom_jvp` call JAX reads the output tree from a store
   filled only when `fun` runs, so skipping it raises
   `StoreException: neither store occupied`. It also yields the expected output
   count.
4. Call the method under `core.set_current_trace(self.parent_trace)`, as
   primitive rules are called.
5. If the number of outputs differs from step 3, raise `ValueError` naming the
   `custom_jvp` and both counts. Otherwise wrap each output with
   `_wrap_if_array` in a `_QuaxTracer`.

`None` falls through to the existing inline path, unchanged.

No dispatch cache on this path: `custom_jvp` calls are far rarer than primitive
binds, and `process_primitive`'s hot path is left untouched. Add a cache if a
profile shows the need.

### Semantics to document

- The rule replaces the function and its derivative. Under an outer `grad` the
  rule's own computation is differentiated; writing the rule with JAX ops on
  plain arrays (`jnp.hypot(x.array, …)`) keeps JAX's derivative rule.
- `jvp`/`grad` inside `quaxify` does not reach the rule.
- `custom_jvp`s built by one factory share a source location and therefore a
  rule.
- Targets such as `ufuncs._hypot` are private JAX objects that exist only on some
  versions; register them behind `getattr(module, name, None)`.

## Testing

`tests/unit/test_custom_jvp.py`, using the `MyArray` fixture:

- rule applies: eager, `quaxify(jit(f))`, `jit(quaxify(f))`, `vmap`, outer `grad`;
- unregistered `custom_jvp` and a type miss both fall back to inline;
- wrong output count raises `ValueError`;
- registering a non-primitive/non-`custom_jvp`, or a `custom_vjp`, raises `TypeError`;
- `nondiff_argnums` arguments are not passed to the rule;
- `precedence` picks between overlapping rules.

`src/quax/examples/unitful`: register `hypot_unitful_unitful` when
`jax._src.numpy.ufuncs` has `_hypot`, with a usage test in `tests/usage/`
(skipped on JAX < 0.12) covering eager, jit, outer grad, and a units mismatch.

## Docs

- `register` docstring: the `custom_jvp` form, rule signature, output contract.
- `skills/quax/SKILL.md`: a section on overriding a `custom_jvp` (code blocks run
  under Sybil).
- `docs/autodiff.md`: short note in the custom-derivative section (also run
  under Sybil).
- `AGENTS.md`: pitfall noting the key is JAX's private source-info string, and
  correct the Doctests bullet, which lists only README.md and SKILL.md as
  Sybil-run although `conftest.py` also runs several `docs/` pages.
