# Dispatchable functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Value type replace a whole function — one quax owns (`quax.dispatchable`) or a JAX `custom_jvp` such as JAX 0.12's `_hypot` — soundly under any composition of `jit`, `vmap` and `grad` on either side of `quaxify`.

**Architecture:** A quax-owned primitive `quax_dispatch_call_p` whose batching rule turns an inner `vmap` into an `n_batch` parameter; one generic quax rule resolves the function's own plum table and applies `vmap^n_batch(rule)` to the Values. Registering a `jax.custom_jvp` swaps its `fun` for a `DispatchableFunction`, so JAX's transforms see the primitive before they rewrite the body, and JAX's own JVP rule is kept. Replaces the branch's first approach (rules keyed by `func_src_info`), which is reverted first.

**Tech Stack:** JAX ≥ 0.7.2 (`jax.interpreters.{ad, batching, mlir}`), plum, equinox, pytest + Sybil.

**Spec:** `docs/superpowers/specs/2026-10-20-register-custom-jvp-design.md` (revised version, commit 450945f)

## Global Constraints

- Supported JAX: 0.7.2 onwards. CI runs the lock (JAX 0.10.2) and `--resolution lowest-direct`.
- Always `uv run`. Verify other JAX versions with overlays only: `uv run --with 'jax[cpu]==0.12.0' …` and `uv run --with 'jax[cpu]==0.7.2' …`. Never `uv run --resolution …`, which rewrites `uv.lock`. On 0.7.2, add `--ignore=tests/unit/test_experimental_hijax.py --ignore=tests/unit/test_hijax_interop.py --ignore=tests/usage/test_hijax.py`, since hijax needs ≥ 0.10.2.
- Tests run with `JAX_CHECK_TRACER_LEAKS=1` and beartype on.
- Private JAX imports go through `src/quax/_compat.py`.
- Every registered rule is named after its target and types, never `def _`.
- No matching rule ⇒ today's behaviour. Plain JAX (no quax) ⇒ same values, derivatives, lowering.
- Don't modify `process_primitive`'s fast path.
- Primitive params must be hashable and stable across calls. `fn` is the `DispatchableFunction` itself, never a per-call lambda, or jit caches miss. No param may be named `key`: on JAX 0.7.2 that collides with an internal cache's argument (`got multiple values for argument 'key'`).
- Test rules are global and permanent: target only functions defined for that test module.

## Review Focus

1. A Value type with no `transpose_p` / `broadcast_in_dim_p` rule under an inner `vmap` with misaligned `in_axes` → a loud error (materialise refusal), never a wrong value. Test in Task 2 using `MyArray`, which has both rules, and assert values.
2. A rule that returns plain arrays where the body returns arrays, or a Value where the body returns an array → shape/dtype check passes; only treedef/shape/dtype mismatches raise. Test in Task 2.
3. Swapping a `custom_jvp` that was already called and jit-cached before `register` → the next quaxified call uses the rule (because of `jax.clear_caches()`). Test in Task 3.
4. A dispatchable function called with only plain arrays under `quaxify` → no Value, so it binds on the parent and has no `quaxify` overhead per call. Test in Task 2 (result equals the body).
5. `jax.jit` of a dispatchable function called twice with the same shapes → traced once, so params are stable. Test in Task 2 by counting traces of the body.

---

### Task 1: Revert the first approach

**Files:**
- Revert commits `251d337`, `13ca758`, `dfbd3a2` (in that order), then restore one hunk in `AGENTS.md`.

**Interfaces:**
- Produces: a tree where `register`, `_trace.py`, `_compat.py`, the Unitful example, SKILL.md and autodiff.md are as on `origin/main`. Keep the AGENTS.md Doctests-bullet correction from `251d337`; it is independent of the approach.

- [ ] **Step 1:** `git revert --no-commit 251d337 13ca758 dfbd3a2`
- [ ] **Step 2:** Re-apply the corrected Doctests bullet in `AGENTS.md` from `251d337`. Do not re-apply the `custom_jvp rules are keyed by a private JAX string` pitfall.
- [ ] **Step 3:** Run `uv run pytest tests/unit/test_custom_jvp.py tests/usage/test_unitful.py skills/quax/SKILL.md docs/autodiff.md -q`. Expected: PASS, and `tests/unit/test_register_custom_jvp.py` no longer exists.
- [ ] **Step 4:** Commit `⏪ revert: drop source-info-keyed custom_jvp rules (unsound under inner vmap)`.

---

### Task 2: `quax.dispatchable` and the dispatch primitive

**Files:**
- Create: `src/quax/_dispatchable.py`, containing the primitive, `DispatchableFunction`, `dispatchable`, and the impl/abstract-eval/lowering/batching/JVP rules. It must not import `_dispatch`.
- Modify: `src/quax/_dispatch.py`. `register` accepts a `DispatchableFunction`, `_rules` key type becomes `jexc.Primitive | DispatchableFunction`, and the dispatcher name uses `fn.__name__`.
- Modify: `src/quax/_primitives.py`. Add the generic rule `dispatch_call_quax`, next to `jit_quax`.
- Modify: `src/quax/__init__.py`. Export `dispatchable` (and `DispatchableFunction` for annotations).
- Create: `tests/unit/test_dispatchable.py`.

**Interfaces:**
- Produces: `quax.dispatchable(fn: Callable) -> DispatchableFunction`. `DispatchableFunction` is a plain class, hashable by identity. It holds `fn`, carries `functools.update_wrapper` metadata, and `__call__(*args)` raises `TypeError` on any keyword argument.
- Produces: `dispatch_call_p` (`multiple_results=True`). `bind(*operands, fn: DispatchableFunction, in_tree: PyTreeDef, static: tuple[tuple[int, Any], ...], n_batch: int)`. Operands are the leaves that are `eqx.is_array`; `static` holds `(leaf_index, value)` for every other leaf.
- Produces: `_out_info(fn, in_tree, static, n_batch, avals) -> tuple[PyTreeDef, tuple[ShapedArray, ...]]`, `functools.cache`d. It uses `jax.eval_shape` of `vmap^n_batch` of the flattened body. Abstract eval and the generic rule share it.
- Produces: `quax.register(target: jexc.Primitive | DispatchableFunction | jax.custom_jvp, /, *, precedence=0)`. The `custom_jvp` branch comes in Task 3; until then it raises `TypeError` like other unsupported targets. Rules for a `DispatchableFunction` live in `_rules[fn]`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_dispatchable.py`. Define a module-local body that `MyArray` can run inline (e.g. `jnp.sin(x) * y`). Register a rule whose value differs (e.g. `MyArray(x.array + 100 * y.array)`). Compare each case against the same `jax` transform applied to a plain-array reference function computing the rule's formula. Test names:

```python
test_plain_jax_unchanged            # eager, jit, grad, vmap == body
test_rule_eager
test_rule_jit_of_quaxify
test_rule_quaxify_of_jit
test_rule_vmap_of_quaxify
test_rule_inner_vmap                # parametrize in_axes (0,0),(0,1),(1,0),(0,None)
test_rule_inner_vmap_of_jit         # same parametrize
test_rule_inner_jit_of_vmap         # same parametrize
test_rule_nested_inner_vmap
test_rule_outer_grad
test_pytree_args_and_outputs        # dict in, (a, {"b": c}) out
test_static_python_scalar_arg       # f(x, 3) with 3 static; rule sees 3
test_no_rule_falls_back_to_body     # DenseArray input (no rule) -> body result
test_plain_arrays_under_quaxify     # Review Focus 4
test_jit_traces_once                # Review Focus 5: body trace counter == 1 after 2 calls
test_rule_output_mismatch_raises    # wrong treedef -> ValueError naming fn; wrong shape -> ValueError
test_rule_returns_array_for_array_output   # Review Focus 2
test_keyword_args_raise
test_register_rejects_other_objects # lambda, custom_vjp -> TypeError
test_precedence_picks_rule
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/test_dispatchable.py -q`. Expected: FAIL (`quax.dispatchable` missing).
- [ ] **Step 3:** Implement `_dispatchable.py`, then `register`, then `dispatch_call_quax`:
  - **Batching rule:** find the batch size from the first batched operand. Move each batched operand's axis to 0 (`jnp.moveaxis`). Broadcast each unbatched operand (`jnp.broadcast_to`). Re-bind with `n_batch + 1`. Return out dims of all 0. Unbatched dims are `None` (`jax.interpreters.batching.not_mapped` does not exist on 0.10).
  - **JVP rule:** `jax.jvp(vmap^n_batch(flat body), tuple(primals), tuple(ad.instantiate_zeros(t) for t in tangents))`. Use tuples: lists make a primal/tangent tree mismatch.
  - **Lowering:** `mlir.register_lowering(p, mlir.lower_fun(impl, multiple_results=True))`.
  - **`dispatch_call_quax`:**
    - If no operand is a Value (`_is_value`), return `dispatch_call_p.bind(...)`.
    - Otherwise rebuild the per-call args and `rule.resolve_method(args)`.
    - Miss: `_Quaxify(vmap^n_batch(flat body), True, dynamic=False)(*operands)`.
    - Hit: `vmap^n_batch` of a function that rebuilds the args from operand leaves, calls the method, flattens, and records the output treedef. Then compare the treedef and each leaf's `shape`/`dtype` (a Value's via `type(v).aval(v)`) with `_out_info`, and raise `ValueError(f"quax rule for {fn.__name__!r} returned …; expected …")` on a mismatch.
  - **`register` docstring:** the per-example contract, plus the requirement that Values have `transpose`/`broadcast_in_dim` rules for misaligned inner `vmap`.
- [ ] **Step 4:** Run `uv run pytest tests/unit/test_dispatchable.py -q`, then with `--with 'jax[cpu]==0.12.0'` and `--with 'jax[cpu]==0.7.2'`. Expected: PASS for all three.
- [ ] **Step 5:** Run `uv run pytest tests/unit -q`. Expected: no new failures.
- [ ] **Step 6:** Run `uv run prek run --all-files`, then commit `✨ feat: quax.dispatchable — whole-function dispatch via a quax-owned primitive`.

---

### Task 3: Registering a `jax.custom_jvp` swaps its body

**Files:**
- Modify: `src/quax/_dispatch.py` (the `register` `custom_jvp` branch)
- Create: `tests/unit/test_register_custom_jvp.py`

**Interfaces:**
- Consumes: `DispatchableFunction` and `dispatchable` from Task 2.
- Produces: `register(cjvp)`. If `isinstance(cjvp.fun, DispatchableFunction)`, reuse it. Otherwise `cjvp.fun = dispatchable(cjvp.fun)` and call `jax.clear_caches()`. In both cases the rule is registered for `cjvp.fun`. Rules take `cjvp`'s full positional arguments; `nondiff_argnums` arguments arrive as static leaves when they aren't arrays.

- [ ] **Step 1:** Write the failing tests. Each uses its own module-local `custom_jvp` with a distinctive JVP rule (e.g. derivative × 10):

```python
test_custom_jvp_rule_applies               # eager, jit(quaxify), quaxify(jit), inner vmap (0,1)
test_custom_jvp_keeps_its_jvp_rule         # plain jax.grad == 10 * cos  (after registration)
test_custom_jvp_inner_grad                 # quax.quaxify(jax.grad(f))(MyArray) uses the 10x rule on Values
test_custom_jvp_outer_grad
test_custom_jvp_registered_twice_swaps_once  # cjvp.fun identical after second register; both rules dispatch
test_custom_jvp_registered_after_jit_cache   # Review Focus 3: call jax.jit(f) first, then register, then quaxify(jax.jit(f)) uses rule
test_custom_jvp_nondiff_argnums
```

- [ ] **Step 2:** Run `uv run pytest tests/unit/test_register_custom_jvp.py -q`. Expected: FAIL (`TypeError` from `register`).
- [ ] **Step 3:** Implement the branch. Extend the docstring: the swap is global and visible in jaxprs outside quax; it calls `jax.clear_caches()`; private targets should be looked up with `getattr`.
- [ ] **Step 4:** Run with the lock, `--with 'jax[cpu]==0.12.0'` and `--with 'jax[cpu]==0.7.2'`. Expected: PASS. Then run `uv run pytest tests/unit -q`. Expected: no new failures.
- [ ] **Step 5:** Run prek and commit `✨ feat: quax.register(custom_jvp) swaps in a dispatchable body`.

---

### Task 4: `Unitful` example: `hypot` and the rules it needs

**Files:**
- Modify: `src/quax/examples/unitful/_core.py`
- Modify: `tests/usage/test_unitful.py`

**Interfaces:**
- Consumes: `register(custom_jvp)` from Task 3.
- Produces:
  - `sub_unitful_unitful` (same units required, `ValueError` otherwise, matching `add_unitful_unitful`'s message style);
  - `div_p` rules for `(Unitful, Unitful)`, `(Unitful, ArrayLike)` and `(ArrayLike, Unitful)`, subtracting exponents and dropping zeros;
  - `transpose_unitful`;
  - `hypot_unitful_unitful(x1, x2) -> tuple[Unitful, Unitful, Unitful]`, registered when `getattr(jax._src.numpy.ufuncs, "_hypot", None)` is a `jax.custom_jvp`, with a comment on why.

- [ ] **Step 1:** Write the failing tests:
  - `test_sub_requires_matching_units`, `test_div_combines_units`, `test_transpose_keeps_units` (these run on every JAX version);
  - gated on `_hypot` existing: `test_hypot_keeps_units`, `test_hypot_under_jit` (both orders), `test_hypot_inner_vmap_misaligned` (`in_axes=(0, 1)` against plain `jax.vmap(jnp.hypot)`), `test_hypot_outer_grad` (0.6), `test_hypot_inner_grad` (`quaxify(grad)`, 0.6), `test_hypot_units_mismatch` (eager and under `jit`, `match="units"`).
- [ ] **Step 2:** Run `uv run --with 'jax[cpu]==0.12.0' pytest tests/usage/test_unitful.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the rules.
- [ ] **Step 4:** Run with `--with 'jax[cpu]==0.12.0'` (PASS) and with the lock (hypot tests SKIPPED, the rest PASS).
- [ ] **Step 5:** Commit `✨ feat(unitful): hypot via custom_jvp registration; sub/div/transpose rules`.

---

### Task 5: Docs

**Files:**
- Modify: `skills/quax/SKILL.md`. Add a `### Replacing a whole function` section before `## Overriding Value.default`.
- Modify: `docs/autodiff.md` (`## Custom derivative rules`)
- Modify: `AGENTS.md` (Architecture table row for `_dispatchable.py`; Key Pitfalls)

**Interfaces:**
- Consumes: Tasks 2–3 contracts. SKILL.md and autodiff.md run under Sybil on the locked JAX 0.10.2, so examples use `quax.dispatchable` or a user-defined `custom_jvp`, and mention `_hypot` in prose only.

- [ ] **Step 1: SKILL.md section.** Include:
  - when to use it;
  - a runnable `quax.dispatchable` example with a `Unitful` rule and an inner `vmap`;
  - registering a `custom_jvp` and the swap's consequences;
  - the per-example rule contract;
  - the requirement that Values have `transpose`/`broadcast_in_dim` rules.
- [ ] **Step 2: autodiff.md note.** A replaced function keeps its own JVP rule, and grad works on either side of `quaxify`. Link with `[`quax.register`][]`.
- [ ] **Step 3: AGENTS.md.**
  - Architecture row for `_dispatchable.py`.
  - Pitfall: registering a `custom_jvp` mutates `cjvp.fun` globally and clears jit caches.
  - Pitfall: primitive params must be hashable and stable, and never named `key`.
- [ ] **Step 4:** Run `uv run pytest skills/quax/SKILL.md docs/autodiff.md -q`. Expected: PASS. Also confirm the SKILL example fails without its rule ("Refusing to materialise").
- [ ] **Step 5:** Commit `📝 docs: replacing whole functions with quax.dispatchable / quax.register`.

---

### Task 6: Full verification, then the PR

- [ ] Run `uv run pytest -q`. Expected: PASS.
- [ ] Run `uv run --with 'jax[cpu]==0.12.0' pytest -q`. Expected: PASS.
- [ ] Run `uv run --with 'jax[cpu]==0.7.2' pytest tests/unit tests/usage -q` with the hijax ignores. Expected: PASS.
- [ ] Run `uv run prek run --all-files`. Expected: clean.
- [ ] Run the final whole-branch review, then `superpowers:finishing-a-development-branch` to push and open the PR against `main`. Confirm with the user before pushing.
