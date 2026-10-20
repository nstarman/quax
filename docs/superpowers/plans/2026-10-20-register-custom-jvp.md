# `quax.register` for `custom_jvp` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `quax.register` take a `jax.custom_jvp` object so a Value type can replace that function (e.g. JAX 0.12's `_hypot` behind `jnp.hypot`) instead of having its body dispatched primitive by primitive.

**Architecture:** Registration keys the existing `_rules` dict by the `custom_jvp`'s source-info string. `_QuaxTrace.process_custom_jvp_call` first tries a helper that looks the call up by `fun.debug_info.func_src_info`, resolves a plum method, runs `jax.eval_shape` on `fun` (fills JAX's output-tree store and gives the expected output count), and calls the rule on the parent trace; on any miss it falls through to today's inline path.

**Tech Stack:** JAX ≥ 0.7.2, plum, equinox, pytest + Sybil.

**Spec:** `docs/superpowers/specs/2026-10-20-register-custom-jvp-design.md`

## Global Constraints

- Supported JAX: 0.7.2 onwards (`ci.yml` runs the lock — JAX 0.10.2 — and `--resolution lowest-direct`).
- Always `uv run`; never bare `python`/`pytest`. Tests run with `JAX_CHECK_TRACER_LEAKS=1` and beartype on.
- Private JAX imports go through `src/quax/_compat.py`.
- Every registered rule is named after its target and types (`hypot_unitful_unitful`), never `def _`.
- A `custom_jvp` with no matching rule must behave exactly as today.
- `process_primitive`'s hot path is not modified.
- Rules registered in tests are global and permanent: target only `custom_jvp` functions defined for that purpose, never ones other tests use.
- Hypot tests need JAX ≥ 0.12, which neither CI job installs. Verify other JAX versions locally with a `--with` overlay
  (`uv run --with 'jax[cpu]==0.12.0' pytest …`, `uv run --with 'jax[cpu]==0.7.2' pytest …`). Never use
  `uv run --resolution …`: it rewrites `uv.lock`.

## Review Focus

1. A single-output `custom_jvp` whose rule returns a bare Value instead of `[value]` → clear `TypeError`, not a silent misread (`MyArray` defines `__len__`, so a length check alone would accept it). Test in Task 1.
2. A Python scalar positional argument (`f(x, 2.0)`) next to a Value → `eval_shape` must accept its aval and the rule still fires. Test in Task 1.
3. Running `eval_shape` while the quax trace is current, under `JAX_CHECK_TRACER_LEAKS=1` → no leak reported. Covered by the suite's global setting; Task 1's tests must pass with it on.
4. A rule under `vmap` sees per-example avals in `eval_shape` but batched values in the rule → output count still matches. Test in Task 1.
5. A rule that raises (e.g. a units mismatch) under `jit` → the user's error surfaces unchanged. Test in Task 2.

---

### Task 1: Register and dispatch `custom_jvp` rules

**Files:**
- Modify: `src/quax/_compat.py` (export `fun_sourceinfo`)
- Modify: `src/quax/_dispatch.py` (`_rules` key type, `register`)
- Modify: `src/quax/_trace.py` (new `_custom_jvp_rule`, called at the top of both `process_custom_jvp_call` definitions, ~L222 and ~L243)
- Create: `tests/unit/test_register_custom_jvp.py`

**Interfaces:**
- Produces: `quax.register(target: jexc.Primitive | jax.custom_jvp, /, *, precedence: int = 0)`. For a `custom_jvp`, the rule takes the flattened differentiable input leaves positionally, no kwargs, and returns a `list`/`tuple` of flattened output leaves.
- Produces: `_compat.fun_sourceinfo(fn: Callable) -> str | None`, re-exported from `jax._src.api_util.fun_sourceinfo` (exists 0.7.2–0.12).
- Produces: `_dispatch._rules: dict[jexc.Primitive | str, plum.Function]`.
- Produces: `_QuaxTrace._custom_jvp_rule(self, fun: lu.WrappedFun, tracers: Sequence[Any]) -> list[_QuaxTracer] | None`.

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_register_custom_jvp.py`

Define module-local `custom_jvp`s used by nothing else, each with a body and JVP that `MyArray` can run inline (e.g. `jnp.sin`, `x * y`), and rules whose results differ from the body so the tests can tell them apart (e.g. the rule returns `[MyArray(x.array + 100.0)]`). Tests:

```python
def test_rule_replaces_body_eager(): ...          # quaxify(f1)(MyArray(1.)) -> MyArray, array == 101.
def test_rule_replaces_body_dict_output(): ...    # cjvp returns {"y": ...}; rule returns [leaf]; result["y"].array == rule value
                                                  # (the eager StoreException case)
def test_rule_quaxify_of_jit(): ...               # quax.quaxify(jax.jit(f1)) -> 101.
def test_rule_jit_of_quaxify(): ...               # jax.jit(quax.quaxify(f1)) -> 101.
def test_rule_vmap(): ...                         # jax.vmap(quaxify(f1))(MyArray(jnp.array([1., 2.]))).array == [101., 102.]
def test_rule_outer_grad(): ...                   # rule returns [MyArray(jnp.sin(x.array) * 3.)];
                                                  # jax.grad(lambda a: quaxify(f1b)(MyArray(a)).array)(0.5) == 3*cos(0.5)
def test_rule_with_python_scalar_arg(): ...       # two-arg cjvp, rule typed (MyArray, ArrayLike) — quax hands the scalar over as a jax.Array; quaxify(f2)(MyArray(1.), 2.0) -> rule value
def test_type_miss_falls_back_to_inline(): ...    # rule typed (MyArray, MyArray); call with (MyArray, jnp.array(3.)) -> body result
def test_unregistered_custom_jvp_inlines(): ...   # fresh cjvp, no rule -> body result (sin)
def test_nondiff_argnums_not_passed(): ...        # @partial(custom_jvp, nondiff_argnums=(1,)); rule takes exactly one arg; call f(MyArray(1.), 3)
def test_precedence_picks_rule(): ...             # two rules, same signature, precedence=1 one wins
def test_wrong_output_count_raises(): ...         # rule returns 2 leaves for 1-output cjvp -> pytest.raises(ValueError, match="expected 1")
def test_bare_value_return_raises(): ...          # rule returns MyArray(...) not [..] -> pytest.raises(TypeError)
def test_register_rejects_custom_vjp(): ...       # quax.register(some_custom_vjp) -> TypeError at decoration
def test_register_rejects_other_objects(): ...    # quax.register(lambda x: x) -> TypeError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_register_custom_jvp.py -v`
Expected: FAIL (`register` rejects/misuses a `custom_jvp`; rules not consulted)

- [ ] **Step 3: Implement `register` for `custom_jvp` in `src/quax/_dispatch.py`**

- `isinstance(target, jexc.Primitive)` → today's path. `isinstance(target, jax.custom_jvp)` → key `fun_sourceinfo(target.fun)`, and raise `ValueError("cannot register <name>: JAX has no source location for it")` if that is falsy. (This guard is effectively unreachable: for a builtin, `fun_sourceinfo` returns the bare name, e.g. `'neg'`. It gets no test.) Anything else (including `jax.custom_vjp`) → `TypeError` naming the accepted types.
- Name the plum dispatcher `f"{target.__name__}_dispatcher"` for a `custom_jvp`.
- Extend the docstring: the `custom_jvp` form, the flattened-leaves input/output contract, `nondiff_argnums` not passed, the rule replaces the derivative too (write rules with JAX ops on arrays), inner `jvp`/`grad` not intercepted, factory-made `custom_jvp`s share a rule.

- [ ] **Step 4: Implement `_custom_jvp_rule` in `src/quax/_trace.py`**

Order, per the spec:
1. `rule = _rules.get(getattr(fun.debug_info, "func_src_info", None))`; `None` → return `None`.
2. Values as in `process_primitive` (unwrap `_DenseArrayValue` to `.array`); `rule.resolve_method(values)`; `plum.NotFoundLookupError` → `None`.
3. `expected = jax.eval_shape(lambda *a: fun.call_wrapped(*a), *[jax.ShapeDtypeStruct(av.shape, av.dtype) for av in avals])`, where `av = v.aval()` for a Value (`_is_value`), else `typeof(v)`.
4. `with core.set_current_trace(self.parent_trace): out = method(*values)`.
5. `out` not a `list`/`tuple` → `TypeError`; `len(out) != len(expected)` → `ValueError(f"quax rule for custom_jvp {name!r} returned {len(out)} outputs; expected {len(expected)}")`, with `name = fun.debug_info.func_name`.
6. Return `[_QuaxTracer(self, _wrap_if_array(o)) for o in out]`.

At the top of both `process_custom_jvp_call` definitions: `if (out := self._custom_jvp_rule(fun, tracers)) is not None: return out`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_register_custom_jvp.py -v` → all PASS.
Run: `uv run --with 'jax[cpu]==0.12.0' pytest tests/unit/test_register_custom_jvp.py -q` → all PASS.
Run: `uv run --with 'jax[cpu]==0.7.2' pytest tests/unit/test_register_custom_jvp.py -q` → all PASS (JAX 0.7.2).
Run: `uv run pytest tests/unit -q` → no new failures.

- [ ] **Step 6: Lint and commit**

```bash
uv run prek run --all-files
git add src/quax/_compat.py src/quax/_dispatch.py src/quax/_trace.py tests/unit/test_register_custom_jvp.py
git commit -m "✨ feat: quax.register accepts jax.custom_jvp functions"
```

---

### Task 2: `hypot` rule for the `Unitful` example

**Files:**
- Modify: `src/quax/examples/unitful/_core.py` (append the rule)
- Modify: `tests/usage/test_unitful.py`

**Interfaces:**
- Consumes: `quax.register(custom_jvp)` from Task 1.
- Produces: `hypot_unitful_unitful(x1: Unitful, x2: Unitful) -> list[Unitful]`, registered only when `getattr(jax._src.numpy.ufuncs, "_hypot", None)` is a `jax.custom_jvp`.

- [ ] **Step 1: Write the failing tests** in `tests/usage/test_unitful.py`, all skipped unless `ufuncs` has `_hypot`:

```python
def test_hypot_keeps_units(): ...        # quaxify(jnp.hypot)(3 m, 4 m) -> Unitful, units {meters: 1}, array == 5.
def test_hypot_under_jit(): ...          # jax.jit(quaxify(jnp.hypot)) and quaxify(jax.jit(jnp.hypot)) -> 5.
def test_hypot_outer_grad(): ...         # d/da hypot(a m, 4 m).array at a=3 == 0.6
def test_hypot_units_mismatch(): ...     # (3 m, 4 s) -> pytest.raises(ValueError, match="units") both eager and under jax.jit
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with 'jax[cpu]==0.12.0' pytest tests/usage/test_unitful.py -k hypot -v`
Expected: FAIL with "Refusing to materialise Unitful array."

- [ ] **Step 3: Implement `hypot_unitful_unitful`** in `_core.py`

If the units differ, raise `ValueError(f"Cannot take hypot of arrays with units {x1.units} and {x2.units}.")`. Otherwise `h = jnp.hypot(x1.array, x2.array)` and return `[Unitful(h, x1.units), Unitful(x1.array / h, {}), Unitful(x2.array / h, {})]`, matching `_hypot`'s three outputs `(hypot, x1/h, x2/h)`. Add a comment giving the reason for the version gate (JAX < 0.12 has no `_hypot`).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --with 'jax[cpu]==0.12.0' pytest tests/usage/test_unitful.py -v` → PASS.
Run: `uv run pytest tests/usage/test_unitful.py -v` → hypot tests SKIPPED, others PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quax/examples/unitful/_core.py tests/usage/test_unitful.py
git commit -m "✨ feat(unitful): hypot rule via custom_jvp registration"
```

---

### Task 3: User-facing docs

**Files:**
- Modify: `skills/quax/SKILL.md` (new `### Overriding a custom_jvp` under `## Writing dispatch rules`, ~L178)
- Modify: `docs/autodiff.md` (`## Custom derivative rules`, ~L63)
- Modify: `AGENTS.md` (Key Pitfalls; Doctests bullet)

**Interfaces:**
- Consumes: the `register` contract from Task 1. Both Markdown files run as tests under Sybil (`conftest.py`), so every `python` block must run on the locked JAX 0.10.2. Demonstrate with a user-defined `custom_jvp`, not `_hypot`, and mention `_hypot` only in prose.

- [ ] **Step 1: Write the SKILL.md section**: when to use it (the body can't be expressed with your type's rules), a runnable example (rule returning a list), the flattened-leaves contract, derivative replacement, the inner-`jvp` limitation, version-gating private targets with `getattr`.
- [ ] **Step 2: Add the autodiff.md note**: two or three sentences plus a link to the SKILL section. Under an outer `grad` the rule replaces the derivative, and inner `jvp` is not intercepted.
- [ ] **Step 3: AGENTS.md.** Add a pitfall: `custom_jvp` rules are keyed by JAX's private `func_src_info` string, imported via `_compat.fun_sourceinfo`. Fix the Doctests bullet to name the `docs/` pages that `conftest.py` also runs under Sybil.
- [ ] **Step 4: Run the doc tests**

Run: `uv run pytest skills/quax/SKILL.md docs/autodiff.md -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/quax/SKILL.md docs/autodiff.md AGENTS.md
git commit -m "📝 docs: overriding custom_jvp functions with quax.register"
```

---

### Task 4: Full verification

- [ ] Run `uv run pytest -q` → PASS.
- [ ] Run `uv run --with 'jax[cpu]==0.12.0' pytest -q` → PASS.
- [ ] Run `uv run --with 'jax[cpu]==0.7.2' pytest tests/unit tests/usage -q` → PASS.
- [ ] Run `uv run prek run --all-files` → clean.
