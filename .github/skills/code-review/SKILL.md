---
name: code-review
description: Use when reviewing a pull request or diff in the quax repository. Covers the quax-specific defects that generic review misses — dispatch rules that disagree with the primitive they replace, JAX version gates set at the wrong boundary, Value subclass invariants, trace hot-path regressions, and error paths that were quietly made quieter.
---

# Reviewing quax changes

Quax runs a JAX program under a custom interpreter, reinterpreting each primitive
according to the types passed in. Two properties follow, and they generate nearly
every real defect in this repository:

- **A dispatch rule replaces a JAX primitive.** If it disagrees with that
  primitive about a value, a shape, or a dtype, the disagreement surfaces as a
  wrong number deep inside somebody else's model.
- **A missing rule is not an error.** Dispatch falls back to materialising every
  operand and calling plain JAX, so the custom type vanishes from the output
  without a warning. Silence is the default failure mode here.

## Scope of this review

Leave these alone — they are already gated or already covered:

- Formatting, import order, naming, line length. `prek` runs ruff (`E`, `F`,
  `I001`, `UP`), pyright, and taplo on every commit.
- Generic security checklists. There is no user input, no network, no
  serialisation of untrusted data, no rendering. Injection and XSS questions do
  not apply.
- Value-level correctness of a primitive that already has a test entry. The
  harness compares against plain JAX for you — see [Tests](#tests).

Spend the review on the sections below instead.

## What changed → what to check

| Change | Check |
|---|---|
| A `@quax.register` rule added or edited | [Dispatch rules](#dispatch-rules) |
| A `Value` / `ArrayValue` subclass, or its fields | [Value subclasses](#value-subclasses) |
| `_compat.py`, or any `JAX_GE_*` / `jax._src` use | [Version compatibility](#version-compatibility) |
| `_trace.py`, `_dispatch.py`, `_module.py`, `_values.py` | [The hot path](#the-hot-path) |
| An `except`, `assert`, or fallback branch | [Silent failure](#silent-failure) |
| Anything under `tests/` | [Tests](#tests) |

## Dispatch rules

The question that finds the most bugs: **does this rule agree with the primitive
it replaces?** Not approximately — in value, in output shape, and in dtype. Read
the rule against `lax`'s own semantics for that primitive rather than against
what the rule looks like it intends.

Three landed bugs, all of this shape:

- `Unitful.integer_pow` raised the *units* to the power but returned the array
  unchanged (#180). The array and its metadata must both transform.
- `Zero`'s `strided_slice` output shape used floor where `lax` uses ceil (#181).
  Shape arithmetic must reproduce `lax`'s rounding exactly.
- `named` validated axes positionally where the semantics are pairwise
  (#192, #194).

Then the mechanical checks:

- **`**kw` accepted and forwarded.** Primitive params come and go across JAX
  versions (`out_dtype` on `mul_p`, `out_sharding` on `reduce_sum_p`,
  `sharding` on `broadcast_in_dim_p`). A rigid signature is a future
  `TypeError: got an unexpected keyword argument` on somebody's upgrade.
- **Mixed-type rules come in pairs**, annotating the operand you do not own as
  `ArrayLike | quax.ArrayValue`, with `precedence=1` on the specific
  `(MyType, MyType)` rule. Without the precedence, both rules match a same-type
  call and plum raises `AmbiguousLookupError`.
- **The return annotation is load-bearing.** plum runs `convert` on the return
  value of any rule with a concrete return annotation, and it is what other
  rules dispatch against. An annotation that overstates the type (claiming
  `MyType` for a comparison that honestly returns bools) is a defect, not a
  cosmetic issue.
- **Does the custom type survive?** If the rule materialises an operand and
  returns a plain array, that is sometimes the honest answer and sometimes the
  type silently disappearing. Decide which, explicitly.

A new rule should also make its way into
[.github/prompts/add-jax-primitive.prompt.md](../../prompts/add-jax-primitive.prompt.md)'s
workflow — that prompt is the authoring counterpart to this section.

## Value subclasses

- **`eqx.field(static=True)` on every non-array field** — stored shapes, dtypes,
  units, axis names, bool flags. Omitting it makes JAX try to trace through the
  value.
- **`aval()` must be pure**: the same instance returns the same `AbstractValue`
  every time, because quax caches it at tracer-construction time. It *may* bind
  primitives (`lora.LoraArray.aval` calls `lax.stop_gradient`); quax evaluates it
  under the parent trace so that this works (#216). A change touching where or
  when `aval()` is called needs care here.
- **A `materialise()` that raises is a design decision, not a bug.** `Unitful`,
  `LoraArray`, `NamedArray`, and the `MyArray` test fixture all raise on purpose,
  because materialising would discard the information the type exists to carry.
  Do not "fix" one.
- **`__array__` must not silently strip.** Returning a bare array from a type
  carrying a unit or an axis name turns a loud failure into a wrong number.
  `jax.numpy.asarray` began calling it in JAX 0.10 where it previously raised.

## Version compatibility

This library supports a JAX range, and the compatibility layer is where version
work goes wrong.

- **Gate on the version where the API actually changed**, not on the nearest
  flag that already exists. #166 guarded `TypedInt` conversions on 0.7.2 when the
  types landed in 0.8.0. If a PR adds a `JAX_GE_*` use, confirm the boundary
  against JAX's own history rather than against the surrounding code.
- **Never reintroduce a removed parameter.** #205 re-added `scan_p`'s `linear`
  after JAX 0.10.2 dropped it. When rebuilding primitive params, pass through
  what the caller gave you; construct only what the current version takes.
- **Both branches must be exercised.** A new gate means two code paths; CI runs
  the supported floor and the newest release, so say which branch the new tests
  hit.
- **`jax._src` reaches need a gate and a comment** explaining what changed and
  why the private API is the only route. `_compat.py` is the model for this —
  match its density of explanation, not its brevity.

## The hot path

`_trace.py`, `_dispatch.py`, `_module.py`, and `Value` construction run **once
per primitive per call**. Cost added here is multiplied by the size of the user's
program, and this repo has spent four PRs (#168, #173, #174, #187) recovering it.

- Per-call object construction, `isinstance` against an ABC, or equinox
  per-instance validation in these files is a regression until benchmarked
  otherwise.
- **Caches must not pin what they watch.** #187 was a weakref-keyed cache holding
  a strong reference to the jaxpr it was tracking, so nothing was ever collected.
- CodSpeed benchmarks live in `tests/benchmark/`. A change plausibly affecting
  dispatch cost should say what happened to them.
- `plum`'s resolution cache is disabled entirely by a single non-faithful type,
  which is why `_compat.py` sets `jax.Array.__faithful__`. Changes near dispatch
  registration can switch that cache off without any visible symptom but a slow
  benchmark.

## Silent failure

Rule 3 of dispatch already degrades silently by design. Anything that makes an
error path *quieter* compounds it and should be treated as a defect rather than a
cleanup:

- `assert False` for an unreachable branch — assertions vanish under `-O`, and
  this was replaced with a real `TypeError` in #171.
- A bare `except`, or a fallback that swallows the reason it was reached.
- Broadening a rule's annotations so that a previously-unmatched call now
  materialises instead of raising.

## Tests

The harness in `tests/unit/test_lax/` and `tests/unit/test_numpy/` runs each
entry through both plain JAX and `quax.quaxify`, then asserts the results are
equal. **Value correctness for a covered primitive is therefore already
checked.** What it cannot check, and what review must:

- **`expect_myarray` is the real assertion.** It says whether the custom type
  survives the operation, per output. A `True` that should be `False` (or a
  tuple with the wrong arity) is a coverage hole the suite will happily pass.
- **`mark_todo` is `pytest.mark.skip`.** A PR that implements a primitive must
  remove the mark on its entry, or the new rule ships untested.
- **Both files.** `test_myarray.py` and `test_jax_array.py` are separate; a rule
  affecting the plain-array path needs an entry in each.
- **Metadata semantics are invisible to `MyArray`**, which carries no units or
  axis names. Bugs like #180 live in `tests/usage/test_<name>.py` — a rule on an
  example type needs coverage there, not only in the unit tests.
- The suite runs with `JAX_CHECK_TRACER_LEAKS=1` and jaxtyping/beartype enabled.

## Repo conventions

- **`uv run` for everything** — `uv run pytest`, `uv run prek run --all-files`.
  Never bare `python` or `pytest`.
- Commits use gitmoji plus conventional commits: `🐛 fix(trace): ...`,
  `✅ test: ...`, `⚡️ perf: ...`, `build(deps): ...`.
- Pre-commit runs pyright on `src/` only, so type errors in `tests/` reach CI
  unreviewed.
- **Doctests are not collected.** Examples in `README.md`, `docs/`, and `src/`
  docstrings never run — a PR changing one has not tested it, and the diff needs
  reading on that basis. The exception is
  [skills/quax/SKILL.md](../../../skills/quax/SKILL.md), whose `python` blocks
  are executed by
  [tests/test_skill_examples.py](../../../tests/test_skill_examples.py).

## Further reading

- [skills/quax/SKILL.md](../../../skills/quax/SKILL.md) — the dispatch resolution
  ladder, writing rules, boundary behaviour, and a troubleshooting table.
- [AGENTS.md](../../../AGENTS.md) — module-by-module architecture and commands.
- [src/quax/examples/zero/_core.py](../../../src/quax/examples/zero/_core.py) —
  minimal reference type;
  [lora/_core.py](../../../src/quax/examples/lora/_core.py) — advanced rules.
