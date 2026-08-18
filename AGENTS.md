# Quax — Agent Instructions

Quax provides multiple dispatch in JAX via custom interpreters, enabling custom array-ish objects to work transparently with existing JAX programs.

## Essential Commands

```bash
uv run pytest                # run all tests (doctests included)
uv run pytest tests/unit/    # unit tests only
uv run pytest tests/usage/   # integration/usage tests
uv run mkdocs serve          # build and serve docs locally
uv run prek run --all-files     # lint + format (ruff, pyright, taplo)
```

> Always use `uv run` — never bare `python` or `pytest`.

## Architecture

| Module | Role |
|--------|------|
| [_values.py](src/quax/_values.py) | `Value` (base `eqx.Module`; requires `aval()` and `materialise()`), `ArrayValue` (exposes `.shape`, `.dtype`, `.ndim`, `.size`), internal `_DenseArrayValue` |
| [_quaxify.py](src/quax/_quaxify.py) | `quaxify(fn)` and `_Quaxify`; wraps any JAX function to enable quax dispatch |
| [_dispatch.py](src/quax/_dispatch.py) | `register(primitive)`, the plum rule table, and the dispatch cache |
| [_primitives.py](src/quax/_primitives.py) | Handlers for `jit_p`, `while_p`, `cond_p`, `scan_p`, and their jaxpr caches |
| [_trace.py](src/quax/_trace.py) | `_QuaxTrace`/`_QuaxTracer`; the interpreter that dispatches via plum |
| [_module.py](src/quax/_module.py) | `_FastModuleMeta`; skips equinox per-instance validation on hot paths |
| [_compat.py](src/quax/_compat.py) | Version flags (`JAX_GE_0_9_2`, etc.) for JAX API differences |

## Using and extending quax

The user-facing material — the dispatch resolution ladder, creating a custom
`ArrayValue`, writing and disambiguating rules, boundary behaviour, and
troubleshooting — lives in [skills/quax/SKILL.md](skills/quax/SKILL.md). It is
the single source of truth for that; do not duplicate it here.

Reference implementations: [src/quax/examples/zero/_core.py](src/quax/examples/zero/_core.py)
(minimal) and [src/quax/examples/lora/_core.py](src/quax/examples/lora/_core.py) (advanced).

## Testing Patterns

| Path | Purpose |
|------|---------|
| [tests/unit/myarray.py](tests/unit/myarray.py) | Shared `MyArray(ArrayValue)` fixture with registered primitives |
| [tests/conftest.py](tests/conftest.py) | `getkey` fixture via `eqxi.GetKey()` |
| `tests/unit/test_numpy/` | Parametrized `jnp.*` tests (separate files for `MyArray` vs plain JAX arrays) |
| `tests/unit/test_lax/` | Same for `lax` primitives |
| `tests/usage/` | Integration tests for each `quax.examples` type |

Tests use `(func_name, args, kw, expect_myarray)` parameter tuples. Common marks: `xfail_quax58`, `mark_todo`, `mark_nomd`.

`pytest` runs with `JAX_CHECK_TRACER_LEAKS=1` and jaxtyping/beartype enabled — type errors surface at test time.

## Key Pitfalls

- **`eqx.field(static=True)` is mandatory** for shapes, dtypes, and bool flags — forget it and JAX will attempt to trace through them.
- **`materialise()` may intentionally raise** — `MyArray.materialise()` (test fixture) and `LoraArray.materialise()` raise on purpose; do not "fix" them.
- **Name every registered rule** — `@quax.register` handlers are named after the primitive and the
  types they dispatch on (`add_unitful_unitful`, `convert_element_type_zero`, `cond_quax`), never
  `def _`. Dispatch ignores the name; tracebacks, plum's ambiguity errors, and profiles do not.
- **`_DenseArrayValue` is internal** — never instantiate or reference it from user code.
- **`_compat.py` version gates** — use `typeof` from `_compat` (not `jax.core.get_aval` directly); `_primitives.py` and `_compat.py` carry dual branches for JAX API differences across versions.
- **Tests import across modules** — e.g. `from ..myarray import MyArray`; keep internal test imports relative.
- **Pre-commit Pyright runs only on `src/`** — the pre-commit hook excludes `tests/`, even though `[tool.pyright]` includes it.
- **Doctests are not collected.** `testpaths` lists `docs`, but no `--doctest-glob`/`--doctest-modules` is set, so examples in `docs/` and `src/` docstrings are never run. Verify them by hand when you change them. The exceptions are [README.md](README.md) and [skills/quax/SKILL.md](skills/quax/SKILL.md), whose `python` blocks are executed by [tests/test_readme_examples.py](tests/test_readme_examples.py) and [tests/test_skill_examples.py](tests/test_skill_examples.py).

## Dependencies

Read [pyproject.toml](pyproject.toml) for the full list. Key dependencies include:

- `jax` (core library)
- `equinox` (base class and utilities)
- `plum` (multiple dispatch)
- `pytest` (testing)
- `mkdocs` (documentation)

## Further Reading

- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) — what to look for when reviewing a quax change (also picked up by GitHub Copilot code review)
- [CONTRIBUTING.md](CONTRIBUTING.md) — setup and workflow
- [src/quax/examples/README.md](src/quax/examples/README.md) — guide for built-in examples
- [docs/](docs/) — full documentation (also served at <https://nstarman.github.io/quax>)
