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

Core implementation: [src/quax/_core.py](src/quax/_core.py)

| Symbol | Role |
|--------|------|
| `Value` | Base `eqx.Module`; requires `aval()` and `materialise()` |
| `ArrayValue(Value)` | Array-like base class; exposes `.shape`, `.dtype`, `.ndim` |
| `quaxify(fn)` | Wraps any JAX function to enable quax dispatch |
| `register(primitive)` | Decorator to add a plum multiple-dispatch rule for a JAX primitive |
| `_QuaxTrace` | Internal JAX interpreter; dispatches via plum |
| `_compat.py` | Version flags (`JAX_GE_0_9_2`, etc.) for JAX API differences |

Dispatch resolution for a primitive:

1. Registered rule matches via plum → use it
2. Exactly one arg overrides `Value.default` → use that
3. Zero overrides → materialise all args, call normal JAX
4. Multiple overrides → runtime error

## Creating a Custom `ArrayValue`

See [src/quax/examples/zero/_core.py](src/quax/examples/zero/_core.py) (minimal) and [src/quax/examples/lora/_core.py](src/quax/examples/lora/_core.py) (advanced).

```python
class MyType(quax.ArrayValue):
    array: jax.Array
    _shape: tuple[int, ...] = eqx.field(static=True)  # shapes/flags MUST be static

    def aval(self) -> jax.core.ShapedArray:      # REQUIRED
        return jax.core.ShapedArray(self._shape, jnp.float32)

    def materialise(self) -> jax.Array:           # REQUIRED
        return self.array                         # raise RuntimeError to forbid fallback

@quax.register(jax.lax.add_p)
def _(x: MyType, y: MyType) -> MyType:
    assert x._shape == y._shape
    array = x.array + y.array
    return MyType(array, _shape=x._shape)
```

Use `@quax.register(prim, precedence=1)` to resolve plum ambiguity between overlapping rules.

## Testing Patterns

| Path | Purpose |
|------|---------|
| [tests/unit/myarray.py](tests/unit/myarray.py) | Shared `MyArray(ArrayValue)` fixture with registered primitives |
| [tests/conftest.py](tests/conftest.py) | Installs the `beartype.claw` runtime-typecheck hook; `getkey` fixture via `eqxi.GetKey()` |
| `tests/unit/test_numpy/` | Parametrized `jnp.*` tests (separate files for `MyArray` vs plain JAX arrays) |
| `tests/unit/test_lax/` | Same for `lax` primitives |
| `tests/usage/` | Integration tests for each `quax.examples` type |

Tests use `(func_name, args, kw, expect_myarray)` parameter tuples. Common marks: `xfail_quax58`, `mark_todo`, `mark_nomd`.

`pytest` runs with `JAX_CHECK_TRACER_LEAKS=1` and bearshape/beartype (via `beartype.claw`) enabled — type errors surface at test time.

## Key Pitfalls

- **`eqx.field(static=True)` is mandatory** for shapes, dtypes, and bool flags — forget it and JAX will attempt to trace through them.
- **`materialise()` may intentionally raise** — `MyArray.materialise()` (test fixture) and `LoraArray.materialise()` raise on purpose; do not "fix" them.
- **`_DenseArrayValue` is internal** — never instantiate or reference it from user code.
- **`_compat.py` version gates** — use `typeof` from `_compat` (not `jax.core.get_aval` directly); `_core.py` has dual branches for JAX API differences across versions.
- **Tests import across modules** — e.g. `from ..myarray import MyArray`; keep internal test imports relative.
- **Pre-commit Pyright runs only on `src/`** — the pre-commit hook excludes `tests/`, even though `[tool.pyright]` includes it.
- **Doctests run** from `README.md`, `docs/`, and `src/` — keep examples in those files valid.
- **Never annotate a closure in `_primitives.py` that gets re-defined on every call** (e.g. `qfun` inside `jit_quax`, `_make_quax_branch`/`flat_quax_call` inside `cond_quax`) — `beartype.claw` decorates any annotated function it sees, and beartype's own `is_object_blacklisted` cache (`@callable_cached`, unbounded, never evicted) permanently pins that specific closure object — and everything it closes over, including live JAX tracers — for the rest of the process. `@no_type_check` alone does NOT prevent this (claw decides whether to decorate purely from annotation presence, before `@no_type_check` is ever consulted) — the fix is to strip the closure's parameter/return annotations entirely, not just mark it `@no_type_check`. Symptom: `JAX_CHECK_TRACER_LEAKS=1` failures with "Leaked trace" errors that trace back into beartype's internals.

## Dependencies

Read [pyproject.toml](pyproject.toml) for the full list. Key dependencies include:

- `jax` (core library)
- `equinox` (base class and utilities)
- `plum` (multiple dispatch)
- `pytest` (testing)
- `mkdocs` (documentation)

## Further Reading

- [CONTRIBUTING.md](CONTRIBUTING.md) — setup and workflow
- [src/quax/examples/README.md](src/quax/examples/README.md) — guide for built-in examples
- [docs/](docs/) — full documentation (also served at <https://nstarman.github.io/quax>)
