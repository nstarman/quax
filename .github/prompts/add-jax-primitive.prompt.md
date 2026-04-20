---
description: "Register a new JAX primitive handler for an existing ArrayValue type. Use when: adding dispatch rules, implementing lax.* or jnp.* support for a custom array type, resolving plum ambiguity errors."
argument-hint: "primitive and ArrayValue type, e.g. 'lax.sin_p for MyArray'"
agent: "agent"
---

Register a new JAX primitive handler for the `$input` operation.

## Steps

1. **Identify the primitive** — Find the `jax.lax.*_p` primitive that corresponds to the operation. When unsure, check what primitive JAX uses by running `jax.make_jaxpr(jnp.op)(array)` or searching [src/quax/\_core.py](../../src/quax/_core.py) and the JAX source.

2. **Determine the type signatures needed** — Consider all combinations of the target `ArrayValue` type with `ArrayLike | quax.ArrayValue` (for mixed-type operations). Common patterns from the codebase:
   - Unary: `(x: MyType) -> MyType`
   - Binary same-type: `(x: MyType, y: MyType) -> MyType`
   - Binary mixed (two rules): `(x: MyType, y: ArrayLike | quax.ArrayValue)` and `(x: ArrayLike | quax.ArrayValue, y: MyType)`

3. **Write the handler(s)** — Place in the same file as the `ArrayValue` definition (e.g. `src/quax/examples/<name>/_core.py`). Follow this pattern:

   ```python
   @quax.register(lax.<primitive>_p)
   def _(x: MyType, ...) -> MyType:
       # implement using x.array, x.shape, x.dtype, etc.
       ...
   ```

   - Keep keyword-only params as `**kw` if the primitive may pass extra args (e.g. `out_dtype` for `mul_p`)
   - Forward unknown kwargs to the underlying JAX op when doing a materialised fallback

4. **Resolve plum ambiguity** — If two rules could both match (e.g. `(MyType, ArrayLike)` and `(ArrayLike, MyType)` overlap when both args are `MyType`), add `precedence=1` to the more specific rule:

   ```python
   @quax.register(lax.<primitive>_p, precedence=1)
   def _(x: MyType, y: MyType) -> MyType:
       ...
   ```

5. **Add tests** — Add a parametrized entry to the relevant test file:
   - For `lax` primitives: `tests/unit/test_lax/test_myarray.py`
   - For `jnp` functions: `tests/unit/test_numpy/test_myarray.py`
   - For a specific `quax.examples` type: `tests/usage/test_<name>.py`

   Test entries use the `(func_name, args, kw, expect_myarray)` tuple format. See [tests/unit/myarray.py](../../tests/unit/myarray.py) for the `MyArray` fixture.

6. **Verify** — Run:
   ```bash
   uv run pytest tests/unit/ -k "<primitive_name>"
   uv run pre-commit run -a
   ```

## Reference examples

- Simple unary/binary rules: [src/quax/examples/zero/\_core.py](../../src/quax/examples/zero/_core.py)
- Advanced rules with kwargs and precedence: [src/quax/examples/lora/\_core.py](../../src/quax/examples/lora/_core.py)
- Mixed-type rules: search for `ArrayLike | quax.ArrayValue` in `src/quax/examples/`
