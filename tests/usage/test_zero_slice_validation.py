import functools as ft

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.zero import Zero


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "start,limit,strides",
    [
        ((0,), (5,), None),
        ((-1,), (2,), None),
        ((1,), (0,), None),
        ((0, 0), (1, 1), None),
    ],
)
def test_invalid_slice_matches_jax(jit, start, limit, strides):
    fn = ft.partial(
        lax.slice, start_indices=start, limit_indices=limit, strides=strides
    )
    dense_fn = jax.jit(fn) if jit else fn
    symbolic_fn = quax.quaxify(fn)
    if jit:
        symbolic_fn = jax.jit(symbolic_fn)
    # JAX 0.7.2 reports eager rank mismatches as AssertionError.
    with pytest.raises((AssertionError, TypeError, ValueError)):
        dense_fn(jnp.zeros(2))
    with pytest.raises((TypeError, ValueError)):
        symbolic_fn(Zero((2,), jnp.float32))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "start,limit,strides", [((0,), (5,), (2,)), ((2,), (2,), None), ((0,), (5,), None)]
)
def test_valid_slice_preserves_symbolic_zero(jit, start, limit, strides, monkeypatch):
    fn = ft.partial(
        lax.slice, start_indices=start, limit_indices=limit, strides=strides
    )
    expected = fn(jnp.zeros(5, dtype=jnp.int32))
    symbolic_fn = quax.quaxify(fn)
    if jit:
        symbolic_fn = jax.jit(symbolic_fn)

    def refuse_materialisation(self):
        raise AssertionError("Slicing must not allocate a dense operand")

    monkeypatch.setattr(Zero, "materialise", refuse_materialisation)
    result = symbolic_fn(Zero((5,), jnp.int32))
    assert isinstance(result, Zero)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
