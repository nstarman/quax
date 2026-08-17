"""Eager (non-jit) ``quaxify`` operation benchmarks.

These exercise the multiple-dispatch hot path — tracer wrapping, primitive
processing, rule resolution, ``Value`` construction, and the inlined-``pjit``
handling — which is what the metaclass / ABC-isinstance / inline-``pjit``
optimisations target. Unlike the ``test_construction`` micro-benchmarks (pure
allocation) these record end-to-end per-primitive dispatch cost, so a regression
(or improvement) in that machinery shows up here.

Small operands are used deliberately: the goal is to measure quax's Python
overhead per primitive, not XLA throughput (pure-JAX eager ``a + b`` is ~6 µs vs
quax's ~150 µs, so the overhead dominates at this size).
"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

import quax
from quax.examples import lora, unitful, zero

from ..unit.myarray import MyArray


_key = jax.random.PRNGKey(0)

_xm = MyArray(jnp.arange(8.0) + 1)
_ym = MyArray(jnp.arange(8.0) + 2)

_z = zero.Zero((8,), jnp.float32)
_arr = jnp.arange(8.0)

# A `Value` whose rules do non-trivial Python work per primitive (dict compare /
# merge of the units), rather than just rewrapping arrays.
_u = unitful.Unitful(jnp.arange(8.0) + 1, unitful.meters)

# LoRA: a low-rank array times a dense matrix (heavy dispatch — the matmul
# expands to several primitives, several of which are inlined pjits).
_lora = lora.LoraArray(jax.random.normal(_key, (32, 16)), rank=4, key=_key)
_rhs = jax.random.normal(_key, (16, 8))
_dot_dn = (((1,), (0,)), ((), ()))

# The same LoRA array, but materialisable: `sin` has no LoRA rule, so it falls
# through to `_default_process` -> `Value.default` -> `materialise`.
_lora_mat = lora.LoraArray(
    jax.random.normal(_key, (32, 16)), rank=4, key=_key, allow_materialise=True
)


# An inner `@jax.jit` that quax does *not* inline, so its `pjit` takes the cached
# `jit_quax` path (`_jit_quax_cache` — a jax.jit-wrapped quaxify callable whose
# compiled kernel is reused), distinct from the inlined-`pjit` path the other
# dispatch benchmarks hit.
@jax.jit
def _inner_jit(a):
    return a * 2.0 + 1.0


def _calls_inner_jit(a):
    return _inner_jit(a)


def _bench(benchmark, fn, *args):
    """Warm the dispatch cache, then benchmark eager ``quaxify(fn)(*args)``."""
    qfn = quax.quaxify(fn)
    qfn(*args)  # warm plum resolution + the dispatch cache
    benchmark(lambda: qfn(*args))


@pytest.mark.benchmark(group="dispatch")
def test_add_myarray(benchmark):
    """`quaxify(add)(MyArray, MyArray)` — the core single-primitive dispatch."""
    _bench(benchmark, lambda a, b: a + b, _xm, _ym)


@pytest.mark.benchmark(group="dispatch")
def test_mul_myarray(benchmark):
    """`quaxify(mul)(MyArray, MyArray)`."""
    _bench(benchmark, lambda a, b: a * b, _xm, _ym)


@pytest.mark.benchmark(group="dispatch")
def test_add_zero(benchmark):
    """`quaxify(add)(Zero, array)` — the symbolic-zero fast rule."""
    _bench(benchmark, lambda a, b: a + b, _z, _arr)


@pytest.mark.benchmark(group="dispatch")
def test_mul_unitful(benchmark):
    """`quaxify(mul)(Unitful, Unitful)` — a rule that does real Python work
    (merging the unit dicts) on top of the array op."""
    _bench(benchmark, lambda a, b: a * b, _u, _u)


@pytest.mark.benchmark(group="dispatch")
def test_materialise_fallback(benchmark):
    """`quaxify(sin)(LoraArray)` — no rule for this (primitive, type), so the
    cached `_DISPATCH_MISS` sends it to `_default_process`, which materialises
    every `Value` operand and binds the primitive on plain arrays."""
    _bench(benchmark, jnp.sin, _lora_mat)


@pytest.mark.benchmark(group="dispatch")
def test_nested_quaxify_add(benchmark):
    """Nested `quaxify(quaxify(add))(MyArray)` — two dispatch layers."""
    outer = quax.quaxify(quax.quaxify(lambda a, b: a + b))
    outer(_xm, _ym)  # warm both dispatch layers (plum resolution + cache)
    benchmark(lambda: outer(_xm, _ym))


@pytest.mark.benchmark(group="dispatch")
def test_matmul_lora(benchmark):
    """`quaxify(dot_general)(LoraArray, array)` — LoRA matmul (many primitives,
    inlined pjits)."""
    _bench(benchmark, lax.dot_general, _lora, _rhs, _dot_dn)


@pytest.mark.benchmark(group="dispatch")
def test_jit_cache_hit(benchmark):
    """`quaxify` over a function calling an inner `@jax.jit` — the cached
    (non-inlined) `jit_quax` path, hitting `_jit_quax_cache`."""
    _bench(benchmark, _calls_inner_jit, _xm)
