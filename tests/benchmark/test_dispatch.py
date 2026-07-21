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
from quax.examples import lora, zero

from ..unit.myarray import MyArray


_key = jax.random.PRNGKey(0)

_xm = MyArray(jnp.arange(8.0) + 1)
_ym = MyArray(jnp.arange(8.0) + 2)

_z = zero.Zero((8,), jnp.float32)
_arr = jnp.arange(8.0)

# LoRA: a low-rank array times a dense matrix (heavy dispatch — the matmul
# expands to several primitives, several of which are inlined pjits).
_lora = lora.LoraArray(jax.random.normal(_key, (32, 16)), rank=4, key=_key)
_rhs = jax.random.normal(_key, (16, 8))
_dot_dn = (((1,), (0,)), ((), ()))


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
