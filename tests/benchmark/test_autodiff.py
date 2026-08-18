"""Benchmarks for the autodiff path through a quax trace.

``_QuaxTrace.process_custom_jvp_call`` and ``process_custom_vjp_call``, with
their ``lu.transformation`` wrappers, are the ``process_*`` overrides besides
``process_primitive``. They do noticeably more per call than plain dispatch --
flatten/unflatten of every `Value` through a treedef, the `SymbolicZero`
promotion scan, and the primal/tangent `materialise` fallback when the two come
back as different classes -- so a regression there is invisible to the
``test_dispatch`` benchmarks.

The vjp side adds a third wrapper: ``_custom_vjp_fwd_wrap`` re-splices residuals
that JAX pruned as forwarded inputs, and ``_custom_vjp_bwd_wrap`` expands each
cotangent back to one entry per input *leaf*. Both are per-call work that only
the grad benchmark reaches.

`test_lora_mlp_grad` is the end-to-end version: `filter_grad` over a loraified
MLP under `jax.jit`, i.e. the workload quax's flagship example actually exists
for. It is warmed and jitted, so it measures the steady-state re-trace + compiled
call, not compilation.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import quax
from quax.examples import lora

from ..unit.myarray import MyArray


@jax.custom_jvp
def _custom_jvp_fn(x):
    return jnp.sin(x)


@_custom_jvp_fn.defjvp
def _custom_jvp_fn_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    return jnp.sin(x), jnp.cos(x) * x_dot


_xm = MyArray(jnp.arange(8.0) + 1)


@pytest.mark.benchmark(group="autodiff")
def test_custom_jvp_forward(bench):
    """`quaxify` over a `custom_jvp` function — `process_custom_jvp_call` +
    `_custom_jvp_fun_wrap` (forward only; the jvp rule is never entered)."""
    bench(_custom_jvp_fn, _xm)


@pytest.mark.benchmark(group="autodiff")
def test_custom_jvp_grad(bench):
    """`quaxify(grad(custom_jvp fn))` — additionally exercises
    `_custom_jvp_jvp_wrap`: the tangent `SymbolicZero` scan and the
    primal/tangent class reconciliation."""
    bench(eqx.filter_grad(lambda a: jnp.sum(_custom_jvp_fn(a))), _xm)


@jax.custom_vjp
def _custom_vjp_fn(x):
    return jnp.sin(x)


def _custom_vjp_fn_fwd(x):
    return jnp.sin(x), jnp.cos(x)


def _custom_vjp_fn_bwd(res, ct):
    return (res * ct,)


_custom_vjp_fn.defvjp(_custom_vjp_fn_fwd, _custom_vjp_fn_bwd)


@pytest.mark.benchmark(group="autodiff")
def test_custom_vjp_forward(bench):
    """`quaxify` over a `custom_vjp` function — `process_custom_vjp_call` +
    `_custom_vjp_fun_wrap` (forward only; fwd/bwd are never entered)."""
    bench(_custom_vjp_fn, _xm)


@pytest.mark.benchmark(group="autodiff")
def test_custom_vjp_grad(bench):
    """`quaxify(grad(custom_vjp fn))` — additionally exercises
    `_custom_vjp_fwd_wrap` (residual re-splicing, the substituted `out_trees`
    thunk) and `_custom_vjp_bwd_wrap` (per-leaf cotangent expansion)."""
    bench(eqx.filter_grad(lambda a: jnp.sum(_custom_vjp_fn(a))), _xm)


_key = jr.PRNGKey(0)
_mlp = lora.loraify(
    eqx.nn.MLP(8, 8, 32, 2, activation=jax.nn.relu, key=_key), rank=4, key=_key
)
_vector = jr.normal(_key, (8,))


@pytest.mark.benchmark(group="autodiff")
def test_lora_mlp_grad(benchmark):
    """`jit(grad(quaxify(...)))` over a loraified MLP — the end-to-end LoRA
    training step. `jax.nn.relu` is a `custom_jvp`, so this covers the autodiff
    path together with LoRA's `dot_general` rule."""
    run = eqx.filter_jit(eqx.filter_grad(quax.quaxify(lambda m, x: jnp.sum(m(x)))))
    run(_mlp, _vector)  # compile, and fill the quax caches
    benchmark(lambda: jax.block_until_ready(run(_mlp, _vector)))
