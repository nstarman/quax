"""Benchmarks for the autodiff path through a quax trace.

``_QuaxTrace.process_custom_jvp_call`` and its two ``lu.transformation_with_aux``
wrappers (``_custom_jvp_fun_wrap`` / ``_custom_jvp_jvp_wrap``) are the only
``process_*`` override besides ``process_primitive``, and had no benchmark
coverage at all. They do noticeably more per call than plain dispatch --
flatten/unflatten of every `Value` through a treedef, the `SymbolicZero`
promotion scan over the tangents, and the primal/tangent `materialise` fallback
when the two come back as different classes -- so a regression there is invisible
to the ``test_dispatch`` benchmarks.

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
def test_custom_jvp_forward(benchmark):
    """`quaxify` over a `custom_jvp` function — `process_custom_jvp_call` +
    `_custom_jvp_fun_wrap` (forward only; the jvp rule is never entered)."""
    qfn = quax.quaxify(_custom_jvp_fn)
    qfn(_xm)  # warm the dispatch cache
    benchmark(lambda: qfn(_xm))


@pytest.mark.benchmark(group="autodiff")
def test_custom_jvp_grad(benchmark):
    """`quaxify(grad(custom_jvp fn))` — additionally exercises
    `_custom_jvp_jvp_wrap`: the tangent `SymbolicZero` scan and the
    primal/tangent class reconciliation."""
    qfn = quax.quaxify(eqx.filter_grad(lambda a: jnp.sum(_custom_jvp_fn(a))))
    qfn(_xm)
    benchmark(lambda: qfn(_xm))


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
