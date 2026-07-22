"""Benchmarks for quax's control-flow primitive handlers.

``scan_quax``, ``while_quax``, and ``cond_quax`` each build (and cache) a
quaxified sub-jaxpr for the loop/branch body and re-bind the primitive with it.
They are a core part of quax -- distinct from the elementwise dispatch in
``test_dispatch`` -- and had no other benchmark coverage. A regression in that
machinery (the jaxpr caches, the ``_compat`` scan-param handling, or the
carry/branch pytree bookkeeping) shows up here.

Each body is wrapped in ``jax.jit`` and warmed once before timing, so the timed
call is the steady-state path a real program hits: the quaxified body jaxpr is
built and compiled during warm-up (outside the measurement), and each timed call
re-enters the quax trace, hits the cached ``jit_quax`` kernel, and runs the
compiled loop/branch. (Eager ``quaxify`` without ``jax.jit`` would instead rebuild
and recompile the body on every call -- and ``scan_p`` never even reaches
``scan_quax`` outside ``jit``.) Small operands keep XLA a sliver.
"""

import jax
import jax.numpy as jnp
import pytest

import quax

from ..unit.myarray import MyArray


_scan_init = MyArray(jnp.array(1.0))
_scan_xs = MyArray(jnp.arange(1.0, 4.0))
_while_init = MyArray(jnp.array(0.0))
_cond_x = MyArray(jnp.arange(8.0) + 1)


def _bench(benchmark, fn, *args):
    """Warm the jit + dispatch caches, then benchmark the steady-state call."""
    qfn = quax.quaxify(jax.jit(fn))
    qfn(*args)  # compile the body and fill the quax caches
    benchmark(lambda: qfn(*args))


@pytest.mark.benchmark(group="control_flow")
def test_scan(benchmark):
    """`quaxify(jit(...))` over a `lax.scan` — exercises `scan_quax` + its cache."""

    def f(init, xs):
        return jax.lax.scan(lambda c, x: (c + x, c * x), init, xs)

    _bench(benchmark, f, _scan_init, _scan_xs)


@pytest.mark.benchmark(group="control_flow")
def test_while(benchmark):
    """`quaxify(jit(...))` over a `lax.while_loop` — exercises `while_quax`."""

    def f(x):
        return jax.lax.while_loop(lambda c: c < 5.0, lambda c: c + 1.0, x)

    _bench(benchmark, f, _while_init)


@pytest.mark.benchmark(group="control_flow")
def test_cond(benchmark):
    """`quaxify(jit(...))` over a `lax.switch` — exercises `cond_quax`."""

    def f(index, x):
        return jax.lax.switch(index, [lambda a: a + 1.0, lambda a: a * 2.0], x)

    _bench(benchmark, f, 0, _cond_x)
