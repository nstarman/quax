"""Benchmarks for `_Quaxify.__call__`'s per-call overhead.

The other suites all measure *per-primitive* cost with a fixed, tiny argument
pytree. This one measures what `quaxify` charges before and after the traced
function runs, which is per-*call* and scales with the argument pytree rather
than with the number of primitives:

- the no-`Value` short-circuit (`_is_value` over every leaf of `(fn, args,
  kwargs)`, then straight through to `self.fn`) — pure tax on quaxified code
  called with plain arrays;
- `_partition_and_wrap`'s general branch, taken whenever `filter_spec is not
  True`, which pays `eqx.partition` + `eqx.combine` on top of the `tree_map` the
  fast branch does alone;
- wrapping/unwrapping a wide pytree of `Value`s, where the two `tree_map` passes
  (in `_partition_and_wrap` and over the output) dominate.

None of these paths were covered before; a regression in any of them is
invisible to a benchmark that passes two arrays and one primitive.
"""

import jax.numpy as jnp
import pytest

import quax

from ..unit.myarray import MyArray


_arr = jnp.arange(8.0)
_xm = MyArray(_arr)
_ym = MyArray(_arr + 1)


@pytest.mark.benchmark(group="quaxify")
def test_no_value_shortcut(bench):
    """`quaxify(f)(array)` with no `Value` anywhere — the short-circuit that
    skips the trace entirely, leaving only the `tree_leaves` scan."""
    bench(lambda a, b: a + b, _arr, _arr)


@pytest.mark.benchmark(group="quaxify")
def test_filter_spec_partition(bench):
    """`quaxify(..., filter_spec=False)` — `_partition_and_wrap`'s general
    branch (`eqx.partition` + `eqx.combine`), passing the `Value`s through to a
    nested `quaxify` (the redispatch pattern)."""
    inner = quax.quaxify(lambda a, b: a + b)
    bench(lambda a, b: inner(a, b), _xm, _ym, filter_spec=False)


_tree = [MyArray(_arr) for _ in range(32)]


@pytest.mark.benchmark(group="quaxify")
def test_wide_pytree(bench):
    """32 `Value`s in one argument pytree — the `tree_map` wrap/unwrap passes,
    plus 32 primitives' worth of dispatch."""
    bench(lambda ts: [a + a for a in ts], _tree)
