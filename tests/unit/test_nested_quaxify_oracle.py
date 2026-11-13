"""A quaxified call inside a quaxified function, against a flat oracle.

The nested call should give the same result as one `quaxify` over the same
operation with every Value passed in as an argument, wherever its operands were
created. And when none of its operands came from outside, the code calling it
should see that result as a Value of the same type, as it would outside. The cases
vary the Value types, where each operand is created, the operation, the nesting
depth, and whether `jax.jit` wraps the whole.

A custom `filter_spec`, and calls from rules, deliberately differ from this; see
`test_core.py`.
"""

import itertools as it
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

import quax
from quax.examples.unitful import meters, seconds, Unitful
from quax.examples.zero import Zero


_MAKE: dict[str, Callable[[Any], Any]] = {
    "metres": lambda a: Unitful(a, meters),
    "seconds": lambda a: Unitful(a, seconds),
    "zero": lambda a: Zero(jnp.shape(a), jnp.result_type(a)),
}
_BASE = (np.array([3.0, 4.0], np.float32), np.array([2.0, 5.0], np.float32))
# Where an operand is created: passed to the outer `quaxify`; built inside the
# outer function from a constant; or built inside from a traced operand, so that
# its field is a tracer of the outer trace.
_WHERE = ("outer", "local", "traced")
_OPS = {"multiply": jnp.multiply, "add": jnp.add}


def _nested(op, kinds, wheres, depth, use_jit, seen):
    """`op` called inside `depth - 1` quaxified levels. Appends to `seen` the tree
    structure the calling code sees for the result (structures only: keeping the
    result would leak tracers under `jit`)."""
    inner = quax.quaxify(op)
    passed = [
        _MAKE[k](jnp.asarray(b))
        for k, w, b in zip(kinds, wheres, _BASE)
        if w == "outer"
    ]

    # `bystander` is an outer Value `op` never sees. It makes the outer `quaxify`
    # trace even when no operand comes from outside; with no Value passed in, it
    # would run `f` untraced.
    bystander = Unitful(jnp.ones(2), meters)

    def f(traced, bystander, *outer):
        del bystander
        outer = iter(outer)
        operands = []
        for i, (kind, where) in enumerate(zip(kinds, wheres)):
            if where == "outer":
                operands.append(next(outer))
            elif where == "local":
                operands.append(_MAKE[kind](jnp.asarray(_BASE[i])))
            else:
                operands.append(_MAKE[kind](traced[i] * 1.0))
        call = inner
        for _ in range(depth - 2):  # quaxified levels in between
            call = quax.quaxify(lambda *o, c=call: c(*o))
        out = call(*operands)
        seen.append(jtu.tree_structure(out))
        return out

    g = quax.quaxify(f)
    if use_jit:
        g = jax.jit(g)
    return g(jnp.stack(_BASE), bystander, *passed)


def _flat(op, kinds):
    return quax.quaxify(op)(*(_MAKE[k](jnp.asarray(b)) for k, b in zip(kinds, _BASE)))


def _assert_same(got, expected):
    assert jtu.tree_structure(got) == jtu.tree_structure(expected)
    for g, e in zip(jtu.tree_leaves(got), jtu.tree_leaves(expected)):
        assert jnp.shape(g) == jnp.shape(e)
        np.testing.assert_allclose(np.asarray(g), np.asarray(e))


_CASES = [
    pytest.param(
        op,
        kinds,
        wheres,
        depth,
        use_jit,
        id="-".join((op, *kinds, *wheres, f"d{depth}", *(("jit",) if use_jit else ()))),
    )
    for op in _OPS
    for kinds in it.product(_MAKE, repeat=2)
    for wheres in it.product(_WHERE, repeat=2)
    for depth, use_jit in ((2, False), (2, True), (3, False))
    # Each case takes ~0.1 s with the suite's runtime type checking, so only the
    # plain case runs every pair of types; the variants use a mixed pair.
    if (depth, use_jit) == (2, False) or set(kinds) <= {"metres", "zero"}
]


@pytest.mark.parametrize(("op", "kinds", "wheres", "depth", "use_jit"), _CASES)
def test_nested_matches_flat(op, kinds, wheres, depth, use_jit):
    seen: list[Any] = []
    try:
        expected = _flat(_OPS[op], kinds)
    except Exception as e:  # noqa: BLE001
        with pytest.raises(type(e)):
            _nested(_OPS[op], kinds, wheres, depth, use_jit, seen)
        return
    got = _nested(_OPS[op], kinds, wheres, depth, use_jit, seen)
    _assert_same(got, expected)
    if "outer" not in wheres:
        assert seen == [jtu.tree_structure(expected)]
