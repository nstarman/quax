"""Tests for the _jit_quax_cache in quax/_core.py.

Two structural assumptions are verified:

1. JAX's jaxpr is stable (same Python object) across repeated traces with
   identical argument avals, making id(jaxpr) a reliable cache key.
2. The cache key is (id(jaxpr), treedef) — inline is NOT part of the key;
   calls that JAX asked to inline at trace time bypass the cache entirely
   (early return).

Plus the GC-safety invariant: cache entries store weakrefs to the jaxpr so
the finalizer can evict stale entries, preventing unbounded cache growth.

Both eager mode and JIT-inside-JIT mode populate the cache (non-inlined only).
The jaxpr is kept alive by JAX's own pjit layer, so the weakref finalizer is
rarely if ever triggered in normal usage.
"""

import gc
import weakref
from typing import Any

import jax
import jax.numpy as jnp
import pytest

import quax
from quax._compat import JAX_GE_0_11_0, jit_p
from quax._primitives import _jit_quax_cache, jit_quax

from .myarray import MyArray


# quaxify short-circuits to `fn(*args)` when no operand is a quax Value (#58), so
# the jit_quax cache — which only matters when Values flow through a `jit_p` — is
# exercised by passing a `MyArray`. `_v` wraps a plain array as that Value.
_v = MyArray


@pytest.fixture
def inner_add_one():
    """A fresh `@jax.jit` function of `x + 1.0`, with the cache cleared first."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner(x):
        return x + 1.0

    return inner


def _extract_jit_jaxpr(closed_jaxpr: Any) -> Any:
    """Return the ClosedJaxpr param from the first jit_p equation, or None."""
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is jit_p:
            return eqn.params["jaxpr"]
    return None


def test_jaxpr_id_stable_same_avals():
    """JAX reuses the same ClosedJaxpr object on repeated traces with identical
    argument avals — so id(jaxpr) is a reliable cache key."""

    @jax.jit
    def inner(x):
        return x * 2.0

    def outer(x):
        return inner(x)

    x = jnp.array(1.0)
    closed1 = jax.make_jaxpr(outer)(x)
    closed2 = jax.make_jaxpr(outer)(x)

    jaxpr1 = _extract_jit_jaxpr(closed1)
    jaxpr2 = _extract_jit_jaxpr(closed2)

    assert jaxpr1 is not None, "No jit_p equation found in outer jaxpr"
    assert jaxpr2 is not None, "No jit_p equation found in outer jaxpr"
    assert jaxpr1 is jaxpr2, (
        "JAX produced a new ClosedJaxpr object for identical avals — "
        "id(jaxpr) is no longer a reliable cache key for _jit_quax_cache."
    )


def test_jaxpr_id_differs_for_different_avals():
    """Different argument avals produce distinct ClosedJaxpr objects, so
    id(jaxpr) does not produce false cache collisions."""

    @jax.jit
    def inner(x):
        return x * 2.0

    def outer(x):
        return inner(x)

    x_scalar = jnp.array(1.0)  # shape=()
    x_vector = jnp.array([1.0, 2.0])  # shape=(2,)

    closed_scalar = jax.make_jaxpr(outer)(x_scalar)
    closed_vector = jax.make_jaxpr(outer)(x_vector)

    jaxpr_scalar = _extract_jit_jaxpr(closed_scalar)
    jaxpr_vector = _extract_jit_jaxpr(closed_vector)

    assert jaxpr_scalar is not None
    assert jaxpr_vector is not None
    assert jaxpr_scalar is not jaxpr_vector, (
        "Expected distinct ClosedJaxpr objects for inputs with different shapes."
    )


def test_jit_quax_cache_populates_on_first_call(inner_add_one):
    """The first quaxify call involving an inner @jax.jit function adds an
    entry to _jit_quax_cache."""
    quax.quaxify(inner_add_one)(_v(jnp.array(0.0)))
    assert len(_jit_quax_cache) > 0, "Cache was not populated on first call."


def test_jit_quax_cache_hit_same_avals(inner_add_one):
    """A second quaxify call with the same argument avals hits the existing
    cache entry and does not add a new one."""
    x = _v(jnp.array(0.0))
    quax.quaxify(inner_add_one)(x)
    size_after_first = len(_jit_quax_cache)
    assert size_after_first > 0

    quax.quaxify(inner_add_one)(x)
    assert len(_jit_quax_cache) == size_after_first, (
        "Cache grew on the second call with identical avals — expected a hit."
    )


def test_jit_quax_cache_miss_different_treedef():
    """Functions with structurally different argument trees produce separate
    cache entries — treedef is part of the cache key."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner_one(x):
        return x + 1.0

    @jax.jit
    def inner_two(x, y):
        return x + y

    x = _v(jnp.array(1.0))
    quax.quaxify(inner_one)(x)
    quax.quaxify(inner_two)(x, x)

    assert len(_jit_quax_cache) >= 2, (
        "Expected >=2 cache entries for functions with different treedefs, "
        f"got {len(_jit_quax_cache)}."
    )


def test_jit_quax_cache_entry_weakref_matches_key():
    """Each cache entry stores a weakref.ref to the ClosedJaxpr as entry[0].
    While the jaxpr is alive the weakref target's id must equal key[0]."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner(x):
        return x * 3.0

    quax.quaxify(inner)(_v(jnp.array(1.0)))
    assert _jit_quax_cache, "Cache was not populated — nothing to check."

    for key, entry in _jit_quax_cache.items():
        stored_id, _treedef = key  # key is now (id(jaxpr), treedef) — no inline
        wref = entry[0]
        assert isinstance(wref, weakref.ref), (
            "entry[0] should be a weakref.ref to the jaxpr."
        )
        live_jaxpr = wref()
        assert live_jaxpr is not None, "Weakref target was already collected."
        assert id(live_jaxpr) == stored_id, (
            f"Weakref target has id={id(live_jaxpr)}, "
            f"but the cache key records id={stored_id}."
        )


def test_jit_quax_cache_stable_while_jaxpr_alive():
    """Cache entries are not evicted while the referenced jaxpr is still alive,
    and re-calling after a GC cycle returns the correct result."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner(x):
        return x * 3.0

    x = _v(jnp.array(1.0))
    quax_fn = quax.quaxify(inner)
    expected = float(quax_fn(x).array)

    keys_before = set(_jit_quax_cache.keys())
    assert keys_before, "Cache should be populated after first call."

    # inner holds a strong reference to the jaxpr via JAX's pjit cache, so
    # the weakref finalizer must not fire here.
    gc.collect()
    assert set(_jit_quax_cache.keys()) == keys_before, (
        "Cache entries were evicted while jaxpr was still reachable."
    )

    assert float(quax_fn(x).array) == expected


# ---------------------------------------------------------------------------
# JIT-inside-JIT cache behaviour
# ---------------------------------------------------------------------------


def test_jit_inside_jit_populates_cache(inner_add_one):
    """A quaxified outer @jax.jit that calls an inner @jax.jit produces at
    least two _jit_quax_cache entries — one for the outer jit_p equation and
    one for the inner jit_p equation encountered while tracing the outer body.
    This exercises the JIT-inside-JIT population path described in the module
    docstring, which is distinct from the single-level cache-hit tests above."""

    @jax.jit
    def outer(x):
        return inner_add_one(x)

    result = quax.quaxify(outer)(_v(jnp.array(0.0)))
    assert float(result.array) == 1.0  # correct answer

    assert len(_jit_quax_cache) >= 2, (
        "Expected >=2 cache entries for a JIT-inside-JIT call "
        f"(outer + inner), got {len(_jit_quax_cache)}."
    )


# ---------------------------------------------------------------------------
# inline=True early-return path
# ---------------------------------------------------------------------------


def test_inline_true_bypasses_cache(inner_add_one):
    """jit_p with inline=True takes the early-return path in jit_quax and
    must not read from or write to _jit_quax_cache.  We call jit_quax
    directly with inline=True to avoid any JAX-version sensitivity around
    when jax.jit(inline=True) produces a jit_p equation vs. inlines at
    trace time."""

    def outer(x):
        return inner_add_one(x)

    # Extract the ClosedJaxpr that jit_p carries for inner — same object
    # _QuaxTrace would pass to jit_quax via process_primitive.
    closed_outer = jax.make_jaxpr(outer)(jnp.array(0.0))
    inner_jaxpr = _extract_jit_jaxpr(closed_outer)
    assert inner_jaxpr is not None, "No jit_p equation found in outer jaxpr"

    keys_before = set(_jit_quax_cache.keys())

    # Directly invoke the handler with inline=True — same path _QuaxTrace
    # takes when jit_p params carry inline=True.
    result = jit_quax(jnp.array(0.0), jaxpr=inner_jaxpr, inline=True)

    # jaxpr_as_fun returns outputs as a list; unpack the single element.
    out = result[0] if isinstance(result, list) else result
    assert float(out) == 1.0  # correct answer
    assert set(_jit_quax_cache.keys()) == keys_before, (
        "inline=True must bypass _jit_quax_cache entirely — no entries "
        "should be added or removed."
    )


@pytest.mark.skipif(not JAX_GE_0_11_0, reason="jax.Inline was introduced in JAX 0.11.0")
@pytest.mark.parametrize(
    ("member", "inlines"),
    [
        ("JAX_EARLY", True),
        ("JAX_LATE", False),
        ("XLA_EARLY", False),
        ("XLA_LATE", False),
        ("AUTO", False),
    ],
)
def test_inline_enum_members(member, inlines, inner_add_one):
    """From JAX 0.11.0 `inline` is a `jax.Inline` enum rather than a bool.
    Only `JAX_EARLY` (the old `True`) takes the early-return path; every other
    member keeps the call in the jaxpr and so must populate the cache, like the
    old `False` did.  Enum members are all truthy, so a plain `if inline:`
    would wrongly inline every one of them."""

    def outer(x):
        return inner_add_one(x)

    closed_outer = jax.make_jaxpr(outer)(jnp.array(0.0))
    inner_jaxpr = _extract_jit_jaxpr(closed_outer)
    assert inner_jaxpr is not None, "No jit_p equation found in outer jaxpr"

    inline = getattr(jax.Inline, member)
    result = jit_quax(jnp.array(0.0), jaxpr=inner_jaxpr, inline=inline)

    out = result[0] if isinstance(result, list) else result
    assert float(out) == 1.0  # correct answer either way

    assert (len(_jit_quax_cache) == 0) is inlines, (
        f"Inline.{member} should {'bypass' if inlines else 'populate'} "
        f"_jit_quax_cache, but the cache has {len(_jit_quax_cache)} entries."
    )
