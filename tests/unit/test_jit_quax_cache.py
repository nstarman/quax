"""Tests for the _jit_quax_cache in quax/_core.py.

Two structural assumptions are verified:

1. JAX's jaxpr is stable (same Python object) across repeated traces with
   identical argument avals, making id(jaxpr) a reliable cache key.
2. The cache key includes (inline, treedef) to distinguish different call
   shapes and the inline vs jit-wrapped code path.

Plus the GC-safety invariant: cache entries store weakrefs to the jaxpr so
the finalizer can evict stale entries, preventing unbounded cache growth.
"""

import gc
import weakref
from typing import Any

import jax
import jax.numpy as jnp

import quax
from quax._compat import jit_p
from quax._core import _jit_quax_cache


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


def test_jit_quax_cache_populates_on_first_call():
    """The first quaxify call involving an inner @jax.jit function adds an
    entry to _jit_quax_cache."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner(x):
        return x + 1.0

    quax.quaxify(inner)(jnp.array(0.0))
    assert len(_jit_quax_cache) > 0, "Cache was not populated on first call."


def test_jit_quax_cache_hit_same_avals():
    """A second quaxify call with the same argument avals hits the existing
    cache entry and does not add a new one."""
    _jit_quax_cache.clear()

    @jax.jit
    def inner(x):
        return x + 1.0

    x = jnp.array(0.0)
    quax.quaxify(inner)(x)
    size_after_first = len(_jit_quax_cache)
    assert size_after_first > 0

    quax.quaxify(inner)(x)
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

    x = jnp.array(1.0)
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

    quax.quaxify(inner)(jnp.array(1.0))

    for key, entry in _jit_quax_cache.items():
        stored_id, _inline, _treedef = key
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

    x = jnp.array(1.0)
    expected = float(quax.quaxify(inner)(x))

    keys_before = set(_jit_quax_cache.keys())
    assert keys_before, "Cache should be populated after first call."

    # inner holds a strong reference to the jaxpr via JAX's tracing cache, so
    # the weakref finalizer must not fire here.
    gc.collect()
    assert set(_jit_quax_cache.keys()) == keys_before, (
        "Cache entries were evicted while jaxpr was still reachable."
    )

    assert float(quax.quaxify(inner)(x)) == expected
