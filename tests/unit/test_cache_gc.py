"""Tests for GC-friendly weakref caching in jaxpr caches."""

import gc
import weakref

import jax
import jax.core
import jax.numpy as jnp

import quax
from quax._core import _jit_quax_cache, _scan_quax_cache, _while_quax_cache


# Minimal ArrayValue whose materialise works — used to force scan_quax dispatch
# (plain JAX arrays fall through to _default_process via plum's variadic dispatch).
class _ScanValue(quax.ArrayValue):
    array: jax.Array

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.array.shape, self.array.dtype)

    def materialise(self) -> jax.Array:
        return self.array


def test_jit_cache_uses_weakref():
    """jit_quax should cache a weakref to the jaxpr, not a strong reference."""

    @jax.jit
    def inner(x):
        return x + 1.0

    x = jnp.array(1.0)
    before = set(_jit_quax_cache.keys())
    quax.quaxify(inner)(x)
    new_keys = set(_jit_quax_cache.keys()) - before
    assert new_keys, "Expected _jit_quax_cache to be populated"

    entry = _jit_quax_cache[next(iter(new_keys))]
    assert isinstance(entry[0], weakref.ref), (
        "entry[0] should be a weakref.ref to the jaxpr, not a strong reference; "
        "strong refs prevent GC and cause unbounded cache growth"
    )


def test_while_cache_uses_weakrefs():
    """while_quax should cache weakrefs to cond_jaxpr and body_jaxpr.

    The function must be @jax.jit so JAX's own compilation cache keeps the
    outer jaxpr (and its while_p params: cond_jaxpr, body_jaxpr) alive while
    we inspect the cache.  In eager mode the jaxprs are freed immediately
    after while_quax returns and the finalizer evicts the entry before we
    can check it — which is correct weakref behaviour, not a bug.
    """

    @jax.jit
    def f(x):
        return jax.lax.while_loop(lambda y: y < 5.0, lambda y: y + 1.0, x)

    x = jnp.array(0.0)
    before = set(_while_quax_cache.keys())
    quax.quaxify(f)(x)
    new_keys = set(_while_quax_cache.keys()) - before
    assert new_keys, "Expected _while_quax_cache to be populated"

    entry = _while_quax_cache[next(iter(new_keys))]
    assert isinstance(entry[0], weakref.ref), (
        "entry[0] should be a weakref.ref to cond_jaxpr, not a strong reference"
    )
    assert isinstance(entry[1], weakref.ref), (
        "entry[1] should be a weakref.ref to body_jaxpr, not a strong reference"
    )


def test_scan_cache_uses_weakref():
    """scan_quax should cache a weakref to the jaxpr, not a strong reference.

    Note: lax.scan runs eagerly outside of jax.jit (Python for-loop), so
    scan_p only reaches process_primitive — and scan_quax — when the scan is
    inside a jitted function.
    """

    def f(const, carry, xs):
        def body(c, x):
            return c + x + const, c * x

        return jax.lax.scan(body, carry, xs)

    const = _ScanValue(jnp.array(0.0))
    carry = _ScanValue(jnp.array(1.0))
    xs = _ScanValue(jnp.array([1.0, 2.0, 3.0]))
    before = set(_scan_quax_cache.keys())
    quax.quaxify(jax.jit(f))(const, carry, xs)
    new_keys = set(_scan_quax_cache.keys()) - before
    assert new_keys, "Expected _scan_quax_cache to be populated"

    entry = _scan_quax_cache[next(iter(new_keys))]
    assert isinstance(entry[0], weakref.ref), (
        "entry[0] should be a weakref.ref to the jaxpr, not a strong reference"
    )


# ── GREEN regression tests: verify weakref + finalizer eviction mechanism ─────
# These directly insert entries using the fixed format (weakref + finalizer)
# and verify that GC fires the finalizer and removes the entry.
# They serve as regression tests for the eviction mechanism itself.


def test_jit_cache_evicts_on_gc():
    """Entry is removed from _jit_quax_cache when the weakreffed jaxpr is collected."""

    class _FakeJaxpr:
        pass

    jaxpr = _FakeJaxpr()
    key = (id(jaxpr), False, "dummy_treedef")

    def _fin(_ref):
        _jit_quax_cache.pop(key, None)

    _jit_quax_cache[key] = (weakref.ref(jaxpr, _fin), "fake_fn")
    assert key in _jit_quax_cache

    del jaxpr
    gc.collect()
    assert key not in _jit_quax_cache, (
        "Cache entry should have been evicted after jaxpr was GC'd"
    )


def test_while_cache_evicts_on_gc():
    """Entry is removed from _while_quax_cache when weakreffed jaxpr is collected.

    Either jaxpr being GC'd triggers the finalizer that evicts the cache entry.
    """

    class _FakeJaxpr:
        pass

    cond_j = _FakeJaxpr()
    body_j = _FakeJaxpr()
    key = (id(cond_j), id(body_j), "dummy_treedef")

    def _fin(_ref):
        _while_quax_cache.pop(key, None)

    _while_quax_cache[key] = (
        weakref.ref(cond_j, _fin),
        weakref.ref(body_j, _fin),
        "fake_quax_cond",
        "fake_quax_body",
        "dummy_treedef",
    )
    assert key in _while_quax_cache

    del cond_j, body_j
    gc.collect()
    assert key not in _while_quax_cache, (
        "Cache entry should have been evicted after jaxprs were GC'd"
    )


def test_scan_cache_evicts_on_gc():
    """Entry is removed from _scan_quax_cache when the weakreffed jaxpr is collected."""

    class _FakeJaxpr:
        pass

    jaxpr = _FakeJaxpr()
    key = (id(jaxpr), "c_tree", "v_tree", "x_tree")

    def _fin(_ref):
        _scan_quax_cache.pop(key, None)

    _scan_quax_cache[key] = (
        weakref.ref(jaxpr, _fin),
        "fake_quax_jaxpr",
        "out_tree",
        0,
        0,
    )
    assert key in _scan_quax_cache

    del jaxpr
    gc.collect()
    assert key not in _scan_quax_cache, (
        "Cache entry should have been evicted after jaxpr was GC'd"
    )
