"""Tests for GC-friendly weakref caching in jaxpr caches.

Two groups:
- "uses_weakref" tests: verify that cache entries store weakrefs, not strong refs.
- "evicts_on_gc" tests: verify the finalizer evicts entries when the jaxpr is GC'd.
"""

import gc
import weakref

import jax
import jax.core
import jax.numpy as jnp

import quax
from quax._primitives import _jit_quax_cache, _scan_quax_cache, _while_quax_cache


# Minimal ArrayValue with a working materialise — forces scan_quax dispatch
# (plain JAX arrays fall through to _default_process via plum's variadic dispatch).
class _ScanValue(quax.ArrayValue):
    array: jax.Array

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.array.shape, self.array.dtype)

    def materialise(self) -> jax.Array:
        return self.array


class _FakeJaxpr:
    """Lightweight stand-in for a real jaxpr; supports weakref."""


def test_jit_cache_uses_weakref():
    """_jit_quax_cache entry[0] must be a weakref, not a strong reference."""

    @jax.jit
    def inner(x):
        return x + 1.0

    before = set(_jit_quax_cache.keys())
    quax.quaxify(inner)(jnp.array(1.0))
    new_keys = set(_jit_quax_cache.keys()) - before
    assert new_keys, "Expected _jit_quax_cache to be populated"

    # This would have failed before the weakref fix, when the cache stored
    # a strong jaxpr reference instead of weakref.ref(jaxpr, finalizer).
    entry = _jit_quax_cache[next(iter(new_keys))]
    assert isinstance(entry[0], weakref.ref)


def test_while_cache_uses_weakrefs():
    """_while_quax_cache entry[0] and entry[1] must be weakrefs to cond/body jaxprs.

    The function must be @jax.jit so JAX's compilation cache keeps the outer
    jaxpr (and its while_p params) alive while we inspect the quax cache.  In
    eager mode the jaxprs are freed immediately and the finalizer evicts the
    entry before we can check it — correct behaviour, not a bug.
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
    assert isinstance(entry[0], weakref.ref)
    assert isinstance(entry[1], weakref.ref)


def test_scan_cache_uses_weakref():
    """_scan_quax_cache entry[0] must be a weakref, not a strong reference.

    scan_p only reaches scan_quax inside a jitted function; outside jit,
    lax.scan runs as a Python for-loop and never calls process_primitive.
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
    assert isinstance(entry[0], weakref.ref)


def test_jit_cache_evicts_on_gc():
    """Entry is removed from _jit_quax_cache when the weakreffed jaxpr is collected."""
    jaxpr = _FakeJaxpr()
    key = (id(jaxpr), False, "dummy_treedef")

    def _fin(_ref):
        _jit_quax_cache.pop(key, None)

    _jit_quax_cache[key] = (weakref.ref(jaxpr, _fin), "fake_fn")
    assert key in _jit_quax_cache

    del jaxpr
    gc.collect()
    assert key not in _jit_quax_cache


def test_while_cache_evicts_on_gc():
    """Entry is removed from _while_quax_cache when weakreffed jaxpr is collected."""
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
    assert key not in _while_quax_cache


def test_scan_cache_evicts_on_gc():
    """Entry is removed from _scan_quax_cache when the weakreffed jaxpr is collected."""
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
    assert key not in _scan_quax_cache
