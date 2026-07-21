"""Correctness guarantees for the `_FastModuleMeta` construction fast path.

The fast path skips `equinox.Module`'s per-instance *validation* but must preserve
every behaviour user code can depend on: field converters, `__check_init__`
invariants, static fields, pytree flatten/unflatten, and abstract-instantiation
errors. It must also degrade gracefully when its fast path is unavailable.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import quax
from quax._compat import typeof
from quax._module import _FastModuleMeta, _FASTPATH_AVAILABLE


def test_value_uses_fast_metaclass():
    """`quax.Value` (and thus every subclass) is built by `_FastModuleMeta`."""
    assert isinstance(quax.Value, _FastModuleMeta)
    assert isinstance(quax.ArrayValue, _FastModuleMeta)


def test_fast_path_available():
    """CI canary: the fast construction path must stay live on supported equinox.

    The fast path relies on a few `equinox` internals behind a guarded import that
    falls back to a correct-but-slow path if they move. That fallback keeps Quax
    working but silently removes the whole point of this module. This test makes the
    regression loud: if a future `equinox` breaks the fast path, CI (including the
    "newest supported deps" job) goes red here.

    If this fails after an `equinox` upgrade, update `quax/_module.py` to the new
    internals — do not delete this test or the guard.
    """
    assert _FASTPATH_AVAILABLE is True


def test_fast_path_actually_bypasses_equinox_slow_call(monkeypatch):
    """Stronger canary: fast construction must not go through equinox's slow
    `_ModuleMeta.__call__`, even if the guarded import happens to still succeed."""
    import equinox._module._module as eqxmod

    seen: list[str] = []
    original = eqxmod._ModuleMeta.__call__

    def spy(cls, *args, **kwargs):
        seen.append(cls.__name__)
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(eqxmod._ModuleMeta, "__call__", spy)

    _WithConverter(jnp.arange(3.0))  # concrete fast type
    assert "_WithConverter" not in seen  # never hit equinox's per-instance validation


def test_degradation_is_not_silent():
    """An incompatible equinox must *warn* (not silently) and fall back correctly.

    Runs in a subprocess so simulating a broken equinox (by removing an internal
    before quax imports) cannot corrupt this test session's module state.
    """
    import subprocess
    import sys
    import textwrap

    # Make ONLY quax's `from equinox._module._module import ... is_abstract_module`
    # fail, by returning a shim missing that name — while leaving the real equinox
    # module (which its own code depends on) intact. This mirrors a future equinox
    # that has moved the internal, without breaking equinox itself.
    script = textwrap.dedent(
        """
        import builtins, types, warnings

        _real_import = builtins.__import__
        _target = "equinox._module._module"
        _hidden = "is_abstract_module"

        def _patched_import(name, g=None, l=None, fromlist=(), level=0):
            mod = _real_import(name, g, l, fromlist, level)
            if name == _target and fromlist and _hidden in fromlist:
                shim = types.ModuleType(name)
                for k, v in vars(mod).items():
                    if k != _hidden:
                        setattr(shim, k, v)
                return shim
            return mod

        builtins.__import__ = _patched_import
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                import quax
        finally:
            builtins.__import__ = _real_import

        cats = {w.category.__name__ for w in rec}
        assert "FastPathUnavailableWarning" in cats, f"no warning; saw {cats}"
        assert quax._module._FASTPATH_AVAILABLE is False
        # Fallback must still construct Values correctly (equinox slow path).
        import jax.numpy as jnp
        from quax._values import _DenseArrayValue
        assert _DenseArrayValue(jnp.arange(2.0)).array is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


class _WithConverter(quax.ArrayValue):
    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self):
        return self.array

    def aval(self):
        return typeof(self.array)


def test_converter_is_applied():
    """A field converter still runs on the fast path."""
    v = _WithConverter([1.0, 2.0, 3.0])  # python list -> converter -> jax.Array
    assert isinstance(v.array, jax.Array)
    assert jnp.array_equal(v.array, jnp.asarray([1.0, 2.0, 3.0]))


class _Checked(quax.ArrayValue):
    data: jax.Array
    scale: float = eqx.field(static=True)

    def __init__(self, data, scale=2.0):
        self.data = jnp.asarray(data)
        self.scale = scale

    def __check_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def materialise(self):
        return self.data * self.scale

    def aval(self):
        return typeof(self.data)


def test_check_init_is_enforced():
    """`__check_init__` still raises through the fast path (custom __init__)."""
    _Checked(jnp.arange(3.0), scale=1.0)  # ok
    with pytest.raises(ValueError, match="scale must be positive"):
        _Checked(jnp.arange(3.0), scale=-1.0)


def test_static_field_and_pytree_roundtrip():
    """Static fields are preserved and excluded from the dynamic pytree leaves."""
    v = _Checked(jnp.arange(4.0), scale=3.0)
    leaves, treedef = jtu.tree_flatten(v)
    # `scale` is static, so `data` is the only dynamic leaf.
    assert len(leaves) == 1
    assert jnp.array_equal(leaves[0], jnp.arange(4.0))
    v2 = jtu.tree_unflatten(treedef, leaves)
    assert v2.scale == 3.0
    assert jnp.array_equal(v2.data, v.data)


def test_abstract_instantiation_still_errors():
    """Abstract Values cannot be instantiated (equinox's error is preserved)."""
    with pytest.raises(TypeError):
        quax.ArrayValue()  # abstract: aval/materialise unimplemented


@quax.register(jax.lax.mul_p)
def _mul_with_converter(
    a: _WithConverter, b: _WithConverter, **params: object
) -> _WithConverter:
    # Forward the primitive's params (e.g. JAX's newer `out_dtype`) rather than
    # hard-coding `a.array * b.array`, so the rule stays valid across JAX versions.
    return _WithConverter(jax.lax.mul_p.bind(a.array, b.array, **params))


def test_quaxify_roundtrip_through_fast_type():
    """End-to-end: a fast-constructed Value flows through quaxify correctly.

    Exercises the full round-trip — construction, wrapping into tracers, a
    registered dispatch rule returning a `Value`, and unflattening back out.
    """
    x = _WithConverter(jnp.arange(5.0))
    out = jax.jit(quax.quaxify(lambda z: z * z))(x)
    assert isinstance(out, _WithConverter)
    assert jnp.array_equal(out.array, jnp.arange(5.0) ** 2)


def test_fast_spec_precomputed():
    """Qualifying Values register a spec (keyed off the class, not stored on it)."""
    from quax._module import _fast_specs

    assert _WithConverter in _fast_specs
    assert _Checked in _fast_specs
    # Converter recorded for application; check_init recorded for enforcement.
    assert any(name == "array" for name, _ in _fast_specs[_WithConverter].converters)
    assert len(_fast_specs[_Checked].checks) == 1
    # Metadata lives off the class: no bespoke dunders left on user types.
    assert not any(a.startswith("__quax") for a in vars(_WithConverter))
