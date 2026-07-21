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
from quax._module import _FastModuleMeta


def test_value_uses_fast_metaclass():
    """`quax.Value` (and thus every subclass) is built by `_FastModuleMeta`."""
    assert isinstance(quax.Value, _FastModuleMeta)
    assert isinstance(quax.ArrayValue, _FastModuleMeta)


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


def test_fast_flag_precomputed():
    """Concrete simple Values qualify for the fast path; class metadata is cached."""
    assert _WithConverter.__quax_fast__ is True
    assert _Checked.__quax_fast__ is True
    # Converter recorded for application; check_init recorded for enforcement.
    assert any(name == "array" for name, _ in _WithConverter.__quax_converters__)
    assert len(_Checked.__quax_checks__) == 1
