"""How Quax and `jax.experimental.hijax` interoperate.

Three claims that [Quax and hijax](../../docs/hijax.md) makes, pinned down so
that a change in either library shows up here rather than in that prose:

- a hijax value crossing the `quaxify` boundary is taken apart and rebuilt, so
  no Quax tracer escapes inside it;
- a hijax primitive applied under a Quax trace is expanded on the spot, so Quax
  rules apply to the `lax` operations inside its `expand`;
- an `ArrayValue` whose `aval()` returns a hijax type is rejected by `jnp`,
  which is why `aval()` must return a `ShapedArray`.
"""

import jax
import jax.numpy as jnp
import pytest

import quax


pytest.importorskip(
    "quax.examples.hijax", reason="jax.experimental.hijax is unavailable"
)

from quax.examples.hijax import Quantity, unwrap, wrap  # noqa: E402
from quax.examples.unitful import meters, Unitful  # noqa: E402


def _quax_value():
    """Any Quax value, so that `quaxify` actually installs a trace.

    `quaxify` short-circuits when no argument is a `quax.Value`, which would
    make these tests pass for the wrong reason.
    """
    return Unitful(jnp.array([1.0, 2.0]), meters)


def test_hi_value_returned_from_quaxify_holds_no_tracer():
    """Regression: the returned `Quantity` held a `_QuaxTracer`."""

    def f(_, array):
        return wrap(array * 2.0, meters)

    out = quax.quaxify(f)(_quax_value(), jnp.array([3.0, 4.0]))

    assert isinstance(out, Quantity)
    assert not isinstance(out.array, jax.core.Tracer)
    assert jnp.array_equal(out.array, jnp.array([6.0, 8.0]))


def test_hi_value_nested_in_a_pytree_output_holds_no_tracer():
    """The fix is applied per leaf, so nesting must not matter."""

    def f(_, array):
        return {"q": [wrap(array * 2.0, meters)]}

    out = quax.quaxify(f)(_quax_value(), jnp.array([3.0, 4.0]))

    assert not isinstance(out["q"][0].array, jax.core.Tracer)


def test_hi_value_passed_through_unchanged():
    """A hi value that the function only forwards comes back intact.

    Not necessarily the same object: it is taken apart and rebuilt through its
    own type, which is what makes the tracer case work.
    """
    quantity = wrap(jnp.array([1.0, 2.0]), meters)

    out = quax.quaxify(lambda _, q: q)(_quax_value(), quantity)

    assert isinstance(out, Quantity)
    assert out.units == ((meters, 1),)
    assert jnp.array_equal(out.array, jnp.array([1.0, 2.0]))


def test_a_plain_object_leaf_still_passes_through():
    """The hi-value branch must not disturb ordinary non-JAX leaves."""
    sentinel = object()

    out = quax.quaxify(lambda _, obj: obj)(_quax_value(), sentinel)

    assert out is sentinel


def test_hi_primitive_under_a_quax_trace_dispatches_on_quax_rules():
    """A hijax primitive is expanded under the Quax trace, not recorded.

    JAX asks each trace whether it needs primitives lowered before it sees them.
    Quax's does, so `expand` runs immediately and the `lax` operations inside it
    dispatch on Quax's rules -- here, on `quax.examples.unitful`'s `mul_p` rule,
    which is what multiplies the units.
    """
    from quax.examples.hijax import mul

    def f(value):
        return unwrap(mul(wrap(value, meters), wrap(value, meters)))

    out = quax.quaxify(f)(Unitful(jnp.array([2.0, 3.0]), meters))

    assert isinstance(out, Unitful)
    assert out.units == {meters: 2}
    assert jnp.allclose(out.array, jnp.array([4.0, 9.0]))


def test_an_array_value_aval_must_stay_a_shaped_array():
    """`jnp` rejects a tracer whose aval is a hijax type.

    `jax.Array`'s metaclass answers `isinstance` by looking at the tracer's
    aval, and every `jnp` function gates its arguments on that. So an
    `ArrayValue` cannot report a `HiType` from `aval()` and still run
    unmodified array code -- the reason `quax.examples.hijax.Unitful` reports a
    `ShapedArray` built *from* its leaf's hijax type instead. If JAX ever makes
    this work, that reason is gone and this test is the place it will show.
    """
    from quax.examples.hijax import QuantityTy

    class HiAval(quax.ArrayValue):
        array: jax.Array

        def aval(self):
            return QuantityTy(self.array.shape, self.array.dtype, ())

        def materialise(self):
            raise AssertionError

    value = HiAval(jnp.array([1.0, 2.0]))

    assert not quax.quaxify(lambda a: isinstance(a, jax.Array))(value)
    with pytest.raises(TypeError, match="requires ndarray or scalar"):
        quax.quaxify(jnp.multiply)(value, value)
