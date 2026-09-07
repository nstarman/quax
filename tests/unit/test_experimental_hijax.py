"""`quax.experimental.hijax`: the `HiValue` wrapper and `register_rules`.

`quax.examples.hijax` is the end-to-end proof that these work -- it is built on
them -- so these tests cover the mechanism's own edges instead: which operand
combinations get a rule, which primitive parameters are forwarded, and what
happens to a result that is not a hijax value.
"""

from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

import quax


pytest.importorskip(
    "quax.experimental.hijax", reason="jax.experimental.hijax is unavailable"
)

import quax.examples.hijax._quantity as hi  # noqa: E402
from quax.examples.unitful import meters  # noqa: E402
from quax.experimental.hijax import HiValue, register_rules  # noqa: E402


class Tagged(HiValue):
    """A `HiValue` of its own, so its rules cannot collide with the example's."""

    def __init__(self, array: Any, units: Any = (), /) -> None:
        self.leaf = array if isinstance(array, hi.Quantity) else hi.wrap(array, units)

    @property
    def units(self) -> Any:
        return jax.typeof(self.leaf).units


register_rules(
    Tagged,
    {
        lax.mul_p: hi.mul,
        lax.add_p: hi.add,
        lax.integer_pow_p: lambda q, *, y, **kw: hi.int_pow(q, y),
        # Deliberately drops the units, to exercise a non-hijax result.
        lax.reduce_sum_p: lambda q, *, axes, **kw: hi.unwrap(hi.sum(q, axes)),
    },
)


@pytest.fixture
def tagged():
    return Tagged(jnp.array([1.0, 2.0, 3.0]), meters)


def test_aval_is_a_shaped_array_built_from_the_hi_type(tagged):
    """Not the hijax type itself: `jnp` gates its arguments on this."""
    aval = tagged.aval()

    assert type(aval) is jax.core.ShapedArray
    assert aval.shape == (3,)
    assert aval.dtype == jnp.float32
    assert quax.quaxify(lambda a: isinstance(a, jax.Array))(tagged)


def test_materialise_refuses(tagged):
    with pytest.raises(ValueError, match="Refusing to materialise Tagged"):
        tagged.materialise()


def test_the_hi_value_is_the_single_leaf(tagged):
    leaves = jax.tree.leaves(tagged)

    assert len(leaves) == 1
    assert isinstance(leaves[0], hi.Quantity)


def test_rules_cover_every_mixed_operand_combination(tagged):
    """A binary primitive is reachable with the type on either side, or both."""
    assert quax.quaxify(lambda a: a * a)(tagged).units == ((meters, 2),)
    assert quax.quaxify(lambda a: a * 2.0)(tagged).units == ((meters, 1),)
    assert quax.quaxify(lambda a: 2.0 * a)(tagged).units == ((meters, 1),)


def test_unmapped_primitive_falls_through_to_materialise(tagged):
    with pytest.raises(ValueError, match="Refusing to materialise"):
        quax.quaxify(jnp.sin)(tagged)


def test_primitive_params_the_function_does_not_want_are_dropped(tagged):
    """Regression: `mul_p` passes `out_dtype`, which two-argument `mul` rejects.

    Without filtering, every generated rule for `mul_p` raised `TypeError:
    mul() got an unexpected keyword argument 'out_dtype'`.
    """
    out = quax.quaxify(lambda a: a * a)(tagged)

    assert jnp.allclose(out.leaf.array, jnp.array([1.0, 4.0, 9.0]))


def test_primitive_params_the_function_declares_are_forwarded(tagged):
    """`integer_pow_p` carries `y`, and the mapping entry asks for it."""
    out = quax.quaxify(lambda a: a**3)(tagged)

    assert out.units == ((meters, 3),)
    assert jnp.allclose(out.leaf.array, jnp.array([1.0, 8.0, 27.0]))


def test_a_non_hijax_result_is_returned_unwrapped(tagged):
    """A rule may legitimately return a plain array; do not wrap it."""
    out = quax.quaxify(jnp.sum)(tagged)

    assert not isinstance(out, Tagged)
    assert jnp.allclose(out, jnp.asarray(6.0))


def test_generated_rules_are_named_after_the_primitive_and_operands():
    """Tracebacks and plum's ambiguity errors show these names."""
    names = {
        m.implementation.__name__ for m in quax._dispatch._rules[lax.mul_p].methods
    }

    assert "mul_tagged_tagged" in names
    assert "mul_tagged_arraylike" in names
    assert "mul_arraylike_tagged" in names


def test_a_multiple_results_primitive_is_refused():
    import jax.extend.core as jexc

    primitive = jexc.Primitive("test_multi")
    primitive.multiple_results = True

    with pytest.raises(NotImplementedError, match="multiple results"):
        register_rules(Tagged, {primitive: lambda q: q})


def test_grad_still_gets_the_inverted_cotangent_units(tagged):
    """The mechanism must not flatten what the hijax type says about cotangents."""

    def loss(value):
        return quax.quaxify(lambda a: jnp.sum(a * a))(value)

    grad = jax.grad(loss)(tagged)

    assert grad.units == ((meters, -1),)
    assert jnp.allclose(grad.leaf.array, jnp.array([2.0, 4.0, 6.0]))
