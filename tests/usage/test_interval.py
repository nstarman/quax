import jax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.interval import Interval


def bounds(x: Interval) -> tuple[float, float]:
    return float(x.lo), float(x.hi)


def test_bounds_propagate_through_arithmetic():
    """The result brackets the same computation on any values inside the inputs."""
    x = Interval(1.0, 2.0)
    y = Interval(10.0, 20.0)

    assert bounds(quax.quaxify(lambda a, b: a + b)(x, y)) == (11.0, 22.0)
    assert bounds(quax.quaxify(lambda a, b: a - b)(x, y)) == (-19.0, -8.0)
    assert bounds(quax.quaxify(lambda a: -a)(x)) == (-2.0, -1.0)


def test_subtraction_pairs_opposite_bounds():
    """`hi` of a difference uses the *lower* bound of the subtrahend.

    Getting this backwards is the classic interval bug, and it produces bounds
    that are too narrow -- i.e. wrong rather than merely pessimistic.
    """
    out = quax.quaxify(lambda a, b: a - b)(Interval(0.0, 1.0), Interval(0.0, 1.0))

    assert bounds(out) == (-1.0, 1.0)


def test_multiplication_takes_the_extreme_corner():
    """With mixed signs the extreme is not at a matching pair of endpoints."""
    out = quax.quaxify(lambda a, b: a * b)(Interval(-2.0, 3.0), Interval(-5.0, 1.0))

    # corners: 10, -2, -15, 3 -> the true range is [-15, 10]
    assert bounds(out) == (-15.0, 10.0)


def test_multiplying_by_a_negative_scalar_swaps_the_bounds():
    """`Interval(lo * y, hi * y)` would be inverted for negative `y`."""
    out = quax.quaxify(lambda a: -2.0 * a)(Interval(-1.0, 2.0))

    assert bounds(out) == (-4.0, 2.0)


def test_even_power_dips_to_zero_across_zero():
    """The endpoints alone do not see that `x**2` reaches 0 inside the bracket."""
    assert bounds(quax.quaxify(lambda a: a**2)(Interval(-1.0, 2.0))) == (0.0, 4.0)
    # ... but not when the bracket stays on one side of zero
    assert bounds(quax.quaxify(lambda a: a**2)(Interval(1.0, 2.0))) == (1.0, 4.0)
    assert bounds(quax.quaxify(lambda a: a**2)(Interval(-3.0, -2.0))) == (4.0, 9.0)


def test_odd_power_is_monotonic():
    assert bounds(quax.quaxify(lambda a: a**3)(Interval(-1.0, 2.0))) == (-1.0, 8.0)


def test_negative_power_is_refused():
    """A reciprocal is unbounded when the bracket contains zero."""
    with pytest.raises(NotImplementedError, match="unbounded"):
        quax.quaxify(lambda a: a**-1)(Interval(-1.0, 2.0))


def test_the_dependency_problem_is_real_and_sound():
    """`x * x` is wider than `x ** 2`, and both contain the true range.

    Pinned because it is documented behaviour rather than a bug: `mul` cannot
    see that its two operands are the same value. If a future rule set makes
    these agree, this test should be updated rather than deleted.
    """
    x = Interval(-1.0, 2.0)

    via_mul = bounds(quax.quaxify(lambda a: a * a)(x))
    via_pow = bounds(quax.quaxify(lambda a: a**2)(x))

    assert via_mul == (-2.0, 4.0)
    assert via_pow == (0.0, 4.0)
    # sound: the true range [0, 4] sits inside both
    assert via_mul[0] <= 0.0 and via_mul[1] >= 4.0


def test_reduction_and_broadcasting():
    x = Interval(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

    # lo sums to 1 + 2, hi sums to 3 + 4
    assert bounds(quax.quaxify(jnp.sum)(x)) == (3.0, 7.0)
    assert bounds(quax.quaxify(lambda a: jnp.sum(a + 1.0))(x)) == (5.0, 9.0)


def test_jit_agrees_with_eager():
    x = Interval(0.9, 1.1)
    f = lambda a: a**3 - 2.0 * a + 1.0  # noqa: E731

    assert bounds(jax.jit(quax.quaxify(f))(x)) == bounds(quax.quaxify(f)(x))


def test_default_handles_primitives_that_only_move_elements():
    """One `default` covers a whole class, exactly, without a rule each."""
    x = Interval(jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([2.0, 4.0, 6.0, 8.0]))

    sliced = quax.quaxify(lambda a: a[1:3])(x)
    assert sliced.lo.tolist() == [2.0, 3.0]
    assert sliced.hi.tolist() == [4.0, 6.0]

    flipped = quax.quaxify(jnp.flip)(x)
    assert flipped.lo.tolist() == [4.0, 3.0, 2.0, 1.0]

    assert bounds(quax.quaxify(jnp.max)(x)) == (4.0, 8.0)
    assert bounds(quax.quaxify(jnp.min)(x)) == (1.0, 2.0)

    reshaped = quax.quaxify(lambda a: a.reshape(2, 2))(x)
    assert reshaped.lo.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_default_handles_increasing_functions():
    """An increasing function maps the bounds straight across."""
    x = Interval(1.0, 4.0)

    lo, hi = bounds(quax.quaxify(jnp.sqrt)(x))
    assert (lo, hi) == (1.0, 2.0)
    lo, hi = bounds(quax.quaxify(jnp.exp)(x))
    assert lo == pytest.approx(float(jnp.exp(jnp.asarray(1.0))))
    assert hi == pytest.approx(float(jnp.exp(jnp.asarray(4.0))))


def test_default_refuses_a_non_monotone_primitive():
    """Guessing here would be unsound, not merely loose.

    Mapping `sin` over the endpoints of `[0, 2*pi]` gives `[0, 0]` -- a
    confident claim that the value is exactly zero, when it ranges over all of
    `[-1, 1]`. So an unknown primitive raises rather than falling back.
    """
    with pytest.raises(ValueError, match="no rule for `sin`"):
        quax.quaxify(jnp.sin)(Interval(0.0, 1.0))

    # and the message says what it *does* cover, so the fix is obvious
    with pytest.raises(ValueError, match="Handled by default:"):
        quax.quaxify(jnp.sin)(Interval(0.0, 1.0))


def test_materialise_refuses_when_called_directly():
    """A silently collapsed bound would be a wrong answer, not a lost type."""
    with pytest.raises(ValueError, match="Refusing to materialise"):
        Interval(0.0, 1.0).materialise()


def test_mismatched_bound_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        Interval(jnp.zeros(3), jnp.zeros(4))


def test_width_and_midpoint():
    x = Interval(jnp.array([1.0, -2.0]), jnp.array([3.0, 2.0]))

    assert jnp.array_equal(x.width, jnp.array([2.0, 4.0]))
    assert jnp.array_equal(x.midpoint, jnp.array([2.0, 0.0]))


def test_inverted_bounds_are_not_checked():
    """`lo <= hi` depends on the values, so no JAX type can enforce it.

    Documented rather than fixed: the check would have to be a runtime one.
    """
    inverted = Interval(3.0, 1.0)

    assert bounds(inverted) == (3.0, 1.0)
