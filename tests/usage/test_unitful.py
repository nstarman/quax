import jax
import jax.numpy as jnp
import pytest

import quax
from quax.examples.unitful import meters, seconds, Unitful


def test_integer_pow_exponentiates_array():
    """`integer_pow` must raise the array to the power, not just scale the units.

    Regression: the handler scaled `units` by `y` but returned `x.array`
    unchanged, so `Unitful(a, {m: 1}) ** 2` gave the array `a` (not `a ** 2`)
    with units `m ** 2` — a silent wrong value.
    """
    x = Unitful(jnp.array([2.0, 3.0]), meters)

    out = quax.quaxify(lambda a: a**2)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 2}
    assert jnp.array_equal(out.array, jnp.array([4.0, 9.0]))


def test_integer_pow_units_and_values_agree():
    """Cube a velocity-like quantity: both value and units are exponentiated."""
    v = Unitful(jnp.array(2.0), {meters: 1, seconds: -1})

    out = quax.quaxify(lambda a: a**3)(v)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 3, seconds: -3}
    assert jnp.array_equal(out.array, jnp.array(8.0))


def test_copy_preserves_units():
    """`jnp.copy` keeps the units of its operand."""
    x = Unitful(jnp.array([2.0, 3.0]), meters)

    out = quax.quaxify(jnp.copy)(x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.array([2.0, 3.0]))


def test_select_n_requires_matching_units():
    """`jnp.where` between two Unitfuls keeps units, and rejects a mismatch."""
    x = Unitful(jnp.array([2.0, 3.0]), meters)
    y = Unitful(jnp.array([4.0, 5.0]), meters)
    pred = jnp.array([True, False])

    out = quax.quaxify(jnp.where)(pred, x, y)
    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.array([2.0, 5.0]))

    bad = Unitful(jnp.array([4.0, 5.0]), seconds)
    with pytest.raises(ValueError, match="units"):
        quax.quaxify(jnp.where)(pred, x, bad)


def test_select_n_single_case_is_a_no_op():
    """`select_n` with one case has nothing to disagree with, and passes through."""
    x = Unitful(jnp.array([2.0, 3.0]), meters)

    out = quax.quaxify(lambda w, a: jax.lax.select_n(w, a))(jnp.array(0), x)

    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.array([2.0, 3.0]))


def test_select_n_three_cases_checks_every_pair():
    """A mismatch in the *third* case is caught, not just the second."""
    a = Unitful(jnp.array([1.0]), meters)
    b = Unitful(jnp.array([2.0]), meters)
    bad = Unitful(jnp.array([3.0]), seconds)

    which = jnp.array(0)
    out = quax.quaxify(jax.lax.select_n)(which, a, b, Unitful(jnp.array([4.0]), meters))
    assert isinstance(out, Unitful)
    assert out.units == {meters: 1}

    # the message names the two mismatching unit dicts, as `add_p`'s does
    with pytest.raises(ValueError, match=r"units \{m: 1\} and \{s: 1\}"):
        quax.quaxify(jax.lax.select_n)(which, a, b, bad)


def _value(x):
    return x.array if isinstance(x, Unitful) else x


def test_sub_requires_matching_units():
    out = quax.quaxify(jax.lax.sub)(
        Unitful(jnp.array(3.0), meters), Unitful(jnp.array(1.0), meters)
    )
    assert out.units == {meters: 1}
    assert out.array == 2.0
    with pytest.raises(ValueError, match="units"):
        quax.quaxify(jax.lax.sub)(
            Unitful(jnp.array(3.0), meters), Unitful(jnp.array(1.0), seconds)
        )


def test_div_combines_units():
    m, s = Unitful(jnp.array(6.0), meters), Unitful(jnp.array(2.0), seconds)
    div = quax.quaxify(jax.lax.div)
    assert div(m, s).units == {meters: 1, seconds: -1}
    assert div(m, s).array == 3.0
    assert div(m, m).units == {}
    assert div(m, jnp.array(2.0)).units == {meters: 1}
    assert div(jnp.array(2.0), s).units == {seconds: -1}


def test_transpose_keeps_units():
    x = Unitful(jnp.arange(6.0).reshape(2, 3), meters)
    out = quax.quaxify(jnp.transpose)(x)
    assert out.units == {meters: 1}
    assert jnp.array_equal(out.array, jnp.arange(6.0).reshape(2, 3).T)


# From JAX 0.12, `jnp.hypot` is built on the `custom_jvp` function `_hypot`, which
# the example registers.
needs_hypot_custom_jvp = pytest.mark.skipif(
    not isinstance(
        getattr(jax._src.numpy.ufuncs, "_hypot", None),  # pyright: ignore[reportAttributeAccessIssue]
        jax.custom_jvp,
    ),
    reason="jnp.hypot is not built on a custom_jvp before JAX 0.12",
)
_a = jnp.array([[3.0, 6.0], [9.0, 12.0]])
_b = jnp.array([[4.0, 8.0], [12.0, 16.0]])


@needs_hypot_custom_jvp
def test_hypot_keeps_units():
    out = quax.quaxify(jnp.hypot)(Unitful(_a, meters), Unitful(_b, meters))
    assert out.units == {meters: 1}
    assert jnp.allclose(out.array, jnp.hypot(_a, _b))


@needs_hypot_custom_jvp
def test_hypot_under_jit():
    x, y = Unitful(_a, meters), Unitful(_b, meters)
    assert jnp.allclose(jax.jit(quax.quaxify(jnp.hypot))(x, y).array, jnp.hypot(_a, _b))
    assert jnp.allclose(quax.quaxify(jax.jit(jnp.hypot))(x, y).array, jnp.hypot(_a, _b))


@needs_hypot_custom_jvp
def test_hypot_inner_vmap_misaligned():
    expected = jax.vmap(jnp.hypot, in_axes=(0, 1))(_a, _b)
    hypot = quax.quaxify(jax.vmap(jnp.hypot, in_axes=(0, 1)))
    out = hypot(Unitful(_a, meters), Unitful(_b, meters))
    assert out.units == {meters: 1}
    assert jnp.allclose(out.array, expected)


@needs_hypot_custom_jvp
def test_hypot_outer_grad():
    y = Unitful(jnp.array(4.0), meters)
    grad = jax.grad(lambda t: quax.quaxify(jnp.hypot)(Unitful(t, meters), y).array)
    assert jnp.allclose(grad(3.0), 0.6)


@needs_hypot_custom_jvp
def test_hypot_inner_grad():
    x, y = Unitful(jnp.array(3.0), meters), Unitful(jnp.array(4.0), meters)
    assert jnp.allclose(_value(quax.quaxify(jax.grad(jnp.hypot))(x, y)), 0.6)


@needs_hypot_custom_jvp
def test_hypot_units_mismatch():
    x, y = Unitful(jnp.array(3.0), meters), Unitful(jnp.array(4.0), seconds)
    with pytest.raises(ValueError, match="units"):
        quax.quaxify(jnp.hypot)(x, y)
    with pytest.raises(ValueError, match="units"):
        jax.jit(quax.quaxify(jnp.hypot))(x, y)
