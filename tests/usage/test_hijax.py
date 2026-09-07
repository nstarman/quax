"""`quax.examples.hijax`: a Quax value whose leaf carries a hijax type.

The point of the example is what a pure-Quax type cannot do -- put the units in
the JAX type, so they appear in jaxprs and so a cotangent can carry *inverse*
units -- so that is what these tests pin down. They also serve as the canary for
`jax.experimental.hijax`, which is experimental and renames things: when JAX
moves, these fail before anything ships.
"""

from typing import Final

import jax
import jax.numpy as jnp
import pytest
from packaging.version import Version

import quax
from quax._compat import JAX_VERSION


hijax = pytest.importorskip(
    "quax.examples.hijax", reason="jax.experimental.hijax is unavailable"
)

from quax.examples.hijax import MAPPED, Quantity, Unitful  # noqa: E402
from quax.examples.unitful import kilograms, meters, seconds  # noqa: E402


PER_METER = ((meters, -1),)
METERS = ((meters, 1),)
JOULES = ((kilograms, 1), (meters, 2), (seconds, -2))


def kinetic_energy(mass, velocity):
    """Unmodified array code: nothing here mentions units or hijax."""
    return 0.5 * mass * velocity**2


@pytest.fixture
def length():
    return Unitful(jnp.array([1.0, 2.0, 3.0]), meters)


# Consuming a hijax value under `jit` reports a leaked tracer that is not one.
# A JAX regression, reproducible with one `HiPrim` and no Quax at all: 0.10.1
# and earlier are fine, 0.10.2 and 0.11.0 report it, and 0.11.1 onwards are fine
# again. Bracketed by running the same repro against each release.
_FALSE_LEAK_REPORT: Final = Version("0.10.2") <= JAX_VERSION < Version("0.11.1")


@pytest.fixture
def no_tracer_leak_check():
    """Turn off `JAX_CHECK_TRACER_LEAKS`, but only where JAX gets it wrong.

    The suite otherwise runs with the check on (see `[tool.pytest_env]`), and on
    a JAX without the regression this fixture does nothing -- so if the false
    report ever comes back on a fixed version, these tests say so rather than
    staying quiet.
    """
    if not _FALSE_LEAK_REPORT:
        yield
        return
    key = "jax_check_tracer_leaks"
    previous = jax.config.values[key]
    jax.config.update(key, False)
    yield
    jax.config.update(key, previous)


def _sum_of_squares(value):
    """A dimensionless scalar built from a `Unitful`, for differentiating."""
    return quax.quaxify(lambda a: jnp.sum(a * a))(value).array


def test_units_propagate_through_unmodified_code():
    mass = Unitful(jnp.asarray(2.0), kilograms)
    velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

    out = quax.quaxify(kinetic_energy)(mass, velocity)

    assert isinstance(out, Unitful)
    assert out.units == JOULES
    assert jnp.array_equal(out.array, jnp.asarray(9.0))


def test_units_appear_in_the_jaxpr():
    """The headline difference from a pytree type: the units are in the type."""
    mass = Unitful(jnp.asarray(2.0), kilograms)
    velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

    jaxpr = jax.jit(quax.quaxify(kinetic_energy)).trace(mass, velocity).jaxpr

    assert "q[]{kg m^2 s^-2}" in str(jaxpr)


def test_jit_agrees_with_eager(no_tracer_leak_check):
    mass = Unitful(jnp.asarray(2.0), kilograms)
    velocity = Unitful(jnp.asarray(3.0), {meters: 1, seconds: -1})

    out = jax.jit(quax.quaxify(kinetic_energy))(mass, velocity)

    assert out.units == JOULES
    assert jnp.array_equal(out.array, jnp.asarray(9.0))


def test_cotangent_units_are_inverted(length):
    """d(dimensionless)/d(length) is measured per metre, and says so.

    This is the one thing `quax.examples.unitful` cannot do: there a cotangent
    reuses the primal's pytree structure, static metadata included, so it comes
    back in metres. Here the units live in the leaf's *type*, which hijax lets
    `to_ct_aval` change.
    """
    grad = jax.grad(_sum_of_squares)(length)

    assert isinstance(grad, Unitful)
    assert grad.units == PER_METER
    assert jnp.allclose(grad.array, jnp.array([2.0, 4.0, 6.0]))


def test_jit_of_grad_keeps_the_inverted_units(no_tracer_leak_check, length):
    grad = jax.jit(jax.grad(_sum_of_squares))(length)

    assert grad.units == PER_METER
    assert jnp.allclose(grad.array, jnp.array([2.0, 4.0, 6.0]))


def test_tangent_units_match_the_primal(length):
    """A perturbation of a length is a length, unlike a cotangent."""
    tangent_in = Unitful(jnp.ones(3), meters)

    primal, tangent = jax.jvp(quax.quaxify(lambda a: a * a), (length,), (tangent_in,))

    assert primal.units == ((meters, 2),)
    assert tangent.units == ((meters, 2),)
    assert jnp.allclose(tangent.array, jnp.array([2.0, 4.0, 6.0]))


def test_vmap_over_a_unitful_needs_a_mapping_spec():
    xs = Unitful(jnp.arange(6.0).reshape(3, 2), meters)

    out = jax.vmap(
        quax.quaxify(lambda a: a * a), in_axes=MAPPED, out_axes=MAPPED, axis_size=3
    )(xs)

    assert out.units == ((meters, 2),)
    assert jnp.allclose(out.array, jnp.arange(6.0).reshape(3, 2) ** 2)


def test_vmap_inside_quaxify_needs_no_spec():
    xs = Unitful(jnp.arange(6.0).reshape(3, 2), meters)

    out = quax.quaxify(jax.vmap(lambda a: a * a))(xs)

    assert out.units == ((meters, 2),)
    assert jnp.allclose(out.array, jnp.arange(6.0).reshape(3, 2) ** 2)


def test_vmap_of_grad():
    xs = Unitful(jnp.arange(6.0).reshape(3, 2), meters)

    grad = jax.vmap(
        jax.grad(_sum_of_squares), in_axes=MAPPED, out_axes=MAPPED, axis_size=3
    )(xs)

    assert grad.units == PER_METER
    assert jnp.allclose(grad.array, 2 * jnp.arange(6.0).reshape(3, 2))


def test_scan_carries_a_unitful(no_tracer_leak_check, length):
    """`scan` asks the type for its leading axis rather than taking a spec."""

    def body(carry, _):
        return quax.quaxify(lambda a, b: a + b)(carry, length), None

    carry, _ = jax.lax.scan(body, length, None, length=3)

    assert carry.units == METERS
    assert jnp.allclose(carry.array, jnp.array([4.0, 8.0, 12.0]))


def test_adding_mismatched_units_is_a_trace_time_error(length):
    seconds_value = Unitful(jnp.array([1.0, 2.0, 3.0]), seconds)

    with pytest.raises(TypeError, match="units differ"):
        quax.quaxify(lambda a, b: a + b)(length, seconds_value)


def test_an_unregistered_primitive_refuses_to_materialise(length):
    """No rule for `sin`, and `materialise` refuses, so units cannot leak away."""
    with pytest.raises(ValueError, match="Refusing to materialise"):
        quax.quaxify(jnp.sin)(length)


def test_the_quantity_is_the_single_pytree_leaf(length):
    """A `Unitful` is a pytree; the hijax value inside it is not."""
    leaves = jax.tree.leaves(length)

    assert len(leaves) == 1
    assert isinstance(leaves[0], Quantity)
    assert jax.tree.leaves(leaves[0]) == [leaves[0]]


def test_units_are_dropped_only_on_request(length):
    assert length.units == METERS
    assert jnp.array_equal(length.array, jnp.array([1.0, 2.0, 3.0]))
    assert jax.typeof(length.leaf).units == METERS


def test_dimensionless_units_are_canonical():
    """`m / m` is dimensionless, not `m^0`, so the types compare equal."""
    per_meter = Unitful(jnp.asarray(2.0), {meters: -1})
    length = Unitful(jnp.asarray(3.0), meters)

    out = quax.quaxify(lambda a, b: a * b)(per_meter, length)

    assert out.units == ()
