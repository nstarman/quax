"""Unit propagation through a `diffrax` Runge-Kutta solver.

`diffrax` steps its explicit RK stages inside `equinox.internal`'s buffered
loop, which is implemented with `jax.custom_vjp` -- so this exercises
`quax._trace._QuaxTrace.process_custom_vjp_call`.

`diffrax.diffeqsolve` itself is *not* used: it allocates its `SaveAt` output
buffer with `jnp.full`, which Quax never sees, so the buffer is a plain array
and `cond_quax` then rejects the branch whose other side carries a `Value`.
Driving `solver.step` directly avoids that buffer while keeping the numerics.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from equinox.internal._loop.common import maybe_set_p, select_if_vmap_p
from jaxtyping import ArrayLike

import quax
from quax.examples.unitful import meters, seconds, Unitful


# `diffrax` lives in the `integration` dependency group, not `tests`, so a
# default `pytest` run skips this module rather than erroring at collection.
diffrax = pytest.importorskip("diffrax")


# ---------------------------------------------------------------------------
# Bridging rules for equinox's scratch buffers.
#
# `equinox.internal`'s loops pre-allocate their stage buffers with `jnp.zeros`,
# which carries no `Unitful` operand for Quax to dispatch on. A quantity
# written into such a buffer therefore comes back out dimensionless, and the
# arithmetic that consumes it mixes a plain array with a `Unitful`.
#
# These rules re-attribute the `Unitful` operand's units at that boundary. They
# are deliberately *not* in `quax.examples.unitful`: "a plain array may be added
# to a quantity" is unsound as a general units rule, and is only defensible here
# because the plain operand provably originated as a `Unitful` (or as the
# literal `0` that diffrax substitutes when zeroing an unused FSAL stage).
#
# That provenance argument only tells us the *value* is safe to reuse -- it
# says nothing about whether the units silently reattached to it are the
# *right* units. `_step_n`'s while-loop body is traced exactly once (by
# `quax._primitives.while_quax`) and then replayed for every runtime stage, so
# a naive `add` rule that just copies the accumulator's units onto whatever
# comes out of the buffer can never observe a stage whose computed units
# differ from the accumulator's -- there is no separate Quax dispatch per
# dynamic iteration to catch it at. `_buffer_units` closes that gap: the same
# buffer is written every trace by `maybe_set_p` with the *actual* computed
# units of a stage (`rate * y`), and `_step_n` calls `solver.step` once per
# macro-step from ordinary (unquaxified) Python, so those units are available,
# module-global, by the time the *next* macro-step's `add` rule runs and reads
# that buffer back. Comparing against them (skipping the very first add, which
# only ever sees the zero placeholder) is exactly the dimensional check
# `y + dt * f(t, y)` needs, without requiring per-iteration dispatch.
# ---------------------------------------------------------------------------

# Units most recently written into equinox's stage buffer, established once
# per traced while-loop body and reused across the `_step_n` macro-steps that
# read it back. Reset at the top of `_step_n` so state never leaks between
# separate calls -- including between separate tests, since `quax.register`
# is a global registry and these rules run in every test in this session.
_buffer_units: dict[Any, int] | None = None


@quax.register(jax.lax.add_p)
def _(x: Unitful, y: ArrayLike, **kw: Any) -> Unitful:
    global _buffer_units
    if _buffer_units is not None and _buffer_units != x.units:
        raise ValueError(
            f"Cannot add two arrays with units {x.units} and {_buffer_units}."
        )
    return Unitful(jax.lax.add_p.bind(x.array, y, **kw), x.units)


@quax.register(select_if_vmap_p)
def _(pred: ArrayLike, *cases: Unitful | ArrayLike, **kw: Any) -> Any:
    unitful = [c.units for c in cases if isinstance(c, Unitful)]
    if not unitful:
        return select_if_vmap_p.bind(pred, *cases, **kw)
    if any(u != unitful[0] for u in unitful[1:]):
        raise ValueError(f"Cannot select between arrays with units {unitful}.")
    arrays = [c.array if isinstance(c, Unitful) else c for c in cases]
    return Unitful(select_if_vmap_p.bind(pred, *arrays, **kw), unitful[0])


@quax.register(maybe_set_p)
def _(pred: ArrayLike, xs: ArrayLike, x: Unitful, *rest: Any, **kw: Any) -> Any:
    # The buffer `xs` is a plain array; store the magnitude and let the read
    # side re-attach units via the `add`/`select` rules above. Record the
    # units actually being written so the *next* macro-step's `add` rule can
    # check them (see `_buffer_units` above).
    global _buffer_units
    _buffer_units = x.units
    return maybe_set_p.bind(pred, xs, x.array, *rest, **kw)


# ---------------------------------------------------------------------------
# The system: dy/dt = rate * y, with y in metres and `rate` dimensionless
# (time is carried as plain floats, so `rate` must be per-step, not per-second).
# ---------------------------------------------------------------------------

N_STEPS = 5
DT = 0.1


def _step_n(y0, rate):
    global _buffer_units
    _buffer_units = None  # fresh state per call -- see `_buffer_units` above
    term = diffrax.ODETerm(lambda t, y, args: args * y)
    solver = diffrax.Tsit5()
    state = solver.init(term, 0.0, DT, y0, rate)
    y = y0
    for i in range(N_STEPS):
        y, _, _, state, _ = solver.step(
            term, i * DT, (i + 1) * DT, y, rate, state, made_jump=False
        )
    return y


def test_diffrax_step_propagates_units():
    """Units survive a diffrax RK integration, and the values are unchanged."""
    y0_val, rate_val = jnp.array([1.0]), jnp.asarray(-0.5)
    expected = _step_n(y0_val, rate_val)

    got = quax.quaxify(_step_n)(Unitful(y0_val, meters), Unitful(rate_val, {}))

    assert isinstance(got, Unitful)
    assert got.units == {meters: 1}
    assert jnp.allclose(got.array, expected)


def test_diffrax_step_exercises_custom_vjp():
    """The integration really does route through `process_custom_vjp_call`."""
    from quax import _trace

    calls = 0
    original = _trace._QuaxTrace.process_custom_vjp_call

    def counting(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    _trace._QuaxTrace.process_custom_vjp_call = counting
    try:
        quax.quaxify(_step_n)(
            Unitful(jnp.array([1.0]), meters), Unitful(jnp.asarray(-0.5), {})
        )
    finally:
        _trace._QuaxTrace.process_custom_vjp_call = original

    assert calls > 0


def test_diffrax_step_rejects_inconsistent_units():
    """A rate with dimensions makes `y + dt * f(t, y)` dimensionally invalid."""
    with pytest.raises(ValueError, match="Cannot add"):
        quax.quaxify(_step_n)(
            Unitful(jnp.array([1.0]), meters),
            Unitful(jnp.asarray(-0.5), seconds),
        )


@pytest.mark.xfail(
    reason=(
        "Plain-JAX `jax.grad(_step_n)` succeeds on the same inputs, so this is "
        "not a diffrax limitation. Under `quaxify`, `custom_vjp`'s fwd rule "
        "(`_custom_vjp_fwd_wrap` in `quax._trace`) traces "
        "`_checkpointed_while_loop_fwd`, whose residual-saving path hits a "
        "`select_n_p` dispatch over a mix of `Unitful` and plain-array cases "
        "that this module has no bridging rule for; the underlying `ValueError` "
        "is 'Refusing to materialise Unitful array.'. That fwd/residual path is "
        "equinox-internal machinery, not something a test-module bridging rule "
        "can patch -- it is the kind of gap flagged as out of scope in "
        "task-5-brief.md's follow-up items 2 and 3 (symbolic-zero promotion and "
        "residual forwarding in `quax._trace`)."
    ),
    strict=True,
)
def test_diffrax_step_grad():
    """Reverse-mode AD through the quaxified, unit-carrying integration."""
    y0_val, rate_val = jnp.array([1.0]), jnp.asarray(-0.5)
    expected = jax.grad(lambda y: _step_n(y, rate_val).sum())(y0_val)

    got = jax.grad(
        lambda y: quax.quaxify(_step_n)(y, Unitful(rate_val, {})).array.sum()
    )(Unitful(y0_val, meters))

    assert jnp.allclose(got.array, expected)
