"""Unit propagation through a `diffrax` Runge-Kutta solver.

`diffrax` steps its explicit RK stages inside `equinox.internal`'s buffered
loop, which is implemented with `jax.custom_vjp` -- so this exercises
`quax._trace._QuaxTrace.process_custom_vjp_call`.

`diffrax.diffeqsolve` itself is *not* used: it allocates its `SaveAt` output
buffer with `jnp.full`, which Quax never sees, so the buffer is a plain array
and `cond_quax` then rejects the branch whose other side carries a `Value`.
Driving `solver.step` directly avoids that buffer while keeping the numerics.

This module only covers the forward direction (unit propagation through
`quaxify`). It does *not* attempt reverse-mode AD: quax's `custom_vjp`
support does not round-trip equinox's `filter_custom_vjp` perturbation
metadata, so a flag equinox expects to be a static Python `bool` arrives
traced, and `jax.grad` through this call raises `TracerBoolConversionError`
inside equinox itself, at `equinox/internal/_loop/checkpointed.py:766`
(reached via `_checkpointed_while_loop_bwd`). This reproduces with a bare
`eqxi.while_loop(kind="checkpointed")` under `quaxify` -- no diffrax and no
`Unitful` required -- and is not a regression: ordinary `custom_vjp`
reverse-mode already works (see `tests/unit/test_custom_vjp.py`); this gap
is specific to `filter_custom_vjp`. See task-5-report.md's fix-round-2
section for the full writeup.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from equinox.internal._loop.common import maybe_set_p, select_if_vmap_p
from jaxtyping import ArrayLike

import quax
from quax.examples.unitful import meters, Unitful


# `diffrax` lives in the `test-integration` dependency group, not `tests`, so a
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
# That provenance argument justifies *reusing the value* -- it says nothing
# about what units to reattach, and there is no honest way to recover that
# here: `_step_n`'s while-loop body is traced once (by
# `quax._primitives.while_quax`) and replayed for every runtime stage, so by
# the time a buffer-derived plain value reaches this `add` rule, the units it
# had before going into the buffer are gone, and nothing at this call site
# names which buffer it came from or what was written there. An earlier
# version of this module tried to reconstruct those units from bookkeeping
# maintained by the `maybe_set_p` rule below -- comparing the accumulator's
# units against whatever was most recently written to *any* buffer -- but
# that check never actually consulted `y` (the operand whose units were
# erased), so it could neither reject a mismatch between `y` and *its own*
# buffer nor avoid false positives from an unrelated one. It only happened to
# agree with the correct answer for this module's one ODE (`dy = rate * y`,
# where `k`'s units equal `y`'s units iff `rate` is dimensionless). Given
# that, this module makes no attempt at a rejection test -- see the comment
# below `test_diffrax_step_propagates_units` -- and unconditionally trusts the
# accumulator's units, same as the brief originally specified.
#
# Because `quax.register` is a process-global registry, once this module is
# imported this rule accepts *any* `Unitful + plain array` add, for the rest
# of the process -- silently invalidating the invariant documented in
# `docs/examples/custom_rules.ipynb` ("Bad example 2": adding a plain array to
# a `Unitful` should raise, because no such rule exists). That invalidation is
# real but bounded: it only occurs once diffrax is installed (this module
# self-skips via `importorskip` otherwise) and only within a process that also
# imports this module. CI's `integration` job runs `pytest tests/integration`
# only -- the notebook and `tests/usage/test_unitful.py` never run in that
# process -- and every other CI job never installs diffrax, so this module is
# never imported there either. The only place the two can collide is a local
# `uv run --group test-integration pytest tests -q`, a deliberate, explicit
# invocation -- not something that can happen by accident in CI.
# ---------------------------------------------------------------------------


@quax.register(jax.lax.add_p)
def _(x: Unitful, y: ArrayLike, **kw: Any) -> Unitful:
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
    # side re-attach units via the `add`/`select` rules above.
    return maybe_set_p.bind(pred, xs, x.array, *rest, **kw)


# ---------------------------------------------------------------------------
# The system: dy/dt = rate * y, with y in metres and `rate` dimensionless
# (time is carried as plain floats, so `rate` must be per-step, not per-second).
# ---------------------------------------------------------------------------

N_STEPS = 5
DT = 0.1


def _step_n(y0, rate):
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


# No `test_..._rejects_inconsistent_units` here. The `add_p` rule above
# cannot honestly enforce dimensional consistency -- see the comment block
# above it -- so this suite asserts unit *propagation* only, not rejection.
# A real rejection test would need `maybe_set_p` to record units keyed by
# buffer identity, and `add_p` to consult them only when `y` is provably a
# read-back of that specific buffer; nothing here establishes that link.


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


# No `test_diffrax_step_grad` here. quax's `custom_vjp` support does not
# round-trip equinox's `filter_custom_vjp` perturbation metadata: a flag
# equinox expects to be a static Python `bool` arrives traced, and
# `jax.grad` through this call raises `TracerBoolConversionError` inside
# equinox itself, at `equinox/internal/_loop/checkpointed.py:766` (reached
# via `_checkpointed_while_loop_bwd`) -- no quax frame appears in the
# failing path. It reproduces with a bare `eqxi.while_loop(kind=
# "checkpointed")` under `quaxify`, no diffrax or `Unitful` required, and
# is not a regression (ordinary `custom_vjp` reverse-mode already works,
# see `tests/unit/test_custom_vjp.py`). Landing this forward-only and
# documenting the gap is the maintainer-approved plan -- not something to
# paper over with `xfail`. See task-5-report.md's fix-round-2 section.
