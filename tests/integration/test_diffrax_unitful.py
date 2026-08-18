"""Unit propagation through a `diffrax` Runge-Kutta solver.

`diffrax` steps its explicit RK stages inside `equinox.internal`'s buffered
loop, which is built on `jax.custom_vjp` -- so this exercises
`quax._trace._QuaxTrace.process_custom_vjp_call`.

`diffrax.diffeqsolve` is not used *here*: it allocates its `SaveAt` buffer
with `jnp.full`, which Quax never sees, so `Unitful` comes back out of it
dimensionless -- and `Unitful` refuses to materialise, which is the whole
point of it. Driving `solver.step` keeps the numerics without the buffer.
`test_diffrax_dense.py` runs the full `diffeqsolve` with a type that does
materialise.

Forward direction only, and for one reason: `jax.grad` reaches a `select_n`
over mixed plain/`Unitful` operands -- the same buffer erasure the `add_p`
rules below work around -- where `Unitful` refuses to materialise, which is
what it is for. Reverse-mode itself is fine, including through
`equinox.internal.while_loop`; see `test_diffrax_dense.py` and
`tests/unit/test_custom_vjp.py`.
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


# Bridging rules for equinox's scratch buffers.
#
# `equinox.internal`'s loops pre-allocate their stage buffers with `jnp.zeros`,
# so a `Unitful` written into one comes back out dimensionless and then meets a
# `Unitful` in the arithmetic that consumes it. These rules reattach the
# `Unitful` operand's units at that boundary.
#
# They are deliberately not in `quax.examples.unitful`: "a plain array may be
# added to a quantity" is unsound in general, and is only defensible here
# because the plain operand provably came out of a buffer a `Unitful` went
# into. The units it had going in are not recoverable at this call site, so
# these rules trust the `Unitful` operand rather than checking it -- which is
# why this module tests unit *propagation* and not unit *rejection*.
#
# `quax.register` is process-global, so once this module is imported the
# `add_p` rule accepts any `Unitful + plain` add, invalidating the invariant
# in `docs/examples/custom_rules.ipynb` ("Bad example 2"). CI never hits that:
# the integration job runs `tests/integration` alone and no other job installs
# diffrax. A local `pytest` covering both directories at once does.


@quax.register(jax.lax.add_p)
def add_unitful_array_like(x: Unitful, y: ArrayLike, **kw: Any) -> Unitful:
    return Unitful(jax.lax.add_p.bind(x.array, y, **kw), x.units)


@quax.register(select_if_vmap_p)
def select_if_vmap_unitful(
    pred: ArrayLike, *cases: Unitful | ArrayLike, **kw: Any
) -> Any:
    unitful = [c.units for c in cases if isinstance(c, Unitful)]
    if not unitful:
        return select_if_vmap_p.bind(pred, *cases, **kw)
    if any(u != unitful[0] for u in unitful[1:]):
        raise ValueError(f"Cannot select between arrays with units {unitful}.")
    arrays = [c.array if isinstance(c, Unitful) else c for c in cases]
    return Unitful(select_if_vmap_p.bind(pred, *arrays, **kw), unitful[0])


@quax.register(maybe_set_p)
def maybe_set_unitful(
    pred: ArrayLike, xs: ArrayLike, x: Unitful, *rest: Any, **kw: Any
) -> Any:
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


# No `test_diffrax_step_grad`: reverse-mode does not work here. The module
# docstring says why, and names both blocking frames.
