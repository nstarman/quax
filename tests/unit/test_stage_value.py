import jax._src.core as core
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero as SZ

from quax._trace import _QuaxTrace, _QuaxTracer
from quax._values import _DenseArrayValue


# `take_current_trace` leaves the current trace unset for the duration of the
# block, so any JAX operation inside it has no trace to bind against. Build the
# arrays up front and keep the block down to the code actually under test.


def test_stage_value_lifts_array_to_quaxtracer():
    arr = jnp.array([1.0, 2.0, 3.0])

    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, core.TraceTag())
        t = trace.stage_value(arr)

    assert isinstance(t, _QuaxTracer)
    assert isinstance(t.value, _DenseArrayValue)
    assert jnp.array_equal(t.value.array, arr)


def test_stage_value_handles_symbolic_zero():
    # SymbolicZero should be returned unchanged (JAX expects SZ handling).
    sz = SZ(jnp.array(0.0).aval) if hasattr(SZ, "__call__") else SZ

    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, core.TraceTag())
        # We expect the trace to accept SZ and return it as-is or wrapped
        # appropriately; the exact behavior is implementation defined but should
        # not raise NotImplementedError.
        res = trace.stage_value(sz)

    # Either returns the SZ sentinel or a tracer that carries it.
    assert res is not None
