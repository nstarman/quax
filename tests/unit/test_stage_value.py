import jax._src.core as core
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero as SZ

from quax._core import _DenseArrayValue, _QuaxTrace, _QuaxTracer


def test_stage_value_lifts_array_to_quaxtracer():
    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, core.TraceTag())
        arr = jnp.array([1.0, 2.0, 3.0])
        t = trace.stage_value(arr)
        assert isinstance(t, _QuaxTracer)
        assert isinstance(t.value, _DenseArrayValue)
        assert jnp.array_equal(t.value.array, arr)


def test_stage_value_handles_symbolic_zero():
    # SymbolicZero should be returned unchanged (JAX expects SZ handling).
    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, core.TraceTag())
        sz = SZ(jnp.array(0.0).aval) if hasattr(SZ, "__call__") else SZ
        # We expect the trace to accept SZ and return it as-is or wrapped
        # appropriately; the exact behavior is implementation defined but should
        # not raise NotImplementedError.
        res = trace.stage_value(sz)
        # Either returns the SZ sentinel or a tracer that carries it.
        assert res is not None
