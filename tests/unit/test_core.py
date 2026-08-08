from typing import Any, cast

import equinox as eqx
import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import pytest
from jax import Array

import quax
from quax._compat import typeof


def test_jit_inline():
    @quax.quaxify
    def f(x):
        return 2 * x + 1

    jaxpr = jax.make_jaxpr(f)(1.0)
    assert str(jaxpr).count("pjit") == 0


def test_default_override():
    records = []

    class Record(quax.ArrayValue):
        array: Array

        def materialise(self):
            assert False

        def aval(self):
            return cast(jax.core.ShapedArray, typeof(self.array))

        @staticmethod
        def default(primitive, values, params):
            arrays = [x.array if isinstance(x, Record) else x for x in values]
            records.append(primitive)
            out = quax.quaxify(primitive.bind)(*arrays, **params)
            if primitive.multiple_results:
                return [Record(x) for x in out]
            else:
                return Record(out)

    @quax.register(lax.mul_p)
    def _(a: Record, b: Record, /, **kw: Any):
        return Record(lax.mul_p.bind(a.array, b.array, **kw))

    x = Record(jnp.array(1.0))
    y = Record(jnp.array(2.0))
    z = Record(jnp.array(2.0))

    @quax.quaxify
    def f(a, b, c):
        return a + b * c

    f(x, y, z)
    assert records == [lax.add_p]
    records.clear()

    f(jnp.array(1.0), y, jnp.array(2.0))
    assert records == [lax.mul_p, lax.add_p]


def test_double_override():
    def make():
        class Foo(quax.ArrayValue):
            array: Array

            def materialise(self):
                assert False

            def aval(self):
                return cast(jax.core.ShapedArray, typeof(self.array))

            @staticmethod
            def default(primitive, values, params):
                arrays = []
                for value in values:
                    if isinstance(x, Foo):
                        arrays.append(x.array)
                    elif isinstance(x, quax.Value):
                        arrays.append(x.materialise())
                    elif eqx.is_array_like(x):
                        arrays.append(x)
                    else:
                        assert False
                out = primitive.bind(*arrays, **params)
                if primitive.multiple_results:
                    return [Foo(x) for x in out]
                else:
                    return Foo(cast(Array, out))

        return Foo

    Foo1 = make()
    Foo2 = make()

    x = Foo1(jnp.array(1.0))
    y = Foo2(jnp.array(2.0))

    @quax.quaxify
    def f(a, b):
        return a + b

    with pytest.raises(TypeError):
        f(x, y)


# See https://github.com/patrick-kidger/quax/issues/57
def test_default_path():
    x = jnp.array([[1.0, 2], [3, 4]]) / 10
    y = jnp.array([[5.0, 6], [7, 8]]) / 10

    exp = lax.betainc(1.0, x, y)
    got = quax.quaxify(lax.betainc)(jnp.array(1.0), x, y)

    assert jnp.array_equal(got, exp)


def test_quax_tracer_aval_cached():
    """_QuaxTracer caches aval() at construction; repeated .aval access is free."""
    import jax._src.core as jcore

    from quax._trace import _QuaxTrace, _QuaxTracer

    call_count = 0

    class CountedValue(quax.ArrayValue):
        array: Array

        def materialise(self):
            return self.array

        def aval(self):
            nonlocal call_count
            call_count += 1
            return typeof(self.array)

    val = CountedValue(jnp.array(1.0))
    tag = jcore.TraceTag()
    with jcore.take_current_trace() as parent:
        trace = _QuaxTrace(parent, tag)
        tracer = _QuaxTracer(trace, val)

    assert call_count == 1, (
        f"aval() should be called once at construction, got {call_count}"
    )

    # Repeated .aval access must not re-invoke aval()
    for _ in range(3):
        tracer.aval
    assert call_count == 1, (
        f"aval() should not be called again after construction, got {call_count}"
    )


def test_default_process_rejects_non_array_operand():
    """A non-array, non-Value operand is a dispatch miss -> TypeError.

    Older jax routes such operands (e.g. a str passed to an operator) through
    ``_default_process``, which must raise a catchable ``TypeError`` -- so a
    caller can return ``NotImplemented`` -- rather than a bare ``AssertionError``
    (which also vanishes under ``python -O``). Called directly so the check is
    exercised on every jax version; newer jax rejects the operand earlier,
    before quax is consulted.

    quax's own test config runtime-typechecks ``_default_process`` (via the
    ``beartype.claw`` hook installed in ``tests/conftest.py``), which would
    reject the non-array operand at the parameter boundary before this branch
    runs. Call the beartype-undecorated original directly (exposed as
    ``__wrapped__``) so the call takes the production path instead.
    """
    from quax._dispatch import _default_process

    with pytest.raises(TypeError, match="neither a quax.Value nor"):
        _default_process.__wrapped__(lax.add_p, ["not-an-array"], {})


# See https://github.com/nstarman/quax/issues/58
def test_quaxify_no_values_is_passthrough():
    """`quaxify(fn)(*args)` with no quax `Value` operands behaves exactly like
    `fn`, without wrapping the operands in tracers.

    Regression for #58: `jnp.compress` reads its boolean mask concretely, so a
    needless tracer wrap raised `TracerBoolConversionError` even though there was
    nothing to dispatch on.
    """
    xbool = jnp.array([True, False, True])
    x1 = jnp.array([1.0, 2.0, 3.0])
    got = quax.quaxify(jnp.compress)(xbool, x1)
    assert jnp.array_equal(got, jnp.compress(xbool, x1))


def test_quaxify_preserves_bearshape_cross_arg_binding():
    """Cross-argument dimension binding must still be enforced when the
    annotated function is called through `quax.quaxify` with a genuine
    `quax.Value` operand -- not just directly.

    A `quax.Value` operand forces `_Quaxify.__call__`'s slow/trace path
    (`_partition_and_wrap` wraps it in a `_QuaxTracer` before calling the
    wrapped function) rather than its fast-path shortcut, which bypasses the
    trace machinery entirely whenever no operand is a `Value`. bearshape's
    default (no `@bearshape.check`) memo discovery walks the beartype call
    stack via frame introspection; this regression-tests that it still finds
    the right frame through quaxify's extra trace layers, rather than
    silently skipping the cross-argument check.
    """
    import jax
    import jax.numpy as jnp
    from bearshape import B, N
    from bearshape.jax import Shaped
    from beartype import beartype
    from beartype.roar import BeartypeCallHintParamViolation

    from quax._compat import typeof

    variadic_b = ~B  # local, one-off use -- see the ignore comment below

    class _CanaryValue(quax.ArrayValue):
        array: jax.Array

        def materialise(self):
            return self.array

        def aval(self):
            return typeof(self.array)

    @beartype
    def add_same_n(
        x: Shaped[variadic_b, N],  # type: ignore[valid-type]
        y: Shaped[variadic_b, N],  # type: ignore[valid-type]
    ) -> Shaped[variadic_b, N]:  # type: ignore[valid-type]
        return x + y

    quaxified = quax.quaxify(add_same_n)
    value = _CanaryValue(jnp.zeros(3))  # a quax.Value -> forces the trace path

    # Matching N: passes, and still returns the right value.
    result = quaxified(value, jnp.zeros(3))
    assert result.shape == (3,)

    # Mismatched N: must still raise -- this is the frame-discovery canary.
    with pytest.raises(BeartypeCallHintParamViolation):
        quaxified(value, jnp.zeros(4))
