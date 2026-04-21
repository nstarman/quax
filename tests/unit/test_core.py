from typing import Any, cast

import equinox as eqx
import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

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
