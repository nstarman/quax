from typing import Any, cast

import equinox as eqx
import jax
import jax.core
import jax.extend.core as jexc
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


def test_default_process_rejects_non_array_operand(monkeypatch):
    """A non-array, non-Value operand is a dispatch miss -> TypeError.

    Older jax routes such operands (e.g. a str passed to an operator) through
    ``_default_process``, which must raise a catchable ``TypeError`` -- so a
    caller can return ``NotImplemented`` -- rather than a bare ``AssertionError``
    (which also vanishes under ``python -O``). Called directly so the check is
    exercised on every jax version; newer jax rejects the operand earlier,
    before quax is consulted.

    quax's own test config runtime-typechecks ``_default_process`` (via
    ``--jaxtyping-packages=quax,beartype``), which would reject the non-array
    operand at the parameter boundary before this branch runs. Disable that
    checking so the call takes the production path, where it is off.
    """
    import jaxtyping

    from quax._dispatch import _default_process

    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)

    with pytest.raises(TypeError, match="neither a quax.Value nor"):
        _default_process(lax.add_p, ["not-an-array"], {})


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


def test_dispatch_cache_respects_unfaithful_types():
    """Rules on a type whose `isinstance` depends on the value are not cached by type.

    Regression test for https://github.com/nstarman/quax/issues/257.
    """

    class Tagged(quax.ArrayValue):
        array: Array
        tag: str = eqx.field(static=True)

        def materialise(self):
            return self.array

        def aval(self):
            return cast(jax.core.ShapedArray, typeof(self.array))

    class _BigMeta(type(Tagged)):
        def __instancecheck__(cls, x):
            return isinstance(x, Tagged) and x.tag == "big"

    class Big(Tagged, metaclass=_BigMeta):  # never instantiated
        __faithful__ = False

    # A private primitive, so the unfaithful rule doesn't outlive this test.
    prim = jexc.Primitive("unfaithful_test")
    prim.def_impl(lambda x: x)
    prim.def_abstract_eval(lambda x: x)

    @quax.register(prim)
    def unfaithful_test_big(x: Big):
        return Tagged(x.array, "big")

    f = quax.quaxify(prim.bind)
    big = Tagged(jnp.array(1.0), "big")
    small = Tagged(jnp.array(1.0), "small")

    # small -> big -> small: neither the miss nor the hit may be cached by type.
    assert not isinstance(f(small), Tagged)  # no rule: default materialises
    assert f(big).tag == "big"
    assert not isinstance(f(small), Tagged)


def test_nested_quaxify_joins_the_outer_trace():
    """A quaxified call inside a quaxified function sees the outer Values.

    The inner call used to start its own trace, which saw the outer tracer as a
    plain array: a `Unitful` built inside, times the outer `Unitful` input, came
    back as `Unitful(Unitful(..., m), m)` instead of `Unitful(..., m**2)`.
    """
    from quax.examples.unitful import meters, Unitful

    x = Unitful(jnp.array(3.0), meters)
    mul = quax.quaxify(jnp.multiply)

    got = quax.quaxify(lambda a: mul(a, Unitful(jnp.array(2.0), meters)))(x)

    assert isinstance(got, Unitful)
    assert not isinstance(got.array, Unitful)
    assert got.units == {meters: 2}
    assert jnp.array_equal(got.array, 6.0)


def test_nested_quaxify_under_jit_and_grad():
    """Joining the outer trace still composes with `jit` and `grad`, also when
    the inner Value is built inside the outer function from a traced operand.

    Such a Value holds tracers of the outer quax trace in its fields; they are
    lowered to their arrays before it joins that trace.
    """
    from quax.examples.unitful import meters, Unitful

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    def f(w):
        # `w * 1.0` binds under the outer trace, so the inner Unitful's array
        # field is an outer-trace tracer.
        out = quax.quaxify(lambda a: mul(a, Unitful(w * 1.0, meters)))(x)
        assert out.units == {meters: 2}
        return out.array

    assert jnp.array_equal(f(jnp.array(2.0)), 6.0)
    assert jnp.array_equal(jax.jit(f)(jnp.array(2.0)), 6.0)
    assert jnp.array_equal(jax.grad(f)(jnp.array(2.0)), 3.0)
    assert jnp.array_equal(jax.jit(jax.grad(f))(jnp.array(2.0)), 3.0)


def test_nested_quaxify_joins_with_a_closed_over_value():
    """A Value built outside the outer function has plain-array fields; it
    joins the outer trace as is."""
    from quax.examples.unitful import meters, Unitful

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)
    y = Unitful(jnp.array(2.0), meters)

    got = quax.quaxify(lambda a: mul(a, y))(x)

    assert isinstance(got, Unitful)
    assert not isinstance(got.array, Unitful)
    assert got.units == {meters: 2}
    assert jnp.array_equal(got.array, 6.0)


def test_nested_quaxify_three_levels():
    """Each level joins the outermost trace."""
    from quax.examples.unitful import meters, Unitful

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    got = quax.quaxify(
        lambda a: quax.quaxify(lambda b: mul(b, Unitful(jnp.array(2.0), meters)))(a)
    )(x)

    assert isinstance(got, Unitful)
    assert not isinstance(got.array, Unitful)
    assert got.units == {meters: 2}
    assert jnp.array_equal(got.array, 6.0)


def test_nested_quaxify_under_vmap():
    """Joining the outer trace composes with `vmap`."""
    from quax.examples.unitful import meters, Unitful

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    def f(w):
        out = quax.quaxify(lambda a: mul(a, Unitful(w * 1.0, meters)))(x)
        assert out.units == {meters: 2}
        return out.array

    ws = jnp.array([1.0, 2.0, 4.0])
    assert jnp.array_equal(jax.vmap(f)(ws), 3.0 * ws)
    assert jnp.array_equal(jax.jit(jax.vmap(f))(ws), 3.0 * ws)


def test_nested_quaxify_falls_back_when_a_value_wraps_an_outer_value(monkeypatch):
    """A Value whose field is an outer tracer wrapping another Value cannot
    join the outer trace; the inner call uses a fresh trace, as before."""
    import quax._quaxify as qq
    from quax.examples.unitful import meters, Unitful

    raised = []

    class Spy(qq._NotJoinable):
        def __init__(self, *args):
            raised.append(True)
            super().__init__(*args)

    monkeypatch.setattr(qq, "_NotJoinable", Spy)

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    # `Unitful(a, meters)` holds the outer tracer `a`, which wraps a Unitful.
    got = quax.quaxify(lambda a: mul(a, Unitful(a, meters)))(x)

    assert raised
    # The fresh inner trace sees `a` as an array, as before the join existed.
    assert isinstance(got, Unitful)
    assert got.units == {meters: 1}
    assert isinstance(got.array, Unitful)
    assert got.array.units == {meters: 2}
    assert jnp.array_equal(got.array.array, 9.0)


def test_nested_quaxify_with_a_custom_filter_spec_keeps_a_fresh_trace(monkeypatch):
    """An inner `quaxify` with a custom `filter_spec` never joins the outer
    trace, since it may pass Values through on purpose (see the redispatch
    tutorial)."""
    import quax._quaxify as qq
    from quax.examples.unitful import meters, Unitful

    def no_join(*args):
        raise AssertionError("joined the outer trace")

    monkeypatch.setattr(qq, "_join_trace", no_join)

    inner = quax.quaxify(jnp.multiply, filter_spec=lambda v: isinstance(v, Unitful))
    x = Unitful(jnp.array(3.0), meters)
    y = Unitful(jnp.array(2.0), meters)

    got = quax.quaxify(inner)(x, y)

    assert isinstance(got, Unitful)
    assert not isinstance(got.array, Unitful)
    assert got.units == {meters: 2}
    assert jnp.array_equal(got.array, 6.0)
