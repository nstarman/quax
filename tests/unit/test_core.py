from typing import Any, cast

import equinox as eqx
import jax
import jax.core
import jax.extend.core as jexc
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

import quax
from quax._compat import typeof
from quax.examples.unitful import kilograms, meters, Unitful
from quax.examples.zero import Zero


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


class _A(quax.ArrayValue):
    """A Value for the nesting tests below. Never materialises."""

    array: Array

    def materialise(self):
        raise ValueError("Refusing to materialise _A")

    def aval(self):
        return cast(jax.core.ShapedArray, typeof(self.array))


class _B(quax.ArrayValue):
    """A second Value type, to meet `_A` in one dispatch."""

    array: Array

    def materialise(self):
        raise ValueError("Refusing to materialise _B")

    def aval(self):
        return cast(jax.core.ShapedArray, typeof(self.array))


def _sum_primitive(name: str) -> jexc.Primitive:
    """A private `x + y` primitive, so the rules a test registers on it end with it."""
    prim = jexc.Primitive(name)
    prim.def_impl(lambda x, y: x + y)
    prim.def_abstract_eval(lambda x, y: x)
    return prim


def test_nested_quaxify_joins_the_outer_trace():
    """A quaxified call inside a quaxified function sees the outer Values.

    The inner call used to start its own trace, which saw the outer tracer as a
    plain array: a `Unitful` built inside, times the outer `Unitful` input, came
    back as `Unitful(Unitful(..., m), m)` instead of `Unitful(..., m**2)`.
    """

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

    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    def f(w):
        out = quax.quaxify(lambda a: mul(a, Unitful(w * 1.0, meters)))(x)
        assert out.units == {meters: 2}
        return out.array

    ws = jnp.array([1.0, 2.0, 4.0])
    assert jnp.array_equal(jax.vmap(f)(ws), 3.0 * ws)
    assert jnp.array_equal(jax.jit(jax.vmap(f))(ws), 3.0 * ws)


def test_nested_quaxify_falls_back_when_a_value_wraps_an_outer_value():
    """A Value whose field is an outer tracer wrapping another Value cannot
    join the outer trace; the inner call uses a fresh trace, as before."""
    mul = quax.quaxify(jnp.multiply)
    x = Unitful(jnp.array(3.0), meters)

    # `Unitful(a, meters)` holds the outer tracer `a`, which wraps a Unitful.
    got = quax.quaxify(lambda a: mul(a, Unitful(a, meters)))(x)

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


def test_nested_filter_spec_redispatch_inside_inline_jit():
    """The redispatch tutorial's nested `quaxify`, with a primitive under an
    inlined `jit` as `jnp` functions are.

    The inner trace's rule (here `jit_quax`) runs with the outer quax trace
    current. A `quaxify` called there must not join the outer trace: that would
    put both Values in one trace and make the lookup ambiguous, which is what the
    nesting avoids.
    """
    prim = _sum_primitive("redispatch_test")

    @quax.register(prim)
    def _(x: _A, y: jax.typing.ArrayLike | quax.ArrayValue):
        return quax.quaxify(prim.bind)(x.array, y)

    @quax.register(prim)
    def _(x: jax.typing.ArrayLike | quax.ArrayValue, y: _B):
        return x + 10 * y.array

    f = jax.jit(prim.bind, inline=True)
    a, b = _A(jnp.array(1.0)), _B(jnp.array(2.0))

    # One level: both rules match, as the tutorial shows.
    with pytest.raises(Exception, match="ambiguous"):
        quax.quaxify(f)(a, b)

    is_a = lambda x: isinstance(x, _A)
    is_b = lambda x: isinstance(x, _B)
    got = quax.quaxify(quax.quaxify(f, filter_spec=is_a), filter_spec=is_b)(a, b)
    assert jnp.array_equal(got, 21.0)


def test_nested_filter_spec_keeps_dispatch_order_through_jnp():
    """With a `filter_spec` at each level, a `jnp` function and its `lax`
    primitive give the same result.

    `jnp.multiply` goes through an inlined `jit`, whose rule runs with the outer
    quax trace current; joining it there dropped the units.
    """

    is_unitful = lambda v: isinstance(v, Unitful)
    is_zero = lambda v: isinstance(v, Zero)
    x = Unitful(jnp.array(2.0), meters)
    z = Zero((), jnp.float32)

    for fn in (jnp.multiply, lax.mul):
        got = quax.quaxify(
            quax.quaxify(fn, filter_spec=is_unitful), filter_spec=is_zero
        )(x, z)
        assert isinstance(got, Unitful), fn
        assert got.units == {meters: 1}
        assert isinstance(got.array, Zero)


@pytest.mark.parametrize("wrap", ["direct", "custom_jvp", "custom_vjp"])
def test_nested_filter_spec_rule_calls_custom_derivative_helper(wrap):
    """A rule of a nested quax trace calls a helper whose body calls `quaxify`.

    quax runs the body of a `custom_jvp` or `custom_vjp` function under a new trace
    object with the current trace's tag. That object is not the trace whose user
    code is running, so the `quaxify` in the body must not join it either.
    """
    prim = _sum_primitive("custom_helper_outer")
    prim2 = _sum_primitive("custom_helper_inner")

    def body(x, y):
        return quax.quaxify(prim2.bind)(x, y)

    if wrap == "custom_jvp":
        helper = jax.custom_jvp(body)
        helper.defjvp(lambda p, t: (helper(*p), helper(*p)))
    elif wrap == "custom_vjp":
        helper = jax.custom_vjp(body)
        helper.defvjp(lambda x, y: (helper(x, y), None), lambda r, g: (None, None))
    else:
        helper = body

    @quax.register(prim)
    def _(x: _A, y: jax.typing.ArrayLike | quax.ArrayValue):
        return helper(x, y)

    @quax.register(prim2)
    def _(x: _A, y: jax.typing.ArrayLike | quax.ArrayValue):
        return prim2.bind(x.array, y)

    @quax.register(prim2)
    def _(x: jax.typing.ArrayLike | quax.ArrayValue, y: _B):
        return x + 10 * y.array

    is_a = lambda v: isinstance(v, _A)
    is_b = lambda v: isinstance(v, _B)
    got = quax.quaxify(quax.quaxify(prim.bind, filter_spec=is_a), filter_spec=is_b)(
        _A(jnp.array(1.0)), _B(jnp.array(2.0))
    )
    assert jnp.array_equal(got, 21.0)


def test_user_trace_marker_is_restored():
    """The marker of which trace runs user code is reset when `quaxify` returns or
    raises, so a failed call does not change what later calls do."""
    from quax._trace import _user_trace

    mul = quax.quaxify(jnp.multiply)
    add = quax.quaxify(jnp.add)
    x = Unitful(jnp.array(3.0), meters)

    def nested(a):
        return mul(a, Unitful(jnp.array(2.0), meters))

    assert _user_trace.get() is None
    quax.quaxify(nested)(x)
    assert _user_trace.get() is None
    jax.jit(quax.quaxify(nested))(x)
    assert _user_trace.get() is None

    # A rule raises inside a nested call: adding metres to kilograms.
    with pytest.raises(ValueError, match="Cannot add"):
        quax.quaxify(lambda a: add(a, Unitful(jnp.array(2.0), kilograms)))(x)
    assert _user_trace.get() is None

    got = quax.quaxify(nested)(x)
    assert got.units == {meters: 2}


def test_outer_filter_spec_is_not_joined():
    """The pattern in docs/how-to/quaxify-part-of-a-function.md: the outer
    `quaxify` leaves one Value alone, for a plain `quaxify` inside to handle.

    That inner call must not join the outer trace: the two types would meet in one
    dispatch, which is the ambiguity the pattern exists to avoid.
    """
    prim = _sum_primitive("outer_filter_spec_test")

    @quax.register(prim)
    def _(x: _A, y: jax.typing.ArrayLike | quax.ArrayValue):
        return prim.bind(x.array, y)

    @quax.register(prim)
    def _(x: jax.typing.ArrayLike | quax.ArrayValue, y: _B):
        return prim.bind(x, 10 * y.array)

    def outer(a, b):
        # `b` arrives here untouched, so the nested quaxify is what handles it.
        return quax.quaxify(lambda bb: prim.bind(a, bb))(b)

    got = quax.quaxify(outer, filter_spec=(False, (True, False), {}))(
        _A(jnp.array(1.0)), _B(jnp.array(2.0))
    )
    assert jnp.array_equal(got, 21.0)


def test_nested_quaxify_on_local_values_returns_values():
    """A quaxified call on Values made inside a quaxified function gets none of the
    outer Values, so it runs in its own trace and returns a Value, which the code
    can then inspect. Joining would hand back a tracer of the outer trace."""
    mul = quax.quaxify(jnp.multiply)
    seen = []

    def f(a):
        local = Unitful(jnp.array(2.0), meters)
        r = mul(local, local)
        assert isinstance(r, Unitful)
        assert r.units == {meters: 2}
        seen.append(True)
        return a

    quax.quaxify(f)(Unitful(jnp.array(3.0), meters))
    jax.jit(quax.quaxify(f))(Unitful(jnp.array(3.0), meters))
    assert len(seen) == 2


@pytest.mark.filterwarnings("error")
def test_join_lowers_scalar_fields():
    """A field of the joined Value that is a dense tracer of a Python or NumPy
    scalar (here returned by a rule) is lowered to an array, without a dtype
    warning when x64 is off. `S.aval` reads `.shape`, which a Python float lacks."""

    class S(quax.ArrayValue):
        array: Array

        def materialise(self):
            return self.array

        def aval(self):
            return jax.core.ShapedArray(self.array.shape, self.array.dtype)

    @quax.register(lax.reduce_sum_p)
    def _(x: S, **kw: Any):
        return np.float64(1.5)

    @quax.register(lax.reduce_max_p)
    def _(x: S, **kw: Any):
        return 7.0

    mul = quax.quaxify(jnp.multiply)
    add = quax.quaxify(jnp.add)

    def f(x, u):
        # `u` and `x` are outer Values, so these calls join; `jnp.sum(x)` and
        # `jnp.max(x)` are dense tracers holding a NumPy and a Python scalar.
        r1 = mul(u, Unitful(jnp.sum(x), meters))
        r2 = add(x, S(jnp.max(x)))
        return r1, r2

    r1, r2 = quax.quaxify(f)(S(jnp.ones(3)), Unitful(jnp.array(2.0), meters))
    assert r1.units == {meters: 2}
    assert jnp.array_equal(r1.array, 3.0)
    assert jnp.array_equal(r2, jnp.full(3, 8.0))
