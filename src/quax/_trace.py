import itertools as it
from collections.abc import Sequence
from typing import Any, overload, Union

import equinox as eqx
import jax._src.core as core
import jax.extend.core as jexc
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
from jax.custom_derivatives import SymbolicZero as SZ
from jaxtyping import ArrayLike

from ._compat import JAX_GE_0_9_2, typeof
from ._dispatch import (
    _default_process,
    _dispatch_cache,
    _DISPATCH_MISS,
    _rules,
    _wrap_if_array,
)
from ._values import _DenseArrayValue, _is_value, T, Value


class _QuaxTracer(core.Tracer):
    __slots__ = ("value",)

    def __init__(self, trace: "_QuaxTrace", value: Value) -> None:
        assert _is_value(value)
        self._trace = trace
        self.value = value

    @property
    def aval(self) -> core.AbstractValue:  # pyright: ignore[reportIncompatibleVariableOverride]
        v = self.value
        # Fast path for _DenseArrayValue: bypass eqx's __getattribute__ which
        # wraps every method access in a new BoundMethod (another eqx.Module).
        # JAX calls .aval on every tracer throughout tracing, so this is hot.
        if type(v) is _DenseArrayValue:
            return typeof(object.__getattribute__(v, "array"))
        return v.aval()

    def full_lower(self) -> Union[ArrayLike, "_QuaxTracer"]:
        return (
            core.full_lower(self.value.array)  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(self.value, _DenseArrayValue)
            else self
        )


class _QuaxTrace(
    core.Trace if JAX_GE_0_9_2 else core.Trace[_QuaxTracer],  # pyright: ignore[reportGeneralTypeIssues, reportInvalidTypeArguments]
):
    __slots__ = ("tag", "parent_trace")

    def __init__(self, parent_trace: core.Trace | None, tag: core.TraceTag) -> None:
        self.tag = tag
        self.parent_trace = parent_trace
        super().__init__()

    def to_value(self, val: Any) -> Any:
        if isinstance(val, _QuaxTracer) and val._trace.tag is self.tag:  # type: ignore[attr-defined]
            return val.value
        return _DenseArrayValue(val)

    # ===========================================
    # Override methods from jax.core.Trace

    def process_primitive(
        self, primitive: jexc.Primitive, tracers: Sequence[Any], params: dict[str, Any]
    ) -> "_QuaxTracer | list[_QuaxTracer]":
        """Processes a primitive with the given tracers and parameters.

        This is the main entry point for processing primitives in JAX. It is
        called when a primitive is encountered during the evaluation of a JAX
        expression.

        **Arguments:**
        - `primitive`: The primitive to be processed.
        - `tracers`: A sequence of tracers that represent the inputs to the
          primitive.
        - `params`: A dictionary of parameters for the primitive.

        **Returns:**
        A ``_QuaxTracer`` (or list thereof) representing the result of
        processing the primitive.

        """
        # ── O1: fast path for all-dense inputs with no registered rule ──────
        # When every input is one of *our* _QuaxTracer wrappers around a plain
        # JAX array (_DenseArrayValue) and there is no registered quax rule for
        # this primitive, skip the entire dispatch stack and delegate directly
        # to the parent trace.  This avoids per-call allocations of
        # _DenseArrayValue objects, the plum lookup, _default_process, and the
        # materialise loop — the common case when quaxify wraps pure-JAX code.
        if primitive not in _rules:
            tag = self.tag
            arrays: list[Any] = []
            for t in tracers:
                if not (isinstance(t, _QuaxTracer) and t._trace.tag is tag):  # type: ignore[attr-defined]
                    break
                v = t.value
                if not isinstance(v, _DenseArrayValue):
                    break
                arrays.append(v.array)
            else:
                # All inputs were our dense tracers
                with core.set_current_trace(self.parent_trace):
                    out = primitive.bind(*arrays, **params)
                if primitive.multiple_results:
                    return [_QuaxTracer(self, _DenseArrayValue(x)) for x in out]  # pyright: ignore[reportGeneralTypeIssues]
                return _QuaxTracer(self, _DenseArrayValue(out))  # pyright: ignore[reportArgumentType]

        # ── full dispatch path ───────────────────────────────────────────────
        # Parse the tracers into values, unpacking any _DenseArrayValues.
        values = tuple(
            (x.array if isinstance(x := self.to_value(t), _DenseArrayValue) else x)
            for t in tracers
        )

        # ── O2: cache resolved dispatch method per (primitive, arg-types) ───
        # plum's resolve_method performs MRO inspection on every call; cache
        # the result so repeated invocations with the same type signature pay
        # the lookup cost only once.
        cache_key = (primitive, tuple(type(v) for v in values))
        cached = _dispatch_cache.get(cache_key)

        with core.set_current_trace(self.parent_trace):
            if cached is _DISPATCH_MISS:
                out = _default_process(primitive, values, params)
            elif cached is not None:
                out = cached(*values, **params)
            elif (rule := _rules.get(primitive)) is None:
                # First time seeing this (primitive, types) combination.
                _dispatch_cache[cache_key] = _DISPATCH_MISS
                out = _default_process(primitive, values, params)
            else:
                try:
                    method, _ = rule.resolve_method(values)
                except plum.NotFoundLookupError:
                    _dispatch_cache[cache_key] = _DISPATCH_MISS
                    out = _default_process(primitive, values, params)
                else:
                    _dispatch_cache[cache_key] = method
                    out = method(*values, **params)

        # Post-process the output
        if primitive.multiple_results:
            return [_QuaxTracer(self, _wrap_if_array(x)) for x in out]  # pyright: ignore[reportGeneralTypeIssues]
        return _QuaxTracer(self, _wrap_if_array(out))  # pyright: ignore[reportArgumentType]

    if JAX_GE_0_9_2:
        # In JAX v0.9.2+ (PR https://github.com/jax-ml/jax/pull/35730) JAX was
        # changed to have an extra argument -- `avals` -- in bind_with_trace
        # before params.

        def process_custom_jvp_call(
            self, primitive, fun, jvp, tracers, *, symbolic_zeros
        ) -> "list[_QuaxTracer]":
            # Each `t.value` will be some `Value`, and thus a PyTree. Here we
            # flatten the `Value`-ness away.
            in_leaves, in_treedef = jtu.tree_flatten(
                [self.to_value(t) for t in tracers]
            )
            fun, out_treedef1 = _custom_jvp_fun_wrap(fun, self.tag, in_treedef)
            jvp, out_treedef2 = _custom_jvp_jvp_wrap(jvp, self.tag, in_treedef)
            avals = tuple(x.aval if type(x) is SZ else typeof(x) for x in in_leaves)
            params = dict(subfuns=(fun, jvp), symbolic_zeros=symbolic_zeros)
            out_leaves = primitive.bind_with_trace(
                self.parent_trace, tuple(in_leaves), avals, params
            )
            _, out_treedef = lu.merge_linear_aux(out_treedef1, out_treedef2)
            out_values = jtu.tree_unflatten(out_treedef, out_leaves)
            return [_QuaxTracer(self, x) for x in out_values]

    else:

        def process_custom_jvp_call(
            self, primitive, fun, jvp, tracers, *, symbolic_zeros
        ) -> "list[_QuaxTracer]":
            # Each `t.value` will be some `Value`, and thus a PyTree. Here we
            # flatten the `Value`-ness away.
            in_leaves, in_treedef = jtu.tree_flatten(
                [self.to_value(t) for t in tracers]
            )
            fun, out_treedef1 = _custom_jvp_fun_wrap(fun, self.tag, in_treedef)
            jvp, out_treedef2 = _custom_jvp_jvp_wrap(jvp, self.tag, in_treedef)
            out_leaves = primitive.bind_with_trace(
                self.parent_trace,
                (fun, jvp, *in_leaves),
                dict(symbolic_zeros=symbolic_zeros),
            )
            _, out_treedef = lu.merge_linear_aux(out_treedef1, out_treedef2)
            out_values = jtu.tree_unflatten(out_treedef, out_leaves)
            return [_QuaxTracer(self, x) for x in out_values]

    # TODO: add other process_* rules


@lu.transformation_with_aux
def _custom_jvp_fun_wrap(tag, in_treedef, *in_leaves):
    in_values = jtu.tree_unflatten(in_treedef, in_leaves)
    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, tag)
        in_tracers = [x if type(x) is SZ else _QuaxTracer(trace, x) for x in in_values]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            # The symbolic zero branch here will actually create a `quax.zero.Zero`!
            out_tracers = [
                jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t  # pyright: ignore[reportAttributeAccessIssue]
                for t in out_tracers
            ]
            out_values = [trace.to_value(t) for t in out_tracers]
            del out_tracers
        del trace, in_tracers
    out_leaves, out_treedef = jtu.tree_flatten(out_values)
    yield out_leaves, out_treedef


@lu.transformation_with_aux
def _custom_jvp_jvp_wrap(tag, in_treedef, *in_primals_and_tangents):
    split = len(in_primals_and_tangents) // 2
    in_primal_values = jtu.tree_unflatten(in_treedef, in_primals_and_tangents[:split])
    in_tangent_values_raw = jtu.tree_unflatten(
        in_treedef, in_primals_and_tangents[split:]
    )
    # When symbolic_zeros=True, JAX may pass SymbolicZero tangent leaves. After
    # unflattening, SZs can be embedded inside a Value (e.g. MyArray(SZ)),
    # breaking .aval() calls. Promote only fully-symbolic tangents back to a
    # value-level SZ so the JVP rule can check `type(t) is SZ` directly, while
    # leaving mixed tangents untouched.
    in_tangent_values = [
        SZ(p.aval())
        if (leaves := jtu.tree_leaves(t)) and all(type(l) is SZ for l in leaves)
        else t
        for p, t in zip(in_primal_values, in_tangent_values_raw)
    ]
    # Calling `_QuaxTracer` directly here, not using `trace.{pure,lift}` as each `x` is
    # a `Value`, not an array (=> pure) or tracer (=> lift).
    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, tag)
        in_tracers = [
            x if type(x) is SZ else _QuaxTracer(trace, x)
            for x in it.chain(in_primal_values, in_tangent_values)
        ]
        with core.set_current_trace(trace):
            out_tracers = yield in_tracers, {}
            # The symbolic zero branch here will actually create a `quax.zero.Zero`!
            out_tracers = [
                jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t  # pyright: ignore[reportAttributeAccessIssue]
                for t in out_tracers
            ]
            out_values = [trace.to_value(t) for t in out_tracers]
            # Pre-calculate split point
            out_split = len(out_values) // 2
            out_primal_values = out_values[:out_split]
            out_tangent_values = out_values[out_split:]
            out_primal_values2 = []
            out_tangent_values2 = []
            for primal, tangent in zip(
                out_primal_values, out_tangent_values, strict=True
            ):
                if primal.__class__ != tangent.__class__:
                    primal = primal.materialise()
                    tangent = tangent.materialise()
                out_primal_values2.append(primal)
                out_tangent_values2.append(tangent)
            del out_tracers
        del trace, in_tracers
    out_primals, out_primal_treedef = jtu.tree_flatten(out_primal_values2)
    out_tangents, out_tangent_treedef = jtu.tree_flatten(out_tangent_values2)
    if out_primal_treedef != out_tangent_treedef:
        raise ValueError(
            "Primals and tangents had the same class, but different flattened results."
        )
    yield out_primals + out_tangents, out_primal_treedef


# Any -> Any so overloads carry the public types. mypy can't prove the else
# branch is T (since T may be Value). To type the body, use Union[Value, T]
# + cast(T, x), or constrain T to exclude Value.
@overload
def _wrap_tracer(trace: _QuaxTrace, x: Value) -> _QuaxTracer: ...
@overload
def _wrap_tracer(trace: _QuaxTrace, x: T) -> T: ...
def _wrap_tracer(trace: _QuaxTrace, x: Any) -> Any:
    return _QuaxTracer(trace, x) if _is_value(x) else x


def _unwrap_tracer(trace: _QuaxTrace, x: Any) -> Any:
    if eqx.is_array_like(x):
        x = trace.full_raise(x)
    if isinstance(x, _QuaxTracer):
        return x.value.array if isinstance(x.value, _DenseArrayValue) else x.value
    return x
