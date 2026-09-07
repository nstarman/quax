import functools as ft
import itertools as it
from collections.abc import Sequence
from typing import Any, Final, overload

import equinox as eqx
import jax._src.core as core
import jax.extend.core as jexc
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree_util as jtu
import plum
from jax.custom_derivatives import SymbolicZero as SZ
from jax.interpreters.ad import Zero

from ._compat import hi_aval, JAX_GE_0_9_2, JAX_GE_0_11_0, to_ct_aval, typeof
from ._dispatch import (
    _default_process,
    _dispatch_cache,
    _DISPATCH_MISS,
    _rules,
    _wrap_if_array,
)
from ._values import _dense, _DenseArrayValue, _is_value, T, Value


_PY_SCALARS: Final = (bool, int, float, complex)


class _QuaxTracer(core.Tracer):
    __slots__ = ("value", "_cached_aval")

    def __init__(self, trace: "_QuaxTrace", value: Value) -> None:
        assert _is_value(value)
        self._trace = trace
        self.value = value

        # Cache aval() once at construction rather than recomputing on every
        # access. aval() is pure by contract (Value is an eqx.Module — fields
        # are frozen after __init__; any Python/static metadata that determines
        # the resulting shape/dtype must itself be static/immutable; all
        # JAX/equinox transforms return new objects). aval() may still be
        # derived from dynamic jax.Array fields.
        #
        # Call aval unbound (via the type) rather than `value.aval()`: for user
        # Value types the latter goes through equinox's Module.__getattribute__,
        # which wraps the method in a fresh BoundMethod (itself a Module
        # allocation, ~20 µs) so that jax.jit(value.method) works. That wrapping
        # is pure overhead here — this aval is consumed immediately and never
        # handed to jax.jit. _DenseArrayValue already bypasses __getattribute__.
        self._cached_aval: core.AbstractValue
        if type(value) is _DenseArrayValue:
            # Fast path: no user code, no primitives bound, so no need to pay
            # for the trace-context manager below.
            self._cached_aval = _DenseArrayValue.aval(value)
        else:
            # `aval()` is user code and may bind JAX primitives (see
            # `StopGradArray` in tests/unit/test_aval_binds_primitive.py).
            # Tracers are constructed while the current trace has been taken --
            # inside `core.take_current_trace()`, or inside `Primitive.bind`
            # while it dispatches to `process_primitive` -- and JAX >=0.11
            # leaves the current trace as `None` there (it used to be
            # `eval_trace`), so binding anything would fail. Evaluate the aval
            # under the parent trace, which is where the value's leaves live.
            with core.set_current_trace(trace.parent_trace):
                self._cached_aval = type(value).aval(value)

    @property
    def aval(self) -> core.AbstractValue:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self._cached_aval

    def full_lower(self) -> "core.Tracer | _QuaxTracer":  # pyright: ignore[reportIncompatibleVariableOverride]
        return (
            core.full_lower(self.value.array)  # pyright: ignore[reportAttributeAccessIssue]
            if type(self.value) is _DenseArrayValue
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
        return _dense(val)

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
                # `type() is` rather than `isinstance`: _DenseArrayValue is
                # internal and never subclassed, and it inherits an ABC metaclass,
                # so `isinstance` would pay for ABCMeta.__instancecheck__ on every
                # input of every primitive. This is the hottest predicate in O1.
                if type(v) is not _DenseArrayValue:
                    break
                arrays.append(v.array)
            else:
                # All inputs were our dense tracers
                with core.set_current_trace(self.parent_trace):
                    out = primitive.bind(*arrays, **params)
                if primitive.multiple_results:
                    return [_QuaxTracer(self, _dense(x)) for x in out]  # pyright: ignore[reportGeneralTypeIssues]
                return _QuaxTracer(self, _dense(out))  # pyright: ignore[reportArgumentType]

        # ── full dispatch path ───────────────────────────────────────────────
        # Parse the tracers into values, unpacking any _DenseArrayValues.
        values = tuple(
            (x.array if type(x := self.to_value(t)) is _DenseArrayValue else x)
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

    def stage_value(self, val: Any) -> Any:
        """Lifts a value into this trace.

        Behavior:

        - If `val` is a JAX `SymbolicZero` (SZ), return it unchanged.
        - If `val` is a sequence (but not bytes/str), lift each element
          and return a list of tracers.
        - Otherwise wrap array-likes into `_DenseArrayValue` via
          `_wrap_if_array` and return a single `_QuaxTracer`.

        This mirrors the semantic intent of JAX's Trace.stage_value: create
        tracers for concrete values without emitting an identity primitive.
        """
        # Preserve JAX's SymbolicZero sentinel as-is so callers that check
        # `type(x) is SZ` continue to work.
        if type(val) is SZ:
            return val

        # Treat sequences (lists/tuples) as multiple staged values.
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            return [_QuaxTracer(self, _wrap_if_array(v)) for v in val]

        # Default: wrap arrays into a Value and return a tracer.
        return _QuaxTracer(self, _wrap_if_array(val))

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

    def process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, tracers, *, out_trees, symbolic_zeros
    ) -> "list[_QuaxTracer]":
        # Each `t.value` is a `Value`, and thus a PyTree; flatten the
        # `Value`-ness away and re-bind the primitive over the leaves.
        in_values = [self.to_value(t) for t in tracers]
        in_leaves, in_treedef = jtu.tree_flatten(in_values)
        in_leaf_avals = tuple(typeof(x) for x in in_leaves)

        fun, fun_aux = _custom_vjp_fun_wrap(fun, self.tag, in_treedef)
        fwd, fwd_aux = _custom_vjp_fwd_wrap(fwd, self.tag, in_treedef, out_trees)
        bwd = _custom_vjp_bwd_wrap(bwd, self.tag, in_treedef, in_leaf_avals, fwd_aux)

        def quax_out_trees():
            # Downstream only reads leaf counts and the forwarding list. Our
            # `fwd` returns every residual explicitly, so nothing is forwarded.
            out_treedef, res_treedef = fwd_aux()
            return out_treedef, res_treedef, [None] * res_treedef.num_leaves

        if JAX_GE_0_9_2:
            params = dict(
                subfuns=(fun, fwd, bwd),
                out_trees=quax_out_trees,
                symbolic_zeros=symbolic_zeros,
            )
            out_leaves = primitive.bind_with_trace(
                self.parent_trace, tuple(in_leaves), in_leaf_avals, params
            )
        else:
            out_leaves = primitive.bind_with_trace(
                self.parent_trace,
                (fun, fwd, bwd, *in_leaves),
                dict(out_trees=quax_out_trees, symbolic_zeros=symbolic_zeros),
            )

        # Either the primal or the fwd rule ran, depending on the parent trace.
        fst, aux = lu.merge_linear_aux(fun_aux, fwd_aux)
        out_treedef = aux if fst else aux[0]
        out_values = jtu.tree_unflatten(out_treedef, out_leaves)
        return [_QuaxTracer(self, x) for x in out_values]

    # TODO: add other process_* rules


class _held_trace:  # noqa: N801
    """Make a new `_QuaxTrace` current, for a block that spans a `yield`.

    Equivalent to `take_current_trace()` + `set_current_trace(_QuaxTrace(...))`, with
    one difference: nothing is restored if the block is left via `GeneratorExit`.
    The `lu.transformation` generators below keep this trace current while the
    function they wrap runs -- across a `yield` -- and when that function raises,
    JAX drops the generator where it stands rather than throwing into it. The block
    then unwinds whenever Python finalises the generator, by which point JAX has
    moved on, so restoring the stale trace would clobber whatever trace is live and
    surface much later as an `UnexpectedTracerError` in unrelated code. On
    abandonment, leave the trace context to its new owner.
    """

    __slots__ = ("_set", "_take", "_tag")

    def __init__(self, tag: core.TraceTag) -> None:
        self._tag = tag

    def __enter__(self) -> _QuaxTrace:
        self._take = core.take_current_trace()
        trace = _QuaxTrace(self._take.__enter__(), self._tag)
        self._set = core.set_current_trace(trace)
        self._set.__enter__()
        return trace

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is GeneratorExit:
            return
        self._set.__exit__(exc_type, exc_value, traceback)
        self._take.__exit__(exc_type, exc_value, traceback)


@lu.transformation_with_aux
def _custom_jvp_fun_wrap(tag, in_treedef, *in_leaves):
    in_values = jtu.tree_unflatten(in_treedef, in_leaves)
    with _held_trace(tag) as trace:
        in_tracers = [x if type(x) is SZ else _QuaxTracer(trace, x) for x in in_values]
        out_tracers = yield in_tracers, {}
        # The symbolic zero branch here will actually create a `quax.zero.Zero`!
        out_tracers = [
            jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SZ else t  # pyright: ignore[reportAttributeAccessIssue]
            for t in out_tracers
        ]
        out_values = [trace.to_value(t) for t in out_tracers]
        del out_tracers, in_tracers
    del trace
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
        SZ(type(p).aval(p))
        if (leaves := jtu.tree_leaves(t)) and all(type(l) is SZ for l in leaves)
        else t
        for p, t in zip(in_primal_values, in_tangent_values_raw)
    ]
    # Calling `_QuaxTracer` directly here, not using `trace.{pure,lift}` as each `x` is
    # a `Value`, not an array (=> pure) or tracer (=> lift).
    with _held_trace(tag) as trace:
        in_tracers = [
            x if type(x) is SZ else _QuaxTracer(trace, x)
            for x in it.chain(in_primal_values, in_tangent_values)
        ]
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
        for primal, tangent in zip(out_primal_values, out_tangent_values, strict=True):
            if primal.__class__ != tangent.__class__:
                # Unbound calls skip equinox's BoundMethod-wrapping
                # __getattribute__ (see _QuaxTracer.__init__).
                primal = type(primal).materialise(primal)
                tangent = type(tangent).materialise(tangent)
            out_primal_values2.append(primal)
            out_tangent_values2.append(tangent)
        del out_tracers, in_tracers
    del trace
    out_primals, out_primal_treedef = jtu.tree_flatten(out_primal_values2)
    out_tangents, out_tangent_treedef = jtu.tree_flatten(out_tangent_values2)
    if out_primal_treedef != out_tangent_treedef:
        raise ValueError(
            "Primals and tangents had the same class, but different flattened results."
        )
    yield out_primals + out_tangents, out_primal_treedef


def _keep_static(trace: "_QuaxTrace", x: Any) -> Any:
    """Leave Python scalars alone; densifying them would erase their staticness."""
    return x if isinstance(x, _PY_SCALARS) else trace.to_value(x)


def _leaf_counts(treedef: jtu.PyTreeDef, /) -> list[int]:  # pyright: ignore[reportInvalidTypeForm]
    """Number of leaves contributed by each element of a flattened list."""
    return [child.num_leaves for child in treedef.children()]


@ft.partial(lu.transformation_with_aux2, use_eq_store=True)
def _custom_vjp_fun_wrap(f, store, tag, in_treedef, *in_leaves):
    """Run the primal function on re-inflated `Value`s; store the out treedef."""
    in_values = jtu.tree_unflatten(in_treedef, in_leaves)
    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, tag)
        in_tracers = [_QuaxTracer(trace, x) for x in in_values]
        with core.set_current_trace(trace):
            out_values = [trace.to_value(t) for t in f(*in_tracers)]
        del trace, in_tracers
    out_leaves, out_treedef = jtu.tree_flatten(out_values)
    store.store(out_treedef)
    return out_leaves


@ft.partial(lu.transformation_with_aux2, use_eq_store=True)
def _custom_vjp_fwd_wrap(f, store, tag, in_treedef, out_trees, *in_leaves_and_nz):
    """Run the fwd rule on `Value`s, returning `(*residuals, *primal_outs)` as leaves.

    JAX interleaves each argument with a "this argument has a nonzero tangent"
    flag. Those flags arrive per *leaf*; the wrapped rule wants one per `Value`,
    so they are OR-ed together over each value's leaves.
    """
    leaves = in_leaves_and_nz[::2]
    nzs = in_leaves_and_nz[1::2]
    in_values = jtu.tree_unflatten(in_treedef, leaves)

    value_nzs = []
    i = 0
    for n in _leaf_counts(in_treedef):
        value_nzs.append(any(nzs[i : i + n]))
        i += n

    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, tag)
        in_tracers = [_QuaxTracer(trace, x) for x in in_values]
        interleaved = [x for pair in zip(in_tracers, value_nzs) for x in pair]
        with core.set_current_trace(trace):
            res_and_primals_out = f(*interleaved)
            # JAX prunes residuals that are identical to an input, recording the
            # index in `input_forwards`; splice them back so we can flatten the
            # full residual list (see `ad.JVPTrace.process_custom_vjp_call`).
            _, res_tree, input_forwards = out_trees()
            n_forwarded = sum(idx is not None for idx in input_forwards)
            n_res_out = res_tree.num_leaves - n_forwarded
            res_out = iter(res_and_primals_out[:n_res_out])
            res = [
                next(res_out) if idx is None else in_tracers[idx]
                for idx in input_forwards
            ]
            res_values = [_keep_static(trace, t) for t in res]
            out_values = [trace.to_value(t) for t in res_and_primals_out[n_res_out:]]
        del trace, in_tracers

    res_leaves, res_treedef = jtu.tree_flatten(res_values)
    out_leaves, out_treedef = jtu.tree_flatten(out_values)
    store.store((out_treedef, res_treedef))
    return [*res_leaves, *out_leaves]


@lu.transformation2
def _custom_vjp_bwd_wrap(f, tag, in_treedef, in_leaf_avals, fwd_aux, *res_and_cts):
    """Run the bwd rule on `Value`s, returning one cotangent per input *leaf*."""
    out_treedef, res_treedef = fwd_aux()
    n_res = res_treedef.num_leaves
    res_values = jtu.tree_unflatten(res_treedef, res_and_cts[:n_res])
    ct_values = jtu.tree_unflatten(out_treedef, res_and_cts[n_res:])
    # A `Value` holding SZ leaves cannot answer `aval()`, so lift a single-leaf
    # symbolic cotangent back to a value-level SZ (`all(...)` checks the leaf
    # *is* an SZ). Multi-leaf symbolic cotangents fall through unhandled.
    ct_values = [
        # The leaf's aval, not the `Value`'s (`_custom_jvp_jvp_wrap` uses the
        # latter); they agree for every single-leaf `Value` we know of.
        SZ(leaves[0].aval)
        if (leaves := jtu.tree_leaves(c, is_leaf=lambda x: type(x) is SZ))
        and all(type(x) is SZ for x in leaves)
        and len(leaves) == 1
        else c
        for c in ct_values
    ]

    with core.take_current_trace() as parent_trace:
        trace = _QuaxTrace(parent_trace, tag)
        in_tracers = [
            x if type(x) is SZ or isinstance(x, _PY_SCALARS) else _QuaxTracer(trace, x)  # pyright: ignore[reportArgumentType]
            for x in (*res_values, *ct_values)
        ]
        with core.set_current_trace(trace):
            # JAX 0.11's `defvjp_with_logs` made the flat bwd rule return a
            # `(cotangents, logs)` pair, which our caller unpacks in turn.
            raw = f(*in_tracers)
            raw, logs = raw if JAX_GE_0_11_0 else (raw, None)
            cts_in = [
                x if x is None or type(x) in (Zero, SZ) else trace.to_value(x)
                for x in raw
            ]
        del trace, in_tracers

    out: list[Any] = []
    i = 0
    for n, ct in zip(_leaf_counts(in_treedef), cts_in, strict=True):
        avals = in_leaf_avals[i : i + n]
        if ct is None or type(ct) in (Zero, SZ):
            out.extend(Zero(to_ct_aval(a)) for a in avals)
        else:
            leaves = jtu.tree_leaves(ct)
            if len(leaves) != n:
                msg = (
                    "custom_vjp bwd rule returned a cotangent that flattens to "
                    f"{len(leaves)} leaves for an input that flattens to {n}. "
                    "The cotangent must have the same structure as the primal."
                )
                raise TypeError(msg)
            out.extend(leaves)
        i += n
    return (out, logs) if JAX_GE_0_11_0 else out


# Any -> Any so overloads carry the public types. mypy can't prove the else
# branch is T (since T may be Value). To type the body, use Union[Value, T]
# + cast(T, x), or constrain T to exclude Value.
@overload
def _wrap_tracer(trace: _QuaxTrace, x: Value) -> _QuaxTracer: ...
@overload
def _wrap_tracer(trace: _QuaxTrace, x: T) -> T: ...
def _wrap_tracer(trace: _QuaxTrace, x: Any) -> Any:
    return _QuaxTracer(trace, x) if _is_value(x) else x


def _unwrap_tracer(trace: _QuaxTrace, x: Any, /) -> Any:
    if eqx.is_array_like(x):
        x = trace.full_raise(x)
    if isinstance(x, _QuaxTracer):
        return x.value.array if type(x.value) is _DenseArrayValue else x.value
    # A hijax value is an opaque leaf, so `tree_map` hands it here whole rather
    # than descending into it. Its components may be our tracers -- a hi
    # primitive applied under this trace runs its `expand` on whatever we gave
    # it -- and returning it untouched would leak a `_QuaxTracer` past the
    # `quaxify` boundary. Take it apart with the type's own lowering, unwrap
    # each component, and rebuild.
    #
    # `hi_aval` returns `None` (so this costs one dict lookup) for every leaf
    # that is not a hi value, which on this path is all of them: array-likes and
    # tracers were handled above.
    if (aval := hi_aval(x)) is not None:
        # A list comprehension, not a generator: an unconsumed-then-discarded
        # generator frame keeps its tracers reachable long enough for
        # `JAX_CHECK_TRACER_LEAKS` to report them as leaked.
        return aval.raise_val(*[_unwrap_tracer(trace, v) for v in aval.lower_val(x)])
    return x
