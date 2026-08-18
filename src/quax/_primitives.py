"""Quax's primitive handlers, and their caches for quaxified jaxprs."""

__all__ = ()

import weakref
from typing import Any, no_type_check

import jax
import jax._src.core as core
import jax.extend.core as jexc
import jax.tree_util as jtu
from jaxtyping import ArrayLike

from ._compat import is_early_inline, jit_p, scan_bind_params, unpack_scan_args
from ._dispatch import register
from ._quaxify import _Quaxify, quaxify
from ._values import _make_cache_finalizer, ArrayValue


# Cache for jit_quax: maps (id(jaxpr), treedef) -> (jaxpr_wref, jitted_fn).
# Populated on any call that is not inlined at trace time — both from eager mode
# and from the JIT-inside-JIT path.  In both cases the jaxpr is stable (pjit
# caches it keyed by function + abstract args), so id(jaxpr) is a reliable key.
#
# entry[0] is a weakref to the jaxpr; the finalizer evicts the entry when the
# jaxpr is GC'd, preventing unbounded growth in programs that generate many
# distinct jaxprs.  In practice the jaxpr is kept alive by JAX's own pjit cache
# for the lifetime of the decorated function, so eviction is rare.
_jit_quax_cache: dict[tuple, tuple[Any, Any]] = {}


@register(jit_p)
def jit_quax(
    *args: ArrayLike | ArrayValue, jaxpr: Any, inline: Any, **kwargs: Any
) -> Any:
    del kwargs

    # `inline` is a bool before JAX 0.11.0 and a `jax.Inline` enum from 0.11.0
    # on, so it is typed as `Any` here and interpreted by `is_early_inline`.
    #
    # Early inline: JAX has already decided to inline the body — just re-quaxify
    # and interpret it directly without the jax.jit wrapper overhead.
    if is_early_inline(inline):
        # Construct `_Quaxify` directly rather than via `quaxify()`: this runs on
        # every inlined pjit in a quaxified trace, and `quaxify()`'s
        # `module_update_wrapper` (which only exists to copy __wrapped__/__doc__
        # for user introspection) is pure overhead for this transient wrapper.
        return _Quaxify(jexc.jaxpr_as_fun(jaxpr), True, dynamic=False)(*args)

    leaves, treedef = jtu.tree_flatten(args)  # remove all Values

    # Non-inlined path: cache the jax.jit-wrapped quaxify callable so the
    # compiled XLA kernel is reused on subsequent calls with the same jaxpr.
    # This applies both for eager calls to a @jax.jit function and for nested
    # JIT tracing. id(jaxpr) is a reliable key here because the jaxpr is stable
    # for a given cached JAX lowering, and inlining is never requested at this
    # point.
    key = (id(jaxpr), treedef)
    entry = _jit_quax_cache.get(key)
    if entry is None:
        # The cached `jitted` must NOT strongly reference `jaxpr`: doing so would
        # pin the very object `entry[0]`'s weakref watches, so the finalizer could
        # never fire and the cache would grow without bound. Instead `qfun`
        # reaches `jaxpr` through the weakref, resolved only while tracing --
        # `jax.jit` runs compiled XLA on warm calls and never re-enters this
        # Python body. During any call `jaxpr` is alive (the caller passes it in),
        # so the weakref is always live when tracing actually happens.
        jaxpr_ref = weakref.ref(jaxpr, _make_cache_finalizer(_jit_quax_cache, key))

        def qfun(x: Any, _ref: Any = jaxpr_ref) -> Any:
            # Constructing `_Quaxify` directly (and calling `__call__` unbound)
            # bypasses `quaxify()`'s wrapper and eqx.Module.__call__'s dir() +
            # BoundMethod overhead.
            fn = _Quaxify(jexc.jaxpr_as_fun(_ref()), True, dynamic=False)
            return _Quaxify.__call__(fn, *jtu.tree_unflatten(treedef, x))

        jitted = jax.jit(qfun)
        entry = (jaxpr_ref, jitted)
        _jit_quax_cache[key] = entry

    return entry[1](leaves)  # call without Quax; jax.jit reuses compiled kernel


# Cache for while_quax: (id(cond_jaxpr), id(body_jaxpr), val_treedef)
#   -> (cond_wref, body_wref, quax_cond_jaxpr, quax_body_jaxpr, val_treedef)
# entry[0]/entry[1] are weakrefs; their finalizers evict the entry when either
# original jaxpr is collected, preventing unbounded growth.
_while_quax_cache: dict[tuple, tuple] = {}


@register(jax.lax.while_p)
def while_quax(
    *args: ArrayValue | ArrayLike,
    cond_nconsts: int,
    cond_jaxpr: core.ClosedJaxpr,
    body_nconsts: int,
    body_jaxpr: core.ClosedJaxpr,
) -> tuple[ArrayValue | ArrayLike, ...]:
    body_end = cond_nconsts + body_nconsts
    cond_consts = args[:cond_nconsts]
    body_consts = args[cond_nconsts:body_end]
    init_vals = args[body_end:]

    cond_leaves, _ = jtu.tree_flatten(cond_consts)
    body_leaves, _ = jtu.tree_flatten(body_consts)
    init_val_leaves, val_treedef = jtu.tree_flatten(init_vals)

    key = (id(cond_jaxpr), id(body_jaxpr), val_treedef)
    entry = _while_quax_cache.get(key)
    if entry is None:
        quax_cond_fn = quaxify(jexc.jaxpr_as_fun(cond_jaxpr))
        quax_cond_jaxpr = jax.make_jaxpr(quax_cond_fn)(*cond_consts, *init_vals)
        quax_body_fn = quaxify(jexc.jaxpr_as_fun(body_jaxpr))
        quax_body_jaxpr = jax.make_jaxpr(quax_body_fn)(*body_consts, *init_vals)
        fin = _make_cache_finalizer(_while_quax_cache, key)
        entry = (
            weakref.ref(cond_jaxpr, fin),
            weakref.ref(body_jaxpr, fin),
            quax_cond_jaxpr,
            quax_body_jaxpr,
            val_treedef,
        )
        _while_quax_cache[key] = entry
    else:
        _, _, quax_cond_jaxpr, quax_body_jaxpr, val_treedef = entry

    out_val = jax.lax.while_p.bind(
        *cond_leaves,
        *body_leaves,
        *init_val_leaves,
        cond_nconsts=cond_nconsts,
        cond_jaxpr=quax_cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=quax_body_jaxpr,
    )
    result = jtu.tree_unflatten(val_treedef, out_val)
    return result


_sentinel = object()


@register(jax.lax.cond_p)
def cond_quax(
    index: ArrayLike,
    *args: ArrayValue | ArrayLike,
    branches: tuple[core.ClosedJaxpr, ...],
    linear: tuple[bool, ...] | object = _sentinel,
    branches_platforms: tuple[str, ...] | object = _sentinel,
) -> Any:
    flat_args, in_tree = jtu.tree_flatten(args)

    out_trees: list[Any] = []

    def _make_quax_branch(
        jaxpr: core.ClosedJaxpr, /, *, materialise: bool = False
    ) -> core.ClosedJaxpr:
        def flat_quax_call(flat_args: list[Any]) -> list[Any]:
            _args = jtu.tree_unflatten(in_tree, flat_args)
            out = quaxify(jexc.jaxpr_as_fun(jaxpr))(*_args)
            if materialise:
                out = jtu.tree_map(
                    lambda x: (
                        type(x).materialise(x) if isinstance(x, ArrayValue) else x
                    ),
                    out,
                    is_leaf=lambda x: isinstance(x, ArrayValue),
                )
            flat_out, out_tree = jtu.tree_flatten(out)
            out_trees.append(out_tree)
            return flat_out

        return jax.make_jaxpr(flat_quax_call)(flat_args)

    quax_branches = tuple(_make_quax_branch(j) for j in branches)

    if any(t != out_trees[0] for t in out_trees[1:]):
        # The branches disagree on which outputs are `Value`s, and `cond_p`
        # needs one structure. Retrace with every `ArrayValue` materialised --
        # the fallback `Value.default` already applies to any primitive with no
        # rule. A type that refuses to materialise raises from there instead.
        out_trees.clear()
        quax_branches = tuple(_make_quax_branch(j, materialise=True) for j in branches)
        if any(t != out_trees[0] for t in out_trees[1:]):
            raise TypeError("all branches output must have the same pytree.")

    kwargs = {"linear": linear} if linear is not _sentinel else {}
    if branches_platforms is not _sentinel:
        kwargs["branches_platforms"] = branches_platforms

    out_val = jax.lax.cond_p.bind(index, *flat_args, branches=quax_branches, **kwargs)
    return jtu.tree_unflatten(out_trees[0], out_val)


# Cache for scan_quax: (id(jaxpr), consts_treedef, carry_treedef, xs_treedef)
#   -> (jaxpr_wref, quax_jaxpr, out_treedef, nc, nv)
# nc/nv = number of flat consts/carry leaves.
# entry[0] is a weakref; the finalizer evicts the entry when the jaxpr is
# collected, preventing unbounded growth.
_scan_quax_cache: dict[tuple, tuple] = {}


@register(jax.lax.scan_p)
def scan_quax(*args: ArrayValue | ArrayLike, jaxpr, **kwargs: Any) -> Any:
    """Quax handler for ``lax.scan_p``.

    Splits the flat ``args`` sequence into the three groups that ``lax.scan``
    uses — constants, initial carry, and stacked ``xs`` — then builds a
    quaxified jaxpr for the scan body and re-binds the primitive with it.

    How that grouping is encoded in the primitive's parameters changed in JAX
    0.11.0, so both the split and the re-bind go through `_compat`.

    The quaxified body jaxpr is cached by ``(id(jaxpr), consts_treedef,
    carry_treedef, xs_treedef)`` so that repeated calls with the same scan
    body and pytree structure skip the ``jax.make_jaxpr`` tracing step.
    """
    consts, carry, xs = unpack_scan_args(args, kwargs)

    consts_flat, c_tree = jtu.tree_flatten(consts)
    carry_flat, v_tree = jtu.tree_flatten(carry)
    xs_flat, x_tree = jtu.tree_flatten(xs)

    nc = len(consts_flat)
    nv = len(carry_flat)

    key = (id(jaxpr), c_tree, v_tree, x_tree)
    entry = _scan_quax_cache.get(key)
    if entry is None:
        # Trace the body against one abstract per-step slice of each xs (its
        # shape without the leading scan axis). Using an abstract slice rather
        # than a concrete `x[0]` keeps this valid when the scan length is 0,
        # which `x[0]` would index out of bounds.
        xs_slices = [jax.ShapeDtypeStruct(x.shape[1:], x.dtype) for x in xs_flat]
        trace_in = (*consts_flat, *carry_flat, *xs_slices)
        fn = core.jaxpr_as_fun(jaxpr)

        @no_type_check  # for beartype
        def quax_fn(*flat: ArrayValue | ArrayLike) -> Any:
            consts = jtu.tree_unflatten(c_tree, flat[:nc])
            carry = jtu.tree_unflatten(v_tree, flat[nc : nc + nv])
            xs = jtu.tree_unflatten(x_tree, flat[nc + nv :])
            return quaxify(fn)(*consts, *carry, *xs)

        quax_jaxpr, out_shape = jax.make_jaxpr(quax_fn, return_shape=True)(*trace_in)
        out_tree = jtu.tree_structure(out_shape)
        entry = (
            weakref.ref(jaxpr, _make_cache_finalizer(_scan_quax_cache, key)),
            quax_jaxpr,
            out_tree,
            nc,
            nv,
        )
        _scan_quax_cache[key] = entry
    else:
        _, quax_jaxpr, out_tree, nc, nv = entry

    out_flat = jax.lax.scan_p.bind(
        *consts_flat,
        *carry_flat,
        *xs_flat,
        jaxpr=quax_jaxpr,
        **scan_bind_params(
            kwargs,
            num_consts=nc,
            num_carry=nv,
            num_xs=len(xs_flat),
            num_ys=len(quax_jaxpr.out_avals) - nv,
        ),
    )

    return jtu.tree_unflatten(out_tree, out_flat)
