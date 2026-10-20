"""Whole-function dispatch: a quax-owned primitive for a function as a unit.

Calling a [`quax.dispatchable`][] function binds `dispatch_call_p`, which means
"`fn` applied to these arguments, vmapped over `n_batch` leading axes". Outside
quax it evaluates, differentiates and lowers as `fn`. Its batching rule turns a
`vmap` into `n_batch + 1`, so a quax rule for `fn` -- which is per-example -- can
be applied correctly whatever transforms sit between `quaxify` and the call. A
`jax.custom_jvp` cannot offer this: a `vmap` rewrites its body but keeps its
identity. The quax side is `dispatch_call_quax` in `_primitives.py`.
"""

__all__ = ("DispatchableFunction", "dispatch_call_p", "dispatchable")

import functools as ft
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.core as jax_core
import jax.extend.core as jexc
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.interpreters import ad, batching, mlir

from ._compat import typeof


class DispatchableFunction:
    """A function that quax can dispatch on as a whole. See [`quax.dispatchable`][]."""

    def __init__(self, fn: Callable[..., Any], /) -> None:
        self.fn = fn
        ft.update_wrapper(self, fn)

    def __repr__(self) -> str:
        return f"<quax.dispatchable {_name(self)}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(
                f"{_name(self)!r} is a quax.dispatchable function and takes "
                f"positional arguments only; got keyword arguments {sorted(kwargs)}."
            )
        leaves, in_tree = jtu.tree_flatten(args)
        operands = [x for x in leaves if eqx.is_array(x)]
        static = tuple((i, x) for i, x in enumerate(leaves) if not eqx.is_array(x))
        out = dispatch_call_p.bind(
            *operands, fn=self, in_tree=in_tree, static=static, n_batch=0
        )
        out_tree, _ = out_info(
            self, in_tree, static, 0, tuple(typeof(x) for x in operands)
        )
        return jtu.tree_unflatten(out_tree, out)


def dispatchable(fn: Callable[..., Any], /) -> DispatchableFunction:
    """Make `fn` a function that Quax can dispatch on as a whole.

    Outside [`quax.quaxify`][] the result behaves exactly as `fn`: same values,
    derivatives and lowering. Under `quaxify`, a rule registered with
    `@quax.register(the_function)` replaces it for the types the rule dispatches
    on. With no matching rule, `fn` runs under Quax as any other code does.

    A rule has `fn`'s signature: it takes `fn`'s positional arguments (pytrees
    allowed; dispatch is on each top-level argument's type) and returns `fn`'s
    output pytree. Rules are **per-example**: a `jax.vmap` between `quaxify` and
    the call is applied to the rule by Quax, so a rule never sees a batch axis and
    must be vmappable -- built from JAX operations. A `vmap` whose `in_axes` are
    not all 0 moves axes with `transpose` and `broadcast_in_dim`, so the Value
    types involved need rules for those.

    Keyword arguments are not supported. Non-array leaves of the arguments (Python
    scalars, strings, ...) are passed through as static data, and must be
    hashable.

    !!! Example

        ```python
        @quax.dispatchable
        def norm(x):
            return jnp.sqrt(jnp.sum(x**2))


        @quax.register(norm)
        def norm_unitful(x: Unitful) -> Unitful:
            return Unitful(norm(x.array), x.units)
        ```
    """
    return DispatchableFunction(fn)


def _name(fn: DispatchableFunction) -> str:
    return getattr(fn, "__name__", repr(fn.fn))


def rebuild(
    in_tree: Any, static: tuple[tuple[int, Any], ...], operands: Sequence[Any]
) -> tuple[Any, ...]:
    """The call's positional arguments, from its array operands and static leaves."""
    static_at = dict(static)
    ops = iter(operands)
    n = len(operands) + len(static)
    leaves = [static_at[i] if i in static_at else next(ops) for i in range(n)]
    return jtu.tree_unflatten(in_tree, leaves)


def vmapped(fn: Callable[..., Any], n_batch: int, /) -> Callable[..., Any]:
    for _ in range(n_batch):
        fn = jax.vmap(fn)
    return fn


def flat_body(
    fn: DispatchableFunction, in_tree: Any, static: tuple[tuple[int, Any], ...]
) -> Callable[..., list[Any]]:
    """`fn` as a function from array operands to flat output leaves."""

    def body(*operands: Any) -> list[Any]:
        return jtu.tree_leaves(fn.fn(*rebuild(in_tree, static, operands)))

    return body


@ft.cache
def out_info(
    fn: DispatchableFunction,
    in_tree: Any,
    static: tuple[tuple[int, Any], ...],
    n_batch: int,
    avals: tuple[Any, ...],
) -> tuple[Any, tuple[jax_core.ShapedArray, ...]]:
    """The output treedef of `fn` and the output avals of the (batched) call."""
    tree_store: list[Any] = []

    def batched(*operands: Any) -> list[Any]:
        def per_example(*ops: Any) -> list[Any]:
            leaves, tree = jtu.tree_flatten(fn.fn(*rebuild(in_tree, static, ops)))
            tree_store.append(tree)
            return leaves

        return vmapped(per_example, n_batch)(*operands)

    out = jax.eval_shape(
        batched,
        *[jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=a.weak_type) for a in avals],
    )
    out_avals = tuple(
        jax_core.ShapedArray(o.shape, o.dtype, weak_type=o.weak_type) for o in out
    )
    return tree_store[0], out_avals


dispatch_call_p = jexc.Primitive("quax_dispatch_call")
dispatch_call_p.multiple_results = True


def _impl(*operands: Any, fn, in_tree, static, n_batch) -> list[Any]:
    return vmapped(flat_body(fn, in_tree, static), n_batch)(*operands)


def _abstract_eval(*avals: Any, fn, in_tree, static, n_batch) -> Any:
    return out_info(fn, in_tree, static, n_batch, avals)[1]


def _batch(operands, dims, *, fn, in_tree, static, n_batch):
    # Put every batch axis first (broadcasting unbatched operands) and add one
    # vmap level: the new axis is the outermost.
    size = next(x.shape[d] for x, d in zip(operands, dims) if d is not None)
    operands = [
        jnp.broadcast_to(x, (size, *jnp.shape(x)))
        if d is None
        else jnp.moveaxis(x, d, 0)
        for x, d in zip(operands, dims)
    ]
    out = dispatch_call_p.bind(
        *operands, fn=fn, in_tree=in_tree, static=static, n_batch=n_batch + 1
    )
    return out, [0] * len(out)


def _jvp(primals, tangents, *, fn, in_tree, static, n_batch):
    # The body's own derivative. (For a `jax.custom_jvp` whose body was swapped
    # for this, JAX's custom JVP rule sits outside and takes precedence.)
    body = vmapped(flat_body(fn, in_tree, static), n_batch)
    tangents = tuple(ad.instantiate_zeros(t) for t in tangents)
    return jax.jvp(body, tuple(primals), tangents)


dispatch_call_p.def_impl(_impl)
dispatch_call_p.def_abstract_eval(_abstract_eval)
mlir.register_lowering(dispatch_call_p, mlir.lower_fun(_impl, multiple_results=True))
batching.primitive_batchers[dispatch_call_p] = _batch
ad.primitive_jvps[dispatch_call_p] = _jvp
