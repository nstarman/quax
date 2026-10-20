"""Whole-function dispatch: a quax-owned primitive for a function as a unit.

Calling a [`quax.dispatchable`][] function binds `dispatch_call_p`, which means
"`fn` applied to these arguments, vmapped once per entry of `batch_axes` over
leading axes". Outside quax it evaluates, differentiates and lowers as `fn`.

- Its batching rule turns a `vmap` into one more `batch_axes` entry (recording
  `axis_name` and `spmd_axis_name`, so collectives in the body still work), so a
  quax rule for `fn` -- which is per-example -- can be applied correctly whatever
  transforms sit between `quaxify` and the call. A `jax.custom_jvp` cannot offer
  this: a `vmap` rewrites its body but keeps its identity.
- Its JVP rule binds the same primitive for a derived function, `jvp(fn)`, so that
  under Quax a derivative taken inside `quaxify` follows the rule too.

The quax side is `dispatch_call_quax` in `_primitives.py`.
"""

__all__ = ("DispatchableFunction", "dispatch_call_p", "dispatchable")

import functools as ft
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.extend.core as jexc
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.interpreters import ad, batching, mlir

from ._compat import typeof


# One entry per vmap level, outermost first: (axis_name, spmd_axis_name).
BatchAxes: TypeAlias = tuple[tuple[Any, Any], ...]
Static: TypeAlias = tuple[tuple[int, Any], ...]


class DispatchableFunction:
    """A function that quax can dispatch on as a whole. See [`quax.dispatchable`][]."""

    def __init__(self, fn: Callable[..., Any], /) -> None:
        self.fn = fn
        ft.update_wrapper(self, fn)
        self._jvps: dict[tuple[Any, ...], JVPFunction] = {}

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
            *operands, fn=self, in_tree=in_tree, static=static, batch_axes=()
        )
        out_tree, _ = out_info(
            self, in_tree, static, (), tuple(typeof(x) for x in operands)
        )
        return jtu.tree_unflatten(out_tree, out)

    def jvp_of(
        self, in_tree: Any, static: Static, zeros: tuple[bool, ...]
    ) -> "JVPFunction":
        """The derived function computing this one's JVP (cached, so it is stable)."""
        key = (in_tree, static, zeros)
        if key not in self._jvps:
            self._jvps[key] = JVPFunction(self, in_tree, static, zeros)
        return self._jvps[key]


class JVPFunction(DispatchableFunction):
    """The tangent map of `base`: `(*primal_operands, *nonzero_tangents) -> tangents`.

    It returns the tangent output leaves only, and is linear in the tangents.
    `zeros` marks which tangents are symbolic zeros (and so not passed).
    """

    def __init__(
        self,
        base: DispatchableFunction,
        base_in_tree: Any,
        base_static: Static,
        zeros: tuple[bool, ...],
    ) -> None:
        self.base = base
        self.base_in_tree = base_in_tree
        self.base_static = base_static
        self.zeros = zeros
        super().__init__(self._tangents_of_body)
        self.__name__ = f"jvp({_name(base)})"
        self._transposes: dict[tuple[bool, ...], TransposeFunction] = {}

    def _tangents_of_body(self, *operands: Any) -> list[Any]:
        body = flat_body(self.base, self.base_in_tree, self.base_static)
        return self.tangents(body, operands)

    def split(self, operands: Sequence[Any]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Primal operands, and a tangent for every primal (zeros where symbolic)."""
        n = len(self.zeros)
        primals, nonzero = tuple(operands[:n]), iter(operands[n:])
        tangents = tuple(
            _zero_tangent(p) if zero else next(nonzero)
            for p, zero in zip(primals, self.zeros)
        )
        return primals, tangents

    def tangents(self, base_impl: Callable[..., Any], operands: Sequence[Any]) -> Any:
        """Tangent outputs of `base_impl` (a function of the base's operands)."""
        return jax.jvp(base_impl, *self.split(operands))[1]

    def transpose_of(self, ct_zeros: tuple[bool, ...]) -> "TransposeFunction":
        if ct_zeros not in self._transposes:
            self._transposes[ct_zeros] = TransposeFunction(self, ct_zeros)
        return self._transposes[ct_zeros]


class TransposeFunction(DispatchableFunction):
    """The transpose of a `JVPFunction`'s tangent map.

    `(*primal_operands, *nonzero_cotangents) -> cotangents of the nonzero tangents`.
    `ct_zeros` marks which output cotangents are symbolic zeros (and so not passed).
    """

    def __init__(self, jvp_fn: JVPFunction, ct_zeros: tuple[bool, ...]) -> None:
        self.jvp_fn = jvp_fn
        self.ct_zeros = ct_zeros
        super().__init__(self._transpose_of_body)
        self.__name__ = f"transpose({_name(jvp_fn)})"

    def _transpose_of_body(self, *operands: Any) -> list[Any]:
        body = flat_body(
            self.jvp_fn.base, self.jvp_fn.base_in_tree, self.jvp_fn.base_static
        )
        return self.transpose(body, operands)

    def transpose(self, base_impl: Callable[..., Any], operands: Sequence[Any]) -> Any:
        """Cotangents of the nonzero tangents, for `base_impl`'s tangent map."""
        jvp_fn = self.jvp_fn
        n = len(jvp_fn.zeros)
        primals = tuple(operands[:n])
        examples = [
            _zero_tangent(p) for p, zero in zip(primals, jvp_fn.zeros) if not zero
        ]

        def linear(*nonzero: Any) -> Any:
            return jvp_fn.tangents(base_impl, (*primals, *nonzero))

        def flat_linear(*nonzero: Any) -> list[Any]:
            return jtu.tree_leaves(linear(*nonzero))

        # Transpose over array leaves: under Quax an output may be a Value while its
        # cotangent arrives as a plain array (or the reverse).
        out_shapes = jax.eval_shape(linear, *examples)
        given = iter(operands[n:])
        cts: list[Any] = []
        for k, (shape, zero) in enumerate(zip(out_shapes, self.ct_zeros)):
            shapes = jtu.tree_leaves(shape)
            if zero:
                cts.extend(jnp.zeros(x.shape, x.dtype) for x in shapes)
                continue
            leaves = jtu.tree_leaves(next(given))
            if len(leaves) != len(shapes):
                raise NotImplementedError(
                    f"{_name(self)}: the cotangent of output {k} has {len(leaves)} "
                    f"array leaves, but the output has {len(shapes)}."
                )
            cts.extend(leaves)
        return list(jax.linear_transpose(flat_linear, *examples)(cts))


def _zero_tangent(primal: Any) -> Any:
    def zero(x: Any) -> Any:
        if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
            return jnp.zeros_like(x)
        return np.zeros(jnp.shape(x), jax.dtypes.float0)

    return jtu.tree_map(zero, primal)


def dispatchable(fn: Callable[..., Any], /) -> DispatchableFunction:
    """Make `fn` a function that Quax can dispatch on as a whole.

    Outside [`quax.quaxify`][] the result behaves as `fn`: same values, derivatives
    and lowering. Under `quaxify`, a rule registered with
    `@quax.register(the_function)` replaces it for the types the rule dispatches
    on. With no matching rule, `fn` runs under Quax as any other code does.

    A rule has `fn`'s signature: it takes `fn`'s positional arguments (pytrees
    allowed; dispatch is on each top-level argument's type) and returns `fn`'s
    output pytree. A derivative of the function, on either side of `quaxify`, is
    the derivative of the rule.

    Rules are **per-example**: a `jax.vmap` between `quaxify` and the call is
    applied by Quax to the rule, so a rule never sees a batch axis and must be
    vmappable -- built from JAX operations. Quax does this only when every
    [`quax.Value`][] argument sets [`quax.Value.batch_leaves`][]; otherwise it runs
    `fn` itself under Quax, which is always correct. A `vmap` whose `in_axes` are
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
    name = getattr(fn, "__name__", None)
    return name if name is not None else object.__repr__(fn.fn)


def rebuild(in_tree: Any, static: Static, operands: Sequence[Any]) -> tuple[Any, ...]:
    """The call's positional arguments, from its array operands and static leaves."""
    static_at = dict(static)
    ops = iter(operands)
    n = len(operands) + len(static)
    leaves = [static_at[i] if i in static_at else next(ops) for i in range(n)]
    return jtu.tree_unflatten(in_tree, leaves)


def vmapped(fn: Callable[..., Any], batch_axes: BatchAxes, /) -> Callable[..., Any]:
    """`fn`, vmapped over one leading axis per `batch_axes` entry (outermost first)."""
    for axis_name, spmd_axis_name in reversed(batch_axes):
        fn = jax.vmap(fn, axis_name=axis_name, spmd_axis_name=spmd_axis_name)
    return fn


def flat_body(
    fn: DispatchableFunction, in_tree: Any, static: Static
) -> Callable[..., list[Any]]:
    """`fn` as a function from array operands to flat output leaves."""

    def body(*operands: Any) -> list[Any]:
        return jtu.tree_leaves(fn.fn(*rebuild(in_tree, static, operands)))

    return body


@ft.cache
def out_info(
    fn: DispatchableFunction,
    in_tree: Any,
    static: Static,
    batch_axes: BatchAxes,
    avals: tuple[Any, ...],
) -> tuple[Any, tuple[jax_core.ShapedArray, ...]]:
    """The output treedef of `fn` and the output avals of the (batched) call."""
    tree_store: list[Any] = []

    def per_example(*ops: Any) -> list[Any]:
        leaves, tree = jtu.tree_flatten(fn.fn(*rebuild(in_tree, static, ops)))
        tree_store.append(tree)
        return leaves

    out = jax.eval_shape(
        vmapped(per_example, batch_axes),
        *[jax.ShapeDtypeStruct(a.shape, a.dtype, weak_type=a.weak_type) for a in avals],
    )
    out_avals = tuple(
        jax_core.ShapedArray(o.shape, o.dtype, weak_type=o.weak_type) for o in out
    )
    return tree_store[0], out_avals


dispatch_call_p = jexc.Primitive("quax_dispatch_call")
dispatch_call_p.multiple_results = True


def _impl(*operands: Any, fn, in_tree, static, batch_axes) -> list[Any]:
    return vmapped(flat_body(fn, in_tree, static), batch_axes)(*operands)


def _abstract_eval(*avals: Any, fn, in_tree, static, batch_axes) -> Any:
    return out_info(fn, in_tree, static, batch_axes, avals)[1]


def _batch(axis_data, operands, dims, *, fn, in_tree, static, batch_axes):
    params = dict(fn=fn, in_tree=in_tree, static=static)
    if all(d is None for d in dims):  # JAX calls fancy batchers for these too.
        out = dispatch_call_p.bind(*operands, **params, batch_axes=batch_axes)
        return out, [None] * len(out)
    # Put every batch axis first (broadcasting unbatched operands) and add one
    # vmap level: the new axis is the outermost.
    size = axis_data.size
    operands = [
        jnp.broadcast_to(x, (size, *jnp.shape(x)))
        if d is None
        else jnp.moveaxis(x, d, 0)
        for x, d in zip(operands, dims)
    ]
    batch_axes = ((axis_data.name, axis_data.spmd_name), *batch_axes)
    out = dispatch_call_p.bind(
        *operands, fn=fn, in_tree=in_tree, static=static, batch_axes=batch_axes
    )
    return out, [0] * len(out)


def _bind_derived(fn: DispatchableFunction, operands: Sequence[Any], batch_axes):
    return dispatch_call_p.bind(
        *operands,
        fn=fn,
        in_tree=jtu.tree_structure(tuple(range(len(operands)))),
        static=(),
        batch_axes=batch_axes,
    )


def _jvp(primals, tangents, *, fn, in_tree, static, batch_axes):
    # Primal outputs from the primitive itself, so they depend on primals only (as
    # linearization needs); tangent outputs from the derived, linear `jvp(fn)`.
    primals_out = dispatch_call_p.bind(
        *primals, fn=fn, in_tree=in_tree, static=static, batch_axes=batch_axes
    )
    zeros = tuple(type(t) is ad.Zero for t in tangents)
    nonzero = [t for t in tangents if type(t) is not ad.Zero]
    jvp_fn = fn.jvp_of(in_tree, static, zeros)
    tangents_out = _bind_derived(jvp_fn, (*primals, *nonzero), batch_axes)
    return primals_out, tangents_out


def _instantiate(ct: Any) -> Any:
    if type(ct) is not ad.Zero:
        return ct
    aval: Any = ct.aval
    return jnp.zeros(aval.shape, aval.dtype)


def _transpose(cts, *operands, fn, in_tree, static, batch_axes):
    del in_tree, static
    if isinstance(fn, JVPFunction):
        n = len(fn.zeros)
        if any(ad.is_undefined_primal(x) for x in operands[:n]):
            raise NotImplementedError(
                f"{_name(fn)} is linear in its tangents only, not its primals."
            )
        ct_zeros = tuple(type(ct) is ad.Zero for ct in cts)
        nonzero = [ct for ct in cts if type(ct) is not ad.Zero]
        t_fn = fn.transpose_of(ct_zeros)
        out = _bind_derived(t_fn, (*operands[:n], *nonzero), batch_axes)
        return [None] * n + list(out)
    if isinstance(fn, TransposeFunction):
        # Transposing back gives the tangent map again.
        jvp_fn = fn.jvp_fn
        n = len(jvp_fn.zeros)
        if any(ad.is_undefined_primal(x) for x in operands[:n]):
            raise NotImplementedError(
                f"{_name(fn)} is linear in its cotangents only, not its primals."
            )
        tangents = [_instantiate(ct) for ct in cts]
        out = _bind_derived(jvp_fn, (*operands[:n], *tangents), batch_axes)
        # `out` has a tangent for every output; the operands were the nonzero ones.
        return [None] * n + [x for x, zero in zip(out, fn.ct_zeros) if not zero]
    raise NotImplementedError(f"{_name(fn)} is not linear, so it cannot be transposed.")


dispatch_call_p.def_impl(_impl)
dispatch_call_p.def_abstract_eval(_abstract_eval)
mlir.register_lowering(dispatch_call_p, mlir.lower_fun(_impl, multiple_results=True))
batching.fancy_primitive_batchers[dispatch_call_p] = _batch
ad.primitive_jvps[dispatch_call_p] = _jvp
ad.primitive_transposes[dispatch_call_p] = _transpose
