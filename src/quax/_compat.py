"""Compatibility utilities."""

__all__ = (
    "JAX_VERSION",
    # Flags
    "JAX_GE_0_7_2",
    "JAX_GE_0_9_2",
    "JAX_GE_0_10_1",
    "JAX_GE_0_10_2",
    "JAX_GE_0_11_0",
    # Features
    "jit_p",
    "is_early_inline",
    "scan_bind_params",
    "typeof",
    "unpack_scan_args",
)

from collections.abc import Callable, Sequence
from importlib.metadata import version
from typing import Any, Final

import jax
import jax.extend.core as jexc
import jax.numpy as jnp
import plum
from jaxtyping import Array
from packaging.version import Version


JAX_VERSION: Final = Version(version("jax"))
JAX_GE_0_7_0: Final = JAX_VERSION >= Version("0.7.0")
JAX_GE_0_7_2: Final = JAX_VERSION >= Version("0.7.2")
JAX_GE_0_8_2: Final = JAX_VERSION >= Version("0.8.2")
JAX_GE_0_9_2: Final = JAX_VERSION >= Version("0.9.2")
JAX_GE_0_10_1: Final = JAX_VERSION >= Version("0.10.1")
JAX_GE_0_10_2: Final = JAX_VERSION >= Version("0.10.2")
JAX_GE_0_11_0: Final = JAX_VERSION >= Version("0.11.0")

jit_p: jexc.Primitive
if JAX_GE_0_7_0:
    jit_p = jax._src.pjit.jit_p  # pyright: ignore[reportAttributeAccessIssue]
else:
    jit_p = jax._src.pjit.pjit_p  # pyright: ignore[reportAttributeAccessIssue]

# JAX 0.11.0 changed `jit_p`'s `inline` parameter from a `bool` to the
# `jax.Inline` enum: the old `True` became `Inline.JAX_EARLY` and the old
# `False` became `Inline.AUTO` (see `jax._src.pjit._canonicalize_inline`).
# Enum members are always truthy, so `bool(inline)` is no longer a valid test
# for "JAX asked us to inline the body at trace time"; only `JAX_EARLY` means
# that. The other members keep the call in the jaxpr and defer inlining to
# lowering or to XLA, which is the same structural situation as the old
# `False`.
is_early_inline: Callable[[Any], bool]
if JAX_GE_0_11_0:

    def is_early_inline(inline: Any, /) -> bool:
        """Whether `jit_p`'s `inline` parameter asks for inlining at trace time."""
        return inline is True or inline is jax.Inline.JAX_EARLY  # pyright: ignore[reportAttributeAccessIssue]

else:

    def is_early_inline(inline: Any, /) -> bool:
        """Whether `jit_p`'s `inline` parameter asks for inlining at trace time."""
        return bool(inline)


# JAX 0.11.0 reworked `scan_p`'s parameters: the positional `num_consts` /
# `num_carry` split (plus `linear` and `_split_transpose`) was replaced by a
# pair of `FlatTree` descriptors, `ft_in` and `ft_out`, which describe the
# const/carry/xs grouping of the inputs and the carry/ys grouping of the
# outputs.
#
# `unpack_scan_args` recovers the (consts, carry, xs) grouping, and
# `scan_bind_params` builds the parameters for re-binding `scan_p` with a
# quaxified body -- which generally has a *different* number of flat operands,
# since a single `ArrayValue` can flatten to several arrays.
if JAX_GE_0_11_0:
    from jax._src import (
        flattree as jax_flattree,  # pyright: ignore[reportAttributeAccessIssue]
    )

    def unpack_scan_args(
        args: Sequence[Any], params: dict[str, Any], /
    ) -> tuple[list, list, list]:
        """Split `scan_p`'s flat operands into (consts, carry, xs)."""
        groups = params["ft_in"].update(args).unpack()
        return tuple(list(g.vals) for g in groups)  # type: ignore[return-value]

    def scan_bind_params(
        params: dict[str, Any],
        /,
        *,
        num_consts: int,
        num_carry: int,
        num_xs: int,
        num_ys: int,
    ) -> dict[str, Any]:
        """Parameters for re-binding `scan_p` with the given group sizes.

        The rebuilt `ft_in`/`ft_out` use plain `nones`: the quaxified body is
        retraced from scratch, so none of the caller's forwarding or filtering
        optimizations (e.g. the `RightsOnly` markers JAX uses to prune extensive
        outputs) carry over to it.
        """
        rest = {k: v for k, v in params.items() if k not in ("ft_in", "ft_out")}
        nones = jax_flattree.nones
        return {
            **rest,
            "ft_in": jax_flattree.pack(
                (nones(num_consts), nones(num_carry), nones(num_xs))
            ),
            "ft_out": jax_flattree.pack((nones(num_carry), nones(num_ys))),
        }

else:

    def unpack_scan_args(
        args: Sequence[Any], params: dict[str, Any], /
    ) -> tuple[list, list, list]:
        """Split `scan_p`'s flat operands into (consts, carry, xs)."""
        nc = params["num_consts"]
        body_end = nc + params["num_carry"]
        return list(args[:nc]), list(args[nc:body_end]), list(args[body_end:])

    def scan_bind_params(
        params: dict[str, Any],
        /,
        *,
        num_consts: int,
        num_carry: int,
        num_xs: int,
        num_ys: int,
    ) -> dict[str, Any]:
        """Parameters for re-binding `scan_p` with the given group sizes."""
        del num_xs, num_ys
        return {**params, "num_consts": num_consts, "num_carry": num_carry}


typeof: Callable[[Any], Any]
if JAX_GE_0_8_2:
    typeof = jax.typeof
else:
    typeof = jax.core.get_aval  # pyright: ignore[reportAttributeAccessIssue]


# Mark `jax.Array` as "faithful" for `plum` multiple dispatch.
#
# `plum` caches method resolution keyed on the *types* of the arguments, but
# only when every type registered on a function is "faithful" -- i.e. when
# ``isinstance(x, T)`` agrees with ``issubclass(type(x), T)``, so that the type
# alone determines dispatch. Newer JAX gives `jax.Array` a custom metaclass
# ``__instancecheck__``, so `plum` conservatively treats it as non-faithful.
#
# A single non-faithful type on a `plum` function disables its resolution cache
# entirely. Because `quax` registers ``convert`` methods targeting `jax.Array`
# (below), and `plum` runs ``convert`` on the return value of every dispatched
# function with a concrete return annotation, that uncached resolution is re-run
# on every such call -- a large, easily-avoided cost for any library built on
# `quax.ArrayValue`.
#
# Setting ``__faithful__`` re-enables the cache. It only gates *caching* of
# `plum`'s deterministic, type-based resolution; it never changes which method
# is selected, so results are unchanged. This must run before the ``convert``
# methods below are registered, so their signatures are built faithful.
jax.Array.__faithful__ = True  # type: ignore[attr-defined]


# Register plum conversions for JAX's typed literal scalars (TypedInt,
# TypedFloat, TypedComplex). These types were introduced in JAX 0.7.2 to
# preserve dtype information during canonicalization. They allow any library
# using quax.ArrayValue to seamlessly handle Python scalars that JAX internally
# represents as typed literals.
if JAX_GE_0_7_2:
    from jax._src import literals as jax_literals

    @plum.conversion_method(type_from=jax_literals.TypedInt, type_to=Array)  # type: ignore[arg-type]
    @plum.conversion_method(type_from=jax_literals.TypedFloat, type_to=Array)  # type: ignore[arg-type]
    @plum.conversion_method(type_from=jax_literals.TypedComplex, type_to=Array)  # type: ignore[arg-type]
    def _typed_int_to_array(obj: Any, /) -> Array:
        """Convert JAX typed literals to JAX arrays.

        JAX's typed literals carry dtype information through Python scalars.
        When canonicalizing inputs (e.g., int -> TypedInt), JAX preserves dtype
        info. This converter extracts that dtype and creates a proper JAX array.
        """
        return jnp.asarray(obj, dtype=obj.dtype)
