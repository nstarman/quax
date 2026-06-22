"""Compatibility utilities."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Any, Final

import jax
import jax.extend.core as jexc
import jax.numpy as jnp
import plum
from jaxtyping import Array
from packaging.version import Version


__all__ = (
    "JAX_VERSION",
    # Flags
    "JAX_GE_0_7_2",
    "JAX_GE_0_9_2",
    "JAX_GE_0_10_1",
    "JAX_GE_0_10_2",
    # Features
    "jit_p",
    "typeof",
)

JAX_VERSION: Final = Version(version("jax"))
JAX_GE_0_7_0: Final = JAX_VERSION >= Version("0.7.0")
JAX_GE_0_7_2: Final = JAX_VERSION >= Version("0.7.2")
JAX_GE_0_8_2: Final = JAX_VERSION >= Version("0.8.2")
JAX_GE_0_9_2: Final = JAX_VERSION >= Version("0.9.2")
JAX_GE_0_10_1: Final = JAX_VERSION >= Version("0.10.1")
JAX_GE_0_10_2: Final = JAX_VERSION >= Version("0.10.2")

jit_p: jexc.Primitive
if JAX_GE_0_7_0:
    jit_p = jax._src.pjit.jit_p  # pyright: ignore[reportAttributeAccessIssue]
else:
    jit_p = jax._src.pjit.pjit_p  # pyright: ignore[reportAttributeAccessIssue]

typeof: Callable[[Any], Any]
if JAX_GE_0_8_2:
    typeof = jax.typeof
else:
    typeof = jax.core.get_aval  # pyright: ignore[reportAttributeAccessIssue]


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
