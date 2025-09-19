"""Compatibility utilities."""

from importlib.metadata import version
from typing import Final

import jax
import jax.extend.core as jexc


__all__ = ("jit_p", "ArrayLike")

JAX_VERSION: Final = tuple(int(p) for p in version("jax").split(".")[:3])
JAX_GE_0_7_0: Final = JAX_VERSION >= (0, 7, 0)

jit_p: jexc.Primitive
if JAX_GE_0_7_0:
    # In JAX 0.7.0, `pjit_p` was renamed to `jit_p`
    jit_p = jax._src.pjit.jit_p  # pyright: ignore[reportAttributeAccessIssue]

    # In JAX 0.7.2, `LiteralArray` was introduced. It should be part of
    # ArrayLike, but it's not.
    from jaxtyping import ArrayLike as _ArrayLike

    ArrayLike = _ArrayLike | jax._src.literals.LiteralArray  # pyright: ignore[reportAttributeAccessIssue]

else:
    jit_p = jax._src.pjit.pjit_p  # pyright: ignore[reportAttributeAccessIssue]

    from jaxtyping import ArrayLike
