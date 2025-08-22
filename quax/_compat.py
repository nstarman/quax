"""Compatibility utilities."""

from importlib.metadata import version
from typing import Final

import jax
import jax.extend.core as jexc


__all__ = ("jit_p",)

JAX_VERSION: Final = tuple(int(p) for p in version("jax").split(".")[:3])
JAX_GE_0_7_0: Final = JAX_VERSION >= (0, 7, 0)

jit_p: jexc.Primitive
if JAX_GE_0_7_0:
    jit_p = jax._src.pjit.jit_p  # pyright: ignore[reportAttributeAccessIssue]
else:
    jit_p = jax._src.pjit.pjit_p  # pyright: ignore[reportAttributeAccessIssue]
