"""Compatibility utilities."""

from collections.abc import Callable
from importlib.metadata import version
from typing import Any, Final

import jax
import jax.extend.core as jexc
from packaging.version import Version


__all__ = (
    "JAX_VERSION",
    # Flags
    "JAX_GE_0_9_2",
    "JAX_GE_0_10_1",
    # Features
    "jit_p",
    "typeof",
)

JAX_VERSION: Final = Version(version("jax"))
JAX_GE_0_7_0: Final = JAX_VERSION >= Version("0.7.0")
JAX_GE_0_8_2: Final = JAX_VERSION >= Version("0.8.2")
JAX_GE_0_9_2: Final = JAX_VERSION >= Version("0.9.2")
JAX_GE_0_10_1: Final = JAX_VERSION >= Version("0.10.1")

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
