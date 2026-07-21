"""Quax: define custom array types and dispatch rules in JAX."""

__all__ = (
    "register",
    "quaxify",
    "Value",
    "ArrayValue",
)

import importlib.metadata

# Register JAX compatibility conversions (e.g., TypedInt → Array)
# These are registered at module import time
from . import _compat as _compat  # noqa: F401
from ._dispatch import register as register
from ._primitives import *  # noqa: F401, F403
from ._quaxify import quaxify as quaxify
from ._values import ArrayValue as ArrayValue, Value as Value


# After Quax core is imported.
# isort: split
from . import examples as examples


lora = examples.lora  # backward compatibility
zero = examples.zero  # backward compatibility


__version__ = importlib.metadata.version("quax")
