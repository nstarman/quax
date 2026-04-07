import importlib.metadata

from ._core import (
    ArrayValue as ArrayValue,
    quaxify as quaxify,
    register as register,
    Value as Value,
)


# After Quax core is imported.
from . import examples as examples  # isort: skip

lora = examples.lora  # backward compatibility
zero = examples.zero  # backward compatibility


__version__ = importlib.metadata.version("quax")
