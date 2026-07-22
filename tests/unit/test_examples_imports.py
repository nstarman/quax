import importlib
import sys

import quax


# `quax/__init__.py` eagerly aliases these submodules for backwards
# compatibility (`quax.lora`, `quax.zero`), so they are imported at `quax`
# import time and are therefore NOT lazily loaded.
EAGER_MODULES = ("lora", "zero")
# The genuinely lazy submodules, derived from `quax.examples.__all__` so the
# list cannot drift as submodules are added or removed.
LAZY_MODULES = tuple(
    name for name in quax.examples.__all__ if name not in EAGER_MODULES
)


def clear_examples_modules():
    sys.modules.pop("quax.examples", None)
    for name in quax.examples.__all__:
        sys.modules.pop(f"quax.examples.{name}", None)


def test_examples_submodules_are_lazy_loaded():
    clear_examples_modules()
    importlib.import_module("quax.examples")
    # Re-importing `quax.examples` does not re-run `quax/__init__.py`, so the
    # eager aliases are not re-triggered; only assert the lazy submodules.
    assert all(f"quax.examples.{name}" not in sys.modules for name in LAZY_MODULES)


def test_examples_submodule_loaded_on_attribute_access():
    clear_examples_modules()
    examples = importlib.import_module("quax.examples")
    assert "quax.examples.prng" not in sys.modules
    assert examples.prng.__name__ == "quax.examples.prng"
