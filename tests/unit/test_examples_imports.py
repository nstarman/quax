import importlib
import sys


EXAMPLE_MODULES = ("lora", "named", "prng", "sparse", "structured_matrices", "zero")


def clear_examples_modules():
    sys.modules.pop("quax.examples", None)
    for name in EXAMPLE_MODULES:
        sys.modules.pop(f"quax.examples.{name}", None)


def test_examples_submodules_are_lazy_loaded():
    clear_examples_modules()
    importlib.import_module("quax.examples")
    assert all(f"quax.examples.{name}" not in sys.modules for name in EXAMPLE_MODULES)


def test_examples_submodule_loaded_on_attribute_access():
    clear_examples_modules()
    examples = importlib.import_module("quax.examples")
    assert "quax.examples.prng" not in sys.modules
    assert examples.prng.__name__ == "quax.examples.prng"
