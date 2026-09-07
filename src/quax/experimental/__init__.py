"""Experimental Quax APIs, which may change or be removed without notice.

Submodules are imported lazily, so nothing here runs -- and nothing here can
fail on an older JAX -- unless you ask for it by name.
"""

import importlib


__all__ = ("hijax",)


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__():
    return sorted((*globals(), *__all__))
