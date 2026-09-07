import importlib


__all__ = (
    "hijax",
    "lora",
    "named",
    "prng",
    "sparse",
    "structured_matrices",
    "unitful",
    "zero",
)


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted((*globals(), *__all__))
