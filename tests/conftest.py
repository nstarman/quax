import os

from beartype import BeartypeConf, BeartypeStrategy
from beartype.claw import beartype_packages


# `claw_is_pep526=False`: beartype.claw's default type-checks PEP 526 module/class-level
# annotated assignments too (e.g. `x: T = value`), which jaxtyping's old import hook
# never did -- it only ever wrapped functions/dataclasses with `@jaxtyped`. Left at the
# default, claw chokes at import time on `_module.py`'s
# `_fast_specs: "weakref.WeakKeyDictionary[type, _FastSpec]" = ...` (a subscripted
# generic inside a string forward reference isn't a supported claw PEP 526 hint), which
# `beartype.beartype` itself never sees since it isn't a function signature. Disabling
# PEP 526 checking here restores parity with the old hook's actual coverage.
#
# Gate behind an env var so CI's CodSpeed benchmark job (which needs
# production performance, not instrumented) can disable it -- addopts-level
# tricks (`-o addopts=`, `-p no:env`) don't reach this: conftest.py's module
# scope always executes regardless of addopts or disabled plugins.
if os.environ.get("QUAX_DISABLE_RUNTIME_TYPECHECK") != "1":
    beartype_packages(
        ("quax",), conf=BeartypeConf(strategy=BeartypeStrategy.On, claw_is_pep526=False)
    )


import equinox.internal as eqxi
import pytest


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
