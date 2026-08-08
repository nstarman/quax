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
beartype_packages(
    ("quax",), conf=BeartypeConf(strategy=BeartypeStrategy.On, claw_is_pep526=False)
)


import equinox.internal as eqxi
import pytest


@pytest.fixture()
def getkey():
    return eqxi.GetKey()
