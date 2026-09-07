"""Units built on JAX's hijax API, wrapped in a Quax type.

See [the reference](../api/hijax.md) for what is supported, and
[Quax and hijax](../hijax.md) for why the two libraries fit together this way.
"""

from ._core import (
    MAPPED as MAPPED,
    Unitful as Unitful,
)
from ._unitful_array import (
    add as add,
    broadcast_in_dim as broadcast_in_dim,
    int_pow as int_pow,
    mul as mul,
    sum as sum,
    to_units as to_units,
    UnitfulArray as UnitfulArray,
    UnitfulArraySpec as UnitfulArraySpec,
    UnitfulArrayTy as UnitfulArrayTy,
    Units as Units,
    unwrap as unwrap,
    wrap as wrap,
)
