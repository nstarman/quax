"""Micro-benchmarks for the `equinox.Module` construction and trace hot paths.

These isolate the costs the Module-metaclass optimisation targets:

- ``_dense`` / ``_DenseArrayValue`` construction — the single hottest allocation
  in a trace (built per primitive input and output);
- user ``Value`` construction through ``_FastModuleMeta``, for both a
  dataclass-generated ``__init__`` and a custom ``__init__`` with
  ``__check_init__``;
- eager ``quaxify`` trace time over a multi-primitive chain, which exercises
  construction, dispatch, and the ``aval``/``materialise`` call paths together.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import quax
from quax._compat import typeof
from quax._values import _dense, _DenseArrayValue

from ..unit.myarray import MyArray


_arr = jnp.arange(64.0)


class _CustomInit(quax.ArrayValue):
    """A Value with a custom __init__ and a __check_init__ invariant."""

    data: jax.Array
    scale: float = eqx.field(static=True)

    def __init__(self, data, scale=2.0):
        self.data = jnp.asarray(data)
        self.scale = scale

    def __check_init__(self):
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def materialise(self):
        return self.data * self.scale

    def aval(self):
        return typeof(self.data)


@pytest.mark.benchmark(group="construction")
@pytest.mark.parametrize(
    "ctor",
    [_dense, _DenseArrayValue, MyArray, _CustomInit],
    ids=["dense_factory", "dense_value", "user_dataclass_init", "user_custom_init"],
)
def test_construct(benchmark, ctor):
    """Construct each Value flavor from the same array.

    - ``dense_factory``: internal fast-path allocation (``_dense``) — bypasses
      the metaclass.
    - ``dense_value``: direct ``_DenseArrayValue(...)`` construction (through
      the metaclass).
    - ``user_dataclass_init``: user ``Value`` with a converter and
      dataclass-generated ``__init__``.
    - ``user_custom_init``: user ``Value`` with a custom ``__init__`` and
      ``__check_init__``.
    """
    benchmark(ctor, _arr)


# =============================================================================


_xm = MyArray(jnp.ones(64))


def _chain(a):
    for _ in range(25):
        a = a + a
        a = a * a
    return a


@pytest.mark.benchmark(group="trace")
def test_eager_quaxify_trace(benchmark):
    """Eager ``quaxify`` over a ~100-primitive chain (pure trace/dispatch)."""
    quax.quaxify(_chain)(_xm)  # warm any import-time caches
    benchmark(lambda: quax.quaxify(_chain)(_xm))
