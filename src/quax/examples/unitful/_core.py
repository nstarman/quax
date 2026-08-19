from typing import Any

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.core as core
import jax.numpy as jnp
from jaxtyping import ArrayLike  # https://github.com/patrick-kidger/jaxtyping

import quax


class Dimension:
    """A base physical dimension, e.g. metres.

    Three are provided ready-made: `kilograms`, `meters` and `seconds`. Make others
    by instantiating this class with a name.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


kilograms = Dimension("kg")
meters = Dimension("m")
seconds = Dimension("s")


def _dim_to_unit(x: Dimension | dict[Dimension, int]) -> dict[Dimension, int]:
    if isinstance(x, Dimension):
        return {x: 1}
    else:
        return x


class Unitful(quax.ArrayValue):
    """An array with physical units attached.

    Arithmetic propagates the units rather than checking them after the fact:
    multiplying two `Unitful`s adds their exponents, raising to an integer power
    multiplies them, and adding or comparing arrays whose units disagree raises.

    Refuses to [`quax.Value.materialise`][]. A primitive with no registered rule is
    therefore an error rather than a silent unit loss -- see
    [Sharp bits](../sharp-bits.md).

    **Arguments:**

    - `array`: the array to attach units to.
    - `units`: either a single `Dimension`, or a dict from `Dimension` to integer
        exponent -- e.g. `{meters: 1, seconds: -2}` for an acceleration.
    """

    array: ArrayLike
    units: dict[Dimension, int] = eqx.field(static=True, converter=_dim_to_unit)

    def aval(self):
        shape = jnp.shape(self.array)
        dtype = jnp.result_type(self.array)
        return core.ShapedArray(shape, dtype)

    def materialise(self):
        raise ValueError("Refusing to materialise Unitful array.")


@quax.register(jax.lax.add_p)
def add_unitful_unitful(x: Unitful, y: Unitful):
    if x.units == y.units:
        return Unitful(x.array + y.array, x.units)
    else:
        raise ValueError(f"Cannot add two arrays with units {x.units} and {y.units}.")


@quax.register(jax.lax.mul_p)
def mul_unitful_unitful(x: Unitful, y: Unitful, /, **kw: Any) -> Unitful:
    units = x.units.copy()
    for k, v in y.units.items():
        if k in units:
            units[k] += v
        else:
            units[k] = v
    return Unitful(jax.lax.mul_p.bind(x.array, y.array, **kw), units)


@quax.register(jax.lax.mul_p)
def mul_array_like_unitful(x: ArrayLike, y: Unitful, /, **kw: Any) -> Unitful:
    return Unitful(jax.lax.mul_p.bind(x, y.array, **kw), y.units)


@quax.register(jax.lax.mul_p)
def mul_unitful_array_like(x: Unitful, y: ArrayLike, /, **kw: Any) -> Unitful:
    return Unitful(jax.lax.mul_p.bind(x.array, y, **kw), x.units)


@quax.register(jax.lax.integer_pow_p)
def integer_pow_unitful(x: Unitful, *, y: int):
    units = {k: v * y for k, v in x.units.items()}
    return Unitful(jax.lax.integer_pow_p.bind(x.array, y=y), units)


@quax.register(jax.lax.lt_p)
def lt_unitful_unitful(x: Unitful, y: Unitful, **kwargs):
    if x.units == y.units:
        return jax.lax.lt(x.array, y.array, **kwargs)
    else:
        raise ValueError(
            f"Cannot compare two arrays with units {x.units} and {y.units}."
        )


@quax.register(jax.lax.broadcast_in_dim_p)
def broadcast_in_dim_unitful(operand: Unitful, **kwargs):
    kwargs.pop("sharding", None)  # TODO: handle sharding
    new_arr = jax.lax.broadcast_in_dim(operand.array, **kwargs)
    return Unitful(new_arr, operand.units)


@quax.register(jax.lax.copy_p)
def copy_unitful(x: Unitful, **kw: Any) -> Unitful:
    return Unitful(jax.lax.copy_p.bind(x.array, **kw), x.units)


@quax.register(jax.lax.select_n_p)
def select_n_unitful(which: ArrayLike, *cases: Unitful, **kw: Any) -> Unitful:
    units = cases[0].units
    for case in cases[1:]:
        if case.units != units:
            raise ValueError(
                f"Cannot select between arrays with units {units} and {case.units}."
            )
    arrays = [c.array for c in cases]
    return Unitful(jax.lax.select_n_p.bind(which, *arrays, **kw), units)
