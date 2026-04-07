import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.core as core
import jax.numpy as jnp
from jaxtyping import ArrayLike  # https://github.com/patrick-kidger/jaxtyping

import quax


class Dimension:
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
    array: ArrayLike
    units: dict[Dimension, int] = eqx.field(static=True, converter=_dim_to_unit)

    def aval(self):
        shape = jnp.shape(self.array)
        dtype = jnp.result_type(self.array)
        return core.ShapedArray(shape, dtype)

    def materialise(self):
        raise ValueError("Refusing to materialise Unitful array.")


@quax.register(jax.lax.add_p)
def _(x: Unitful, y: Unitful):  # function name doesn't matter
    if x.units == y.units:
        return Unitful(x.array + y.array, x.units)
    else:
        raise ValueError(f"Cannot add two arrays with units {x.units} and {y.units}.")


@quax.register(jax.lax.mul_p)
def _(x: Unitful, y: Unitful):
    units = x.units.copy()
    for k, v in y.units.items():
        if k in units:
            units[k] += v
        else:
            units[k] = v
    return Unitful(x.array * y.array, units)


@quax.register(jax.lax.mul_p)
def _(x: ArrayLike, y: Unitful):
    return Unitful(x * y.array, y.units)


@quax.register(jax.lax.mul_p)
def _(x: Unitful, y: ArrayLike):
    return Unitful(x.array * y, x.units)


@quax.register(jax.lax.integer_pow_p)
def _(x: Unitful, *, y: int):
    units = {k: v * y for k, v in x.units.items()}
    return Unitful(x.array, units)


@quax.register(jax.lax.lt_p)
def _(x: Unitful, y: Unitful, **kwargs):
    if x.units == y.units:
        return jax.lax.lt(x.array, y.array, **kwargs)
    else:
        raise ValueError(
            f"Cannot compare two arrays with units {x.units} and {y.units}."
        )


@quax.register(jax.lax.broadcast_in_dim_p)
def _(operand: Unitful, **kwargs):
    kwargs.pop("sharding", None)  # TODO: handle sharding
    new_arr = jax.lax.broadcast_in_dim(operand.array, **kwargs)
    return Unitful(new_arr, operand.units)
