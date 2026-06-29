import abc
import functools as ft
from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax._src.prng
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Integer, UInt, UInt32

import quax


RealArray: TypeAlias = ArrayLike
DTypeLikeFloat: TypeAlias = Any
DTypeLikeInexact: TypeAlias = Any

PRNG_T = TypeVar("PRNG_T", bound="PRNG")


class PRNG(quax.ArrayValue):
    """Abstract base class for all custom PRNGs."""

    def materialise(self):
        raise TypeError(
            "PRNGs are only valid for certain operations, like `normal` or "
            "`jax.numpy.where`."
        )

    @abc.abstractmethod
    def random_bits(self, bit_width: int, shape: tuple[int, ...]) -> UInt[Array, "..."]:
        """Generate random bits from this PRNG. Must be implemented in subclasses."""

    @abc.abstractmethod
    def split(self: PRNG_T, num: int) -> Sequence[PRNG_T]:
        """Split this PRNG into multiple sub-PRNGs. Must be implemented in
        subclasses.
        """


class ThreeFry(PRNG):
    """Implements a threefry PRNG."""

    value: UInt32[Array, "*batch 2"]

    def __init__(self, seed: Integer[ArrayLike, ""]):
        self.value = jax._src.prng.threefry_seed(jnp.asarray(seed))  # pyright: ignore[reportPrivateImportUsage]

    def aval(self):
        *shape, _ = self.value.shape
        return jax.core.ShapedArray(shape, jnp.uint32)

    def random_bits(self, bit_width: int, shape: tuple[int, ...]) -> UInt[Array, "..."]:
        return jax._src.prng.threefry_random_bits(self.value, bit_width, shape)  # pyright: ignore[reportPrivateImportUsage]

    def split(self, num: int) -> Sequence["ThreeFry"]:
        new_values = jax._src.prng.threefry_split(self.value, (num,))  # pyright: ignore[reportPrivateImportUsage]
        return [eqx.tree_at(lambda s: s.value, self, x) for x in new_values]


ThreeFry.__init__.__doc__ = """**Arguments:**

- `seed`: an integer to use as the seed for the PRNG.
"""


def uniform(
    key: PRNG,
    shape: tuple[int, ...] = (),
    dtype: DTypeLikeFloat = jnp.float_,
    minval: RealArray = 0.0,
    maxval: RealArray = 1.0,
) -> Float[Array, ""]:
    """Samples a random number uniformly distributed over `[minval, maxval)`.

    Arguments as `jax.random.uniform`, except that the first argument must be one of our
    PRNGs, e.g. `prng.ThreeFry(...)`.
    """

    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("Must use floating dtype")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    minval = lax.convert_element_type(minval, dtype)
    maxval = lax.convert_element_type(maxval, dtype)
    minval = jnp.broadcast_to(minval, (1,) * (len(shape) - minval.ndim))
    maxval = jnp.broadcast_to(maxval, (1,) * (len(shape) - maxval.ndim))

    finfo = jnp.finfo(dtype)
    nbits = finfo.bits
    nmant = finfo.nmant
    if nbits == 16:
        uint_dtype = jnp.uint16
    elif nbits == 32:
        uint_dtype = jnp.uint32
    elif nbits == 64:
        uint_dtype = jnp.uint64
    else:
        raise NotImplementedError("Can only use 16-, 32- or 64-bit dtypes.")
    if nmant < 8:
        rng_bits = 8
    else:
        rng_bits = nbits
    bits = key.random_bits(rng_bits, shape)
    if rng_bits != nbits:
        bits = lax.convert_element_type(bits, uint_dtype)
    float_bits = lax.bitwise_or(
        lax.shift_right_logical(bits, np.array(rng_bits - nmant, uint_dtype)),
        np.array(1.0, dtype).view(uint_dtype),
    )
    floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1.0, dtype)
    return lax.max(minval, lax.reshape(floats * (maxval - minval) + minval, shape))


def normal(
    key: PRNG, shape: tuple[int, ...] = (), dtype: DTypeLikeInexact = jnp.float_
) -> Float[Array, ""]:
    """Samples from a normal distribution.

    Arguments as `jax.random.normal`, except that the first argument must be one of our
    PRNGs, e.g. `prng.ThreeFry(...)`.
    """

    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError("Must use floating dtype")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        sqrt2 = np.array(np.sqrt(2), dtype)
        key_re, key_im = split(key)
        real_dtype = np.array(0, dtype).real.dtype  # type: ignore[attr-defined]
        _re = _normal_real(key_re, shape, real_dtype).astype(dtype)
        _im = _normal_real(key_im, shape, real_dtype).astype(dtype)
        return (_re + 1j * _im) / sqrt2
    else:
        return _normal_real(key, shape, dtype)


def _normal_real(key, shape, dtype):
    lo = np.nextafter(np.array(-1.0, dtype), np.array(0.0, dtype), dtype=dtype)
    hi = np.array(1.0, dtype)
    u = uniform(key, shape, dtype, lo, hi)
    return lax.mul(np.array(np.sqrt(2), dtype), lax.erf_inv(u))


def split(key: PRNG_T, num: int = 2) -> Sequence[PRNG_T]:
    """Splits a key in multiple subkeys, each behaving statistically independently.

    Arguments as `jax.random.split`, except that the first argument must be one of our
    PRNGs, e.g. `prng.ThreeFry(...)`.
    """

    return key.split(num)


# Allows for `jnp.where(pred, key1, key2)`.
@quax.register(lax.select_n_p)
def _(pred, *cases: PRNG) -> PRNG:
    return jtu.tree_map(ft.partial(lax.select_n, pred), *cases)
