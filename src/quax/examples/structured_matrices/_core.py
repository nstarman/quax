from typing import cast

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Shaped

import quax


class TridiagonalMatrix(quax.ArrayValue):
    """Represents a tridiagonal matrix, by storing just the values of its three
    diagonals.
    """

    lower_diag: Shaped[Array, "*batch size"]
    main_diag: Shaped[Array, "*batch size+1"]
    upper_diag: Shaped[Array, "*batch size"]
    allow_materialise: bool = eqx.field(default=False, static=True)

    def __check_init__(self):
        *batch1, size1 = self.lower_diag.shape
        *batch2, size2 = self.main_diag.shape
        *batch3, size3 = self.upper_diag.shape
        batches = {tuple(batch1), tuple(batch2), tuple(batch3)}
        if len(batches) != 1:
            raise ValueError(
                f"Got inconsistent batch dimensions {batches} for `TridiagonalMatrix`."
            )
        if size1 != size2 - 1 or size3 != size2 - 1:
            raise ValueError(
                "Got inconsistent diagonal lengths for `TridiagonalMatrix`."
            )
        if (
            self.lower_diag.dtype != self.main_diag.dtype
            or self.upper_diag.dtype != self.main_diag.dtype
        ):
            raise ValueError("Got inconsistent dtypes for `TridiagonalMatrix`.")

    def materialise(self):
        if self.allow_materialise:
            *batch, size = self.main_diag.shape
            out = jnp.zeros((*batch, size, size), dtype=self.main_diag.dtype)
            arange = jnp.arange(size)
            out = out.at[..., arange[1:], arange[:-1]].set(self.lower_diag)
            out = out.at[..., arange, arange].set(self.main_diag)
            return out.at[..., arange[:-1], arange[1:]].set(self.upper_diag)
        else:
            raise RuntimeError(
                "Refusing to materialise `TridiagonalMatrix` with "
                "`allow_materialise=False`."
            )

    def aval(self):
        *batch, size = self.main_diag.shape
        shape = (*batch, size, size)
        return jax.core.ShapedArray(shape, self.main_diag.dtype)


def _tridiagonal_matvec(
    lower_diag: Array, main_diag: Array, upper_diag: Array, vector: Array
):
    (size1,) = lower_diag.shape
    (size2,) = main_diag.shape
    (size3,) = upper_diag.shape
    (size4,) = vector.shape
    assert size1 == size2 - 1
    assert size3 == size2 - 1
    assert size4 == size2
    a = lower_diag * vector[:-1]
    b = main_diag * vector
    c = upper_diag * vector[1:]
    return b.at[:-1].add(c).at[1:].add(a)


@quax.register(lax.dot_general_p)
def _(
    lhs: TridiagonalMatrix,
    rhs: ArrayLike | quax.ArrayValue,
    *,
    dimension_numbers,
    **kwargs,
):
    rhs = cast("Array", rhs)  # quax(jax) is type broadened.
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    lhs_ndim = lhs.ndim
    if lhs_contract == (lhs_ndim - 1,) and (lhs_ndim - 2 not in lhs_batch):
        rhs_ndim = quax.quaxify(jnp.ndim)(rhs)
        lhs_used = set(lhs_contract + lhs_batch)
        lhs_used.add(lhs_ndim - 2)
        rhs_used = set(rhs_contract + rhs_batch)
        lhs_unused = [i for i in range(lhs_ndim) if i not in lhs_used]
        rhs_unused = [i for i in range(rhs_ndim) if i not in rhs_used]
        del lhs_used, rhs_used
        matvec = _tridiagonal_matvec
        for lhs_i, rhs_i in zip(lhs_batch, rhs_batch):
            matvec = jax.vmap(matvec, in_axes=(lhs_i, lhs_i, lhs_i, rhs_i))
        for lhs_i in lhs_unused:
            matvec = jax.vmap(matvec, in_axes=(lhs_i, lhs_i, lhs_i, None))
        for rhs_i in rhs_unused:
            matvec = jax.vmap(matvec, in_axes=(None, None, None, rhs_i), out_axes=-1)
        return quax.quaxify(matvec)(lhs.lower_diag, lhs.main_diag, lhs.upper_diag, rhs)
    else:
        return quax.quaxify(lax.dot_general)(
            lhs.materialise(), rhs, dimension_numbers, **kwargs
        )
