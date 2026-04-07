from typing import get_args

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Integer, Shaped

import quax


class BCOO(quax.ArrayValue):
    """Represents a sparse array stored in batch-coordinate format."""

    data: Shaped[Array, "*batch nse"]
    indices: Integer[Array, "*batch nse n_sparse"]
    _shape: tuple[int, ...] = eqx.field(static=True)  # pyright: ignore
    allow_materialise: bool = eqx.field(static=True)

    def __init__(
        self,
        data: Shaped[Array, "*batch nse"],
        indices: Integer[Array, "*batch nse n_sparse"],
        shape: tuple[int, ...],
        allow_materialise: bool = False,
    ):
        """A BCOO sparse array has `shape = (*batch, *n_sparse)`, indicating some number
        of batch dimensions, plus some number of sparse dimensions.

        `data` must have shape `(*batch, nse)`, where `nse` is the number of sparse
        elements.

        `indices` muat have shape `(*batch, nse, n_sparse)`. Ignoring the batch
        dimensions, its `i`-th index (down the `nse` dimension) will be the indices of
        the `i`-th nonzero element.
        """
        self.data = data
        self.indices = indices
        self._shape = shape
        self.allow_materialise = allow_materialise

    def __check_init__(self):
        *batch1, nse1 = self.data.shape
        *batch2, nse2, n_sparse = self.indices.shape
        batch3 = self.shape[: len(batch1)]
        batches = {tuple(batch1), tuple(batch2), tuple(batch3)}
        if len(batches) != 1:
            raise ValueError(f"Inconsistent batch sizes {batches} in `BCOO`.")
        if nse1 != nse2:
            raise ValueError("Inconsistent number of sparse entries in `BCOO`.")
        if len(self.shape[len(batch1) :]) != n_sparse:
            raise ValueError(
                "Must specify a sparse index for all non-batch dimensions in `BCOO`."
            )

    def aval(self):
        return jax.core.ShapedArray(self._shape, self.data.dtype)

    def materialise(self):
        if self.allow_materialise:
            zeros = jnp.zeros(self.shape, self.data.dtype)
            add = lambda z, i, d: z.at[i].add(d)
            return _op_sparse_to_dense(self, zeros, add)
        else:
            raise RuntimeError(
                "Refusing to materialise sparse matrix with `allow_materialise=False`."
            )

    def enable_materialise(self, allow_materialise: bool = True):
        return BCOO(self.data, self.indices, self.shape, allow_materialise)


def _op_sparse_to_dense(x, y, op):
    indices = tuple(jnp.moveaxis(x.indices, -1, 0))
    for _ in range(x.data.ndim - 1):
        op = jax.vmap(op)
    return op(y, indices, x.data)


@quax.register(lax.broadcast_in_dim_p)
def _(value: BCOO, *, broadcast_dimensions, shape, sharding=None) -> BCOO:
    n_extra_batch_dims = len(shape) - value.ndim
    if broadcast_dimensions != tuple(range(n_extra_batch_dims, len(shape))):
        raise NotImplementedError(
            "BCOO matrices only support broadcasting additional batch dimensions."
        )
    bdims = shape[:n_extra_batch_dims]
    dims = jnp.broadcast_shapes(
        (bdims + value.data.shape)[:-1],
        (bdims + value.indices.shape)[:-2],
        shape[: n_extra_batch_dims + len(value.data.shape) - 1],
    )
    data = jnp.broadcast_to(value.data, dims + value.data.shape[-1:])
    indices = jnp.broadcast_to(value.indices, dims + value.indices.shape[-2:])

    return BCOO(data, indices, shape, allow_materialise=value.allow_materialise)


@quax.register(lax.squeeze_p)
def _(x: BCOO, *, dimensions):
    batch_ndim = x.data.ndim - 1
    for i in dimensions:
        assert x.shape[i] == 1
        if i >= batch_ndim:
            raise NotImplementedError("Cannot squeeze out a sparse dimension.")
    data = lax.squeeze_p.bind(x.data, dimensions=dimensions)
    indices = lax.squeeze_p.bind(x.indices, dimensions=dimensions)
    shape = tuple(x.shape[i] for i in range(x.ndim) if i not in dimensions)
    return BCOO(data, indices, shape, allow_materialise=x.allow_materialise)  # pyright: ignore


@quax.register(lax.add_p)
def _(x: BCOO, y: BCOO):
    x_n_sparse = x.indices.shape[-1]
    y_n_sparse = y.indices.shape[-1]
    if x_n_sparse != y_n_sparse:
        raise NotImplementedError(
            "`BCOO(...) + BCOO(...)` currently requires that both matrices have the "
            "same number of sparse dimensions."
        )
    x, y = quax.quaxify(jnp.broadcast_arrays)(x, y)  # pyright: ignore[reportArgumentType,reportAssignmentType]
    assert isinstance(x, BCOO)
    assert isinstance(y, BCOO)
    data = jnp.concatenate([x.data, y.data], axis=-1)
    indices = jnp.concatenate([x.indices, y.indices], axis=-2)
    allow_materialise = x.allow_materialise and y.allow_materialise
    return BCOO(data, indices, x.shape, allow_materialise)


@quax.register(lax.add_p)
def _add_bcoo_dense(x: BCOO, y: ArrayLike) -> ArrayLike:
    x, y = quax.quaxify(jnp.broadcast_arrays)(x, y)  # pyright: ignore[reportArgumentType,reportAssignmentType]
    assert isinstance(x, BCOO)
    assert isinstance(y, get_args(ArrayLike))
    y = jnp.asarray(y)
    add = lambda z, i, d: z.at[i].add(d)
    return _op_sparse_to_dense(x, y, add)


@quax.register(lax.add_p)
def _(x: ArrayLike, y: BCOO) -> ArrayLike:
    return _add_bcoo_dense(y, x)


@quax.register(lax.mul_p)
def _(x: BCOO, y: BCOO):
    # This is actually surprisingly hard.
    raise NotImplementedError(
        "elementwise multiplication between two sparse matrices is not implemented"
    )


@quax.register(lax.mul_p)
def _mul_bcoo_dense(x: BCOO, y: ArrayLike) -> BCOO:
    x, y = quax.quaxify(jnp.broadcast_arrays)(x, y)  # pyright: ignore[reportArgumentType,reportAssignmentType]
    assert isinstance(x, BCOO)
    assert isinstance(y, get_args(ArrayLike))
    y = jnp.asarray(y)
    indices = tuple(jnp.moveaxis(x.indices, -1, 0))
    # TODO: unify with _sparse_to_dense, above?
    getindex = lambda a, b: a[b]
    for _ in range(x.data.ndim - 1):
        getindex = jax.vmap(getindex)
    # ~
    y_data = getindex(y, indices)
    data = x.data * y_data
    return BCOO(data, x.indices, x.shape, x.allow_materialise)


@quax.register(lax.mul_p)
def _(x: ArrayLike, y: BCOO) -> BCOO:
    return _mul_bcoo_dense(y, x)
