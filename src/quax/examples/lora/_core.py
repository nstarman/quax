from typing import cast

import equinox as eqx
import jax.core
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree, Shaped

import quax


class LoraArray(quax.ArrayValue):
    """Replaces a matrix `w in R^{n x m}` with `w + a @ b`, where `a in R^{n x k}` and
    `b in R^{k x m}`.

    Typically `k` is much smaller than `n` or `m`, and so `w + a @ b` is described as a
    "low rank adaptation" of `w`. The value of `k` is the "rank" of the adaptation.

    Note that this does not materialise the sum `w + a @ b` into a single matrix, but
    instead stores it as three separate `w`, `a`, `b` matrices. This is because the
    typical use-case for LoRA is to update just the `a` and `b` matrices when
    fine-tuning a neural network.

    This implementation makes use of Quax's multiple-dispatch capabilities to calculate
    matrix-vector products `(w + a @ b) @ x` via `w @ x + a @ (b @ x)`, which turns out
    to be computationally cheaper.
    """

    _w: Shaped[Array, "*batch x y"]
    a: Shaped[Array, "*batch x z"]
    b: Shaped[Array, "*batch z y"]
    stop_gradient: bool = eqx.field(static=True)
    allow_materialise: bool = eqx.field(static=True)

    def __init__(
        self,
        weight: Shaped[Array, "*batch x y"],
        *,
        rank: int,
        scale: float = 0.01,
        allow_materialise: bool = False,
        stop_gradient: bool = True,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `weight`: the original weight to wrap.
        - `rank`: the rank of the low-rank adaptation.
        - `scale`: `a` will be initialised at `Normal(0, scale^2)`. (`b` is initialised
            at zero.)
        - `allow_materialise`: if Quax encounters an operation for which there has not
            been a specific override specified for LoraArrays, should it either (a)
            throw an error (`allow_materialise=False`, the default), or (b) silently
            convert the `LoraArray` back into an JAX array, by explicitly calculating
            `w + a @ b` (`allow_materialise=True`).
        - `stop_gradient`: whether to automatically stop the gradient (prevent training)
            of the original weight matrix `weight`.
        - `key`: used to provide randomness for initialising `a`.
        """

        *batch, x, y = weight.shape
        self._w = weight
        self.a = jr.normal(key, (*batch, x, rank), dtype=weight.dtype) * scale
        self.b = jnp.zeros((*batch, rank, y), dtype=weight.dtype)
        self.stop_gradient = stop_gradient
        self.allow_materialise = allow_materialise

    @property
    def w(self):
        if self.stop_gradient:
            return lax.stop_gradient(self._w)
        else:
            return self._w

    def materialise(self):
        if self.allow_materialise:
            batch = tuple(range(self.w.ndim - 2))
            lhs_contract = (self.a.ndim - 1,)
            rhs_contract = (self.b.ndim - 2,)
            dimension_numbers = ((lhs_contract, rhs_contract), (batch, batch))
            return self.w + lax.dot_general(self.a, self.b, dimension_numbers)
        else:
            raise RuntimeError(
                "Refusing to materialise `LoraArray` with `allow_materialise=False`."
            )

    def aval(self):
        return jax.core.ShapedArray(self.w.shape, self.w.dtype)


def _is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def loraify(
    model: PyTree,
    *,
    rank: int,
    scale: float = 0.01,
    allow_materialise: bool = False,
    stop_gradient: bool = True,
    key: PRNGKeyArray,
) -> PyTree:
    """Converts an [Equinox](https://github.com/patrick-kidger/equinox) model into a
    low-rank adapted version.

    **Arguments:**

    - `model`: the model to convert. This is treated as a PyTree, and all
        `eqx.nn.Linear` layers found will have their weight matrices replaced with
        `LoraArray`s.
    - `rank`: the rank of the low-rank adaptation.
    - `scale`: how large to initialise the `a` matrix of the low-rank adaptation.
    - `allow_materialise`: if Quax encounters an operation for which there has not
        been a specific override specified for `LoraArray`s, should it either (a)
        throw an error (`allow_materialise=False`, the default), or (b) silently
        convert the `LoraArray` back into an JAX array, by explicitly calculating
        `w + a @ b` (`allow_materialise=True`).
    - `stop_gradient`: whether to automatically stop the gradient (prevent training)
        of the original weight matrices of the linear layers.
    - `key`: used to provide randomness for initialising the low-rank adaptation.

    **Returns:**

    A copy of `model`, will all linear layers having their weight matrices replaced with
    `LoraArray`s.

    Typically, the result should then be used with a call to `quax.quaxify`, which will
    trace your JAX program, and replace all interactions with LoRA arrays using the
    appropriate multiple dispatch rules.

    !!! Example

        ```python
        import equinox as eqx
        import quax
        import jax.random as jr

        key = jr.PRNGKey(0)
        mlp = eqx.nn.MLP(...)
        mlp = quax.lora.loraify(mlp, rank=2, key=key)
        # Wrap in `quaxify` and call as normal.
        some_output = quax.quaxify(mlp)(some_input)
        ```
    """

    def _loraify(x):
        nonlocal key
        if _is_linear(x):
            key, subkey = jr.split(key)
            lora_weight = LoraArray(
                x.weight,
                rank=rank,
                scale=scale,
                stop_gradient=stop_gradient,
                allow_materialise=allow_materialise,
                key=subkey,
            )
            return eqx.tree_at(lambda l: l.weight, x, lora_weight)
        else:
            return x

    # Note that we do not automatically wrap in `quaxify`, as we don't want to privilege
    # `__call__` over any other method of the model.
    return jtu.tree_map(_loraify, model, is_leaf=_is_linear)


@quax.quaxify
def _lora_array_matmul_impl(w, a, b, rhs, lhs_batch, ndim, dimension_numbers, kwargs):
    n_sharedbatch = len(lhs_batch)  # = len(rhs_batch)
    # All of the lora batch dimensions that aren't a dot_general batch dimension.
    n_lorabatch = ndim - n_sharedbatch - 2
    assert n_lorabatch >= 0
    out1 = lax.dot_general(w, rhs, dimension_numbers, **kwargs)
    # out1 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    out2 = lax.dot_general(b, rhs, dimension_numbers, **kwargs)
    # out2 has shape(*sharedbatch, *lorabatch, z, *otherbatch)
    lhs_contract2 = (w.ndim - 1,)
    rhs_contract2 = (n_sharedbatch + n_lorabatch,)
    rhs_batch2 = tuple(range(n_sharedbatch + n_lorabatch))
    lhs_batch2 = lhs_batch + tuple(i for i in rhs_batch2 if i not in lhs_batch)
    dimension_numbers2 = ((lhs_contract2, rhs_contract2), (lhs_batch2, rhs_batch2))
    out3 = lax.dot_general(a, out2, dimension_numbers2, **kwargs)
    # out3 has shape (*sharedbatch, *lorabatch, x, *otherbatch)
    return out1 + out3


@quax.register(lax.dot_general_p)
def _lora_array_matmul(
    lhs: LoraArray,
    rhs: ArrayLike | quax.ArrayValue,
    *,
    dimension_numbers,
    **kwargs,
) -> ArrayLike | quax.ArrayValue:
    rhs = cast("Array", rhs)  # quax(jax) is type broadened.
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    [ndim] = {lhs.a.ndim, lhs.b.ndim, lhs.w.ndim}
    if lhs_contract == (ndim - 1,) and (ndim - 2 not in lhs_batch):
        return _lora_array_matmul_impl(
            lhs.w, lhs.a, lhs.b, rhs, lhs_batch, ndim, dimension_numbers, kwargs
        )
    elif lhs_contract == (ndim - 2,) and (ndim - 1 not in lhs_batch):
        T = lambda x: jnp.swapaxes(x, -1, -2)
        lhs_contract = (ndim - 1,)
        dimension_numbers = ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))
        return _lora_array_matmul_impl(
            T(lhs.w),
            T(lhs.b),
            T(lhs.a),
            rhs,
            lhs_batch,
            ndim,
            dimension_numbers,
            kwargs,
        )
    else:
        return quax.quaxify(lax.dot_general)(
            lhs.materialise(), rhs, dimension_numbers, **kwargs
        )


@quax.register(lax.dot_general_p)
def _(
    lhs: ArrayLike | quax.ArrayValue,
    rhs: LoraArray,
    *,
    dimension_numbers,
    **kwargs,
) -> ArrayLike | quax.ArrayValue:
    ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = dimension_numbers
    dimension_numbers_flipped = ((rhs_contract, lhs_contract), (rhs_batch, lhs_batch))
    out = _lora_array_matmul(
        rhs, lhs, dimension_numbers=dimension_numbers_flipped, **kwargs
    )
    # out has shape (*sharedbatch, *rhs_uncontracted, *lhs_uncontracted)
    n_sharedbatch = len(lhs_batch)
    n_rhs_uncontracted = rhs.aval().ndim - len(rhs_contract) - len(rhs_batch)
    src = tuple(range(n_sharedbatch, n_sharedbatch + n_rhs_uncontracted))
    dest = tuple(range(-n_rhs_uncontracted, 0))
    return quax.quaxify(jnp.moveaxis)(out, src, dest)  # pyright: ignore[reportArgumentType,reportAssignmentType]
