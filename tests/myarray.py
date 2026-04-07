"""Test with :class:`MyArray` inputs."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, TypeGuard
from typing_extensions import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Bool

from quax import ArrayValue, quaxify, register
from quax._compat import JAX_VERSION, typeof


class MyArray(ArrayValue):
    """A :class:`quax.ArrayValue` that is dense.

    This is different from :class:`quax.MyArray` only in that
    `quax` will not attempt to convert it to a JAX array.
    """

    array: jax.Array = eqx.field(converter=jnp.asarray)

    def materialise(self) -> jax.Array:
        """Convert to a JAX array."""
        raise NotImplementedError

    def aval(self) -> jax.core.ShapedArray:
        """Return the ShapedArray."""
        return typeof(self.array)

    def astype(self, dtype: Any) -> Self:
        """Cast to type."""
        return replace(self, array=self.array.astype(dtype))

    def __getitem__(self, key: Any) -> Self:
        """Get item."""
        return MyArray(self.array[key])

    @property
    def size(self) -> int:
        """Get the size."""
        return self.array.size

    def __len__(self) -> int:
        """Get the length."""
        return self.array.shape[0] if self.array.ndim > 0 else 0

    def __lt__(self, other: Any) -> Bool[Array, "..."]:
        """Less than operator."""
        return self.array < other

    def __le__(self, other: Any) -> Bool[Array, "..."]:
        """Less than or equal operator."""
        return self.array <= other

    def __ge__(self, other: Any) -> Bool[Array, "..."]:
        """Greater than or equal operator."""
        return self.array >= other

    def __gt__(self, other: Any) -> Bool[Array, "..."]:
        """Greater than operator."""
        return self.array > other

    def __rmul__(self, other: Any) -> Self:
        """Multiplication operator."""
        return replace(self, array=other * self.array)

    def __add__(self, other: Any) -> Self:
        """Addition operator."""
        return quaxify(jnp.add)(self, other)

    def sum(self, **kw: Any) -> Self:
        """Sum the array."""
        return MyArray(self.array.sum(**kw))


def is_myarray(x: Any, /) -> TypeGuard[MyArray]:
    """Check if the object is a MyArray."""
    return isinstance(x, MyArray)


def unwrap(x: MyArray | ArrayLike) -> jax.Array:
    """Unwrap the array."""
    return x.array if is_myarray(x) else x


# ==============================================================================


@register(lax.abs_p)
def abs_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.abs(x.array))


# ==============================================================================


@register(lax.acos_p)
def acos_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.acos(x.array))


# ==============================================================================


@register(lax.acosh_p)
def acosh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.acosh(x.array))


# ==============================================================================


@register(lax.add_p)
def add_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.add(x.array, unwrap(y)))


@register(lax.add_p)
def add_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.add(x, y.array))


# ==============================================================================


@register(lax.after_all_p)
def after_all_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_gather_p)
def all_gather_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_to_all_p)
def all_to_all_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.and_p)
def and_p(x1: MyArray, x2: MyArray | ArrayLike, /) -> MyArray:
    return MyArray(lax.and_p.bind(x1.array, unwrap(x2)))


# ==============================================================================


@register(lax.approx_top_k_p)
def approx_top_k_p(x: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(t) for t in lax.approx_top_k_p.bind(x.array, **kw)]


# ==============================================================================


@register(lax.argmax_p)
def argmax_p(operand: MyArray, *, axes: Any, index_dtype: Any) -> MyArray:
    return replace(operand, array=lax.argmax(operand.array, axes[0], index_dtype))


# ==============================================================================


@register(lax.argmin_p)
def argmin_p(operand: MyArray, *, axes: Any, index_dtype: Any) -> MyArray:
    return replace(operand, array=lax.argmin(operand.array, axes[0], index_dtype))


# ==============================================================================


@register(lax.asin_p)
def asin_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.asin(x.array))


# ==============================================================================


@register(lax.asinh_p)
def asinh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.asinh(x.array))


# ==============================================================================


@register(lax.atan2_p)
def atan2_p_m(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.atan2(x.array, y.array))


@register(lax.atan2_p)
def atan2_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.atan2(x, y.array))


# ==============================================================================


@register(lax.atan_p)
def atan_p(x: MyArray) -> MyArray:
    return MyArray(lax.atan(x.array))


# ==============================================================================


@register(lax.atanh_p)
def atanh_p(x: MyArray) -> MyArray:
    return MyArray(lax.atanh(x.array))


# ==============================================================================


@register(lax.axis_index_p)
def axis_index_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i0e_p)
def bessel_i0e_p(x: MyArray, /) -> MyArray:
    return replace(x, array=lax.bessel_i0e(x.array))


# ==============================================================================


@register(lax.bessel_i1e_p)
def bessel_i1e_p(x: MyArray, /) -> MyArray:
    return replace(x, array=lax.bessel_i1e(x.array))


# ==============================================================================


@register(lax.bitcast_convert_type_p)
def bitcast_convert_type_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.bitcast_convert_type_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.broadcast_in_dim_p)
def broadcast_in_dim_p(operand: MyArray, **kw: Any) -> MyArray:
    return replace(operand, array=lax.broadcast_in_dim_p.bind(operand.array, **kw))


# ==============================================================================


@register(lax.cbrt_p)
def cbrt_p(x: MyArray, /, **kw: Any) -> MyArray:
    return MyArray(lax.cbrt(x.array, **kw))


# ==============================================================================


@register(lax.ceil_p)
def ceil_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.ceil(x.array))


# ==============================================================================


@register(lax.clamp_p)
def clamp_p_ama(min: ArrayLike, x: MyArray, max: ArrayLike) -> MyArray:
    return replace(x, array=lax.clamp_p.bind(min, x.array, max))


@register(lax.clamp_p)
def clamp_p_m(min: MyArray, x: MyArray, max: MyArray) -> MyArray:
    return replace(x, array=lax.clamp_p.bind(min.array, x.array, max.array))


# ==============================================================================


@register(lax.clz_p)
def clz_p(x: MyArray, /) -> MyArray:
    """Count leading zeros."""
    return replace(x, array=lax.clz_p.bind(x.array))


# ==============================================================================


@register(lax.complex_p)
def complex_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.complex(x.array, y.array))


# ==============================================================================


@register(lax.concatenate_p)
def concatenate_p_m(
    operand0: MyArray, *operands: MyArray | ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(
        lax.concatenate([operand0.array] + [unwrap(op) for op in operands], **kw)
    )


@register(lax.concatenate_p)
def concatenate_p_am(
    operand0: ArrayLike, operand1: MyArray, *operands: MyArray | ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(
        lax.concatenate_p.bind(
            operand0, operand1.array, *[unwrap(op) for op in operands], **kw
        )
    )


# ==============================================================================


@register(lax.cond_p)  # TODO: implement
def cond_p(index, consts) -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.conj_p)
def conj_p(x: MyArray, **kw: Any) -> MyArray:
    return replace(x, array=lax.conj_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.conv_general_dilated_p)
def conv_general_dilated_p(
    arg0: MyArray, arg1: MyArray | ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(lax.conv_general_dilated_p.bind(arg0.array, unwrap(arg1), **kw))


# ==============================================================================


@register(lax.convert_element_type_p)
def convert_element_type_myarray(operand: MyArray, **kw: Any) -> MyArray:
    return replace(operand, array=lax.convert_element_type_p.bind(operand.array, **kw))


# ==============================================================================


@register(lax.copy_p)
def copy_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.copy_p.bind(x.array))


# ==============================================================================


@register(lax.cos_p)
def cos_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.cos(x.array, **kw))


# ==============================================================================


@register(lax.cosh_p)
def cosh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.cosh(x.array))


# ==============================================================================


@register(lax.create_token_p)
def create_token_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.cumlogsumexp_p)
def cumlogsumexp_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    # TODO: double check units make sense here.
    return replace(
        operand,
        array=lax.cumlogsumexp(operand.array, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cummax_p)
def cummax_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cummax(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cummin_p)
def cummin_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cummin(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cumprod_p)
def cumprod_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(
        operand,
        array=lax.cumprod(operand.array, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cumsum_p)
def cumsum_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cumsum(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.device_put_p)
def device_put_p(x: MyArray, **kw: Any) -> MyArray:
    return replace(x, array=jax.device_put(x.array, **kw))


# ==============================================================================


@register(lax.digamma_p)
def digamma_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.digamma(x.array))


# ==============================================================================


@register(lax.div_p)
def div_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.div(x.array, unwrap(y)))


# ==============================================================================


@register(lax.dot_general_p)  # TODO: implement
def dot_general_p(lhs: MyArray, rhs: MyArray, **kw: Any) -> MyArray:
    return MyArray(lax.dot_general_p.bind(lhs.array, rhs.array, **kw))


# ==============================================================================


@register(lax.dynamic_slice_p)
def dynamic_slice_p(operand: MyArray, *args: MyArray | ArrayLike, **kw: Any) -> MyArray:
    return MyArray(
        lax.dynamic_slice_p.bind(operand.array, *[unwrap(a) for a in args], **kw)
    )


# ==============================================================================


@register(lax.dynamic_update_slice_p)
def _(
    arg0: MyArray, arg1: MyArray, arg2: ArrayLike, arg3: ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(
        lax.dynamic_update_slice_p.bind(arg0.array, arg1.array, arg2, arg3, **kw)
    )


@register(lax.dynamic_update_slice_p)
def _(
    arg0: ArrayLike, arg1: MyArray, arg2: ArrayLike, arg3: ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(lax.dynamic_update_slice_p.bind(arg0, arg1.array, arg2, arg3, **kw))


@register(lax.dynamic_update_slice_p)
def _(arg0: ArrayLike, arg1: MyArray, arg2: ArrayLike, **kw: Any) -> MyArray:
    return MyArray(lax.dynamic_update_slice_p.bind(arg0, arg1.array, arg2, **kw))


@register(lax.dynamic_update_slice_p)
def _(arg0: MyArray, arg1: MyArray, arg2: MyArray, **kw: Any) -> MyArray:
    return MyArray(
        lax.dynamic_update_slice_p.bind(arg0.array, arg1.array, arg2.array, **kw)
    )


# ==============================================================================


@register(lax.eq_p)
def eq_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.eq(x.array, unwrap(y)))


# ==============================================================================


@register(lax.eq_to_p)
def eq_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.erf_inv_p)
def erf_inv_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erf_inv(x.array))


# ==============================================================================


@register(lax.erf_p)
def erf_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erf(x.array))


# ==============================================================================


@register(lax.erfc_p)
def erfc_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erfc(x.array))


# ==============================================================================


@register(lax.exp2_p)
def exp2_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.exp2(x.array, **kw))


# ==============================================================================


@register(lax.exp_p)
def exp_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.exp(x.array, **kw))


# ==============================================================================


@register(lax.expm1_p)
def expm1_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.expm1(x.array, **kw))


# ==============================================================================


@register(lax.fft_p)
def fft_p(x: MyArray, *, fft_type: Any, fft_lengths: Any) -> MyArray:
    return replace(x, array=lax.fft(x.array, fft_type, fft_lengths))


# ==============================================================================


@register(lax.floor_p)
def floor_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.floor(x.array))


# ==============================================================================


@register(lax.gather_p)
def gather_p(
    operand: MyArray, start_indices: MyArray | ArrayLike, **kw: Any
) -> MyArray:
    return MyArray(lax.gather(operand.array, unwrap(start_indices), **kw))


# ==============================================================================


@register(lax.ge_p)
def ge_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.ge(x.array, unwrap(y)))


@register(lax.ge_p)
def ge_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.ge(x, y.array))


# ==============================================================================


@register(lax.gt_p)
def gt_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.gt(x.array, unwrap(y)))


# ==============================================================================


@register(lax.igamma_grad_a_p)
def igamma_grad_a_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.igamma_p)
def igamma_p(a: int | float, x: MyArray) -> MyArray:
    return replace(x, array=lax.igamma_p.bind(a, x.array))


# ==============================================================================


@register(lax.igammac_p)
def igammac_p(a: int | float, x: MyArray) -> MyArray:
    return replace(x, array=lax.igammac_p.bind(a, x.array))


# ==============================================================================


@register(lax.imag_p)
def imag_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.imag(x.array))


# ==============================================================================


@register(lax.integer_pow_p)
def integer_pow_p(x: MyArray, *, y: Any) -> MyArray:
    return replace(x, array=lax.integer_pow(x.array, y))


# ==============================================================================


# @register(lax.iota_p)
# def iota_p(dtype: MyArray) -> MyArray:
#     raise NotImplementedError


# ==============================================================================


@register(lax.is_finite_p)
def is_finite_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.is_finite(x.array))


# ==============================================================================


@register(lax.le_p)
def le_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.le(x.array, unwrap(y)))


# ==============================================================================


@register(lax.le_to_p)
def le_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.lgamma_p)
def lgamma_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.lgamma(x.array))


# ==============================================================================


@register(lax.linear_solve_p)
def linear_solve_p(
    arg0: MyArray,
    arg1: MyArray,
    arg2: MyArray,
    arg3: MyArray,
    arg4: MyArray,
    arg5: MyArray,
    arg6: ArrayLike,
    **kw: Any,
) -> MyArray:
    return MyArray(
        lax.linear_solve_p.bind(
            arg0.array,
            arg1.array,
            arg2.array,
            arg3.array,
            arg4.array,
            arg5.array,
            arg6,
            **kw,
        )
    )


# ==============================================================================


@register(lax.log1p_p)
def log1p_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.log1p(x.array, **kw))


# ==============================================================================


@register(lax.log_p)
def log_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.log(x.array, **kw))


# ==============================================================================


@register(lax.logistic_p)
def logistic_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.logistic(x.array, **kw))


# ==============================================================================


@register(lax.lt_p)
def lt_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.lt(x.array, unwrap(y)))


@register(lax.lt_p)
def lt_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.lt(x, y.array))


# ==============================================================================


@register(lax.lt_to_p)
def lt_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.max_p)
def max_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.max(x.array, unwrap(y)))


@register(lax.max_p)
def max_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.max(x, y.array))


# ==============================================================================


@register(lax.min_p)
def min_p_m(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.min_p.bind(x.array, y.array))


@register(lax.min_p)
def min_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.min_p.bind(x, y.array))


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def mul_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.mul_p.bind(x.array, unwrap(y)))


@register(lax.mul_p)
def mul_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.mul_p.bind(x, y.array))


# ==============================================================================


@register(lax.ne_p)
def ne_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.ne(x.array, unwrap(y)))


@register(lax.ne_p)
def ne_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.ne(x, y.array))


# ==============================================================================


@register(lax.neg_p)
def neg_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.neg(x.array))


# ==============================================================================


@register(lax.nextafter_p)
def nextafter_p(arg0: MyArray, arg1: MyArray) -> MyArray:
    return MyArray(lax.nextafter_p.bind(arg0.array, arg1.array))


# ==============================================================================


@register(lax.not_p)
def not_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.bitwise_not(x.array))


# ==============================================================================


@register(lax.or_p)
def or_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return replace(x, array=lax.bitwise_or(x.array, unwrap(y)))


# ==============================================================================


@register(lax.pad_p)
def pad_p(x: MyArray, v: ArrayLike, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.pad_p.bind(x.array, v, **kw))


# ==============================================================================


@register(lax.pmax_p)
def pmax_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmin_p)
def pmin_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.polygamma_p)
def polygamma_p(a: float | int, x: MyArray) -> MyArray:
    return replace(x, array=lax.polygamma_p.bind(a, x.array))


# ==============================================================================


@register(lax.population_count_p)
def population_count_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.population_count_p.bind(x.array))


# ==============================================================================


@register(lax.pow_p)
def pow_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(array=lax.pow(x.array, unwrap(y)))


@register(lax.pow_p)
def pow_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(array=lax.pow(x, y.array))


# ==============================================================================


@register(lax.ppermute_p)
def ppermute_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.psum_p)
def psum_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================

if JAX_VERSION <= (0, 6, 0):

    @register(lax.random_gamma_grad_p)
    def random_gamma_grad_p(a: float | int, x: MyArray) -> MyArray:
        return replace(x, array=lax.random_gamma_grad_p.bind(a, x.array))


# ==============================================================================


@register(lax.real_p)
def real_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.real(x.array))


# ==============================================================================


@register(lax.reduce_and_p)
def reduce_and_p(
    operand: MyArray,
    *,
    axes: Sequence[int],
) -> Any:
    return lax.reduce_and_p.bind(operand.array, axes=tuple(axes))


# ==============================================================================


@register(lax.reduce_max_p)
def reduce_max_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.reduce_max_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.reduce_min_p)
def reduce_min_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.reduce_min_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.reduce_or_p)
def reduce_or_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.reduce_or_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.reduce_p)
def reduce_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_precision_p)
def reduce_precision_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_prod_p)
def reduce_prod_p(x: MyArray, /, **kw) -> MyArray:
    return replace(x, array=lax.reduce_prod_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.reduce_sum_p)
def reduce_sum_p(
    x: MyArray, *, axes: tuple[int, ...], out_sharding: Any = None
) -> MyArray:
    if JAX_VERSION >= (0, 8, 0):
        array = lax.reduce_sum_p.bind(x.array, axes=axes, out_sharding=out_sharding)
    else:
        array = lax.reduce_sum_p.bind(x.array, axes=axes)

    return replace(x, array=array)


# ==============================================================================


@register(lax.reduce_window_max_p)
def reduce_window_max_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_min_p)
def reduce_window_min_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_p)
def reduce_window_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_sum_p)
def reduce_window_sum_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_xor_p)
def reduce_xor_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.regularized_incomplete_beta_p)
def regularized_incomplete_beta_p(
    a: float, x: MyArray, y: MyArray, /, **kw: Any
) -> MyArray:
    return replace(
        x, array=lax.regularized_incomplete_beta_p.bind(a, x.array, y.array, **kw)
    )


# ==============================================================================


@register(lax.rem_p)
def rem_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.rem(x.array, unwrap(y)))


@register(lax.rem_p)
def rem_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.rem(x, y.array))


# ==============================================================================


@register(lax.reshape_p)
def reshape_p(operand: MyArray, **kw: Any) -> MyArray:
    return replace(operand, array=lax.reshape_p.bind(operand.array, **kw))


# ==============================================================================


@register(lax.rev_p)
def rev_p(operand: MyArray, *, dimensions: Any) -> MyArray:
    return replace(operand, array=lax.rev(operand.array, dimensions))


# ==============================================================================


@register(lax.rng_bit_generator_p)
def rng_bit_generator_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.rng_uniform_p)
def rng_uniform_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.round_p)
def round_p(x: MyArray, *, rounding_method: Any) -> MyArray:
    return replace(x, array=lax.round(x.array, rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def rsqrt_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.rsqrt_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.scan_p)
def scan_p_m(arg0: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.scan_p.bind(arg0.array, **kw)]


@register(lax.scan_p)
def _(a0: ArrayLike, a1: int, a2: MyArray, a3: bool, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.scan_p.bind(a0, a1, a2.array, a3, **kw)]


@register(lax.scan_p)
def _(
    a0: ArrayLike, a1: ArrayLike, a2: int, a3: MyArray, a4: bool, /, **kw: Any
) -> list[MyArray]:
    return [MyArray(x) for x in lax.scan_p.bind(a0, a1, a2, a3.array, a4, **kw)]


@register(lax.scan_p)
def _(
    a0: MyArray, a1: MyArray, a2: ArrayLike, a3: ArrayLike, /, **kw: Any
) -> list[MyArray]:
    return [MyArray(x) for x in lax.scan_p.bind(a0.array, a1.array, a2, a3, **kw)]


# ==============================================================================


@register(lax.scatter_add_p)
def scatter_add_p_m(
    operand: MyArray,
    scatter_indices: MyArray | ArrayLike,
    updates: MyArray | ArrayLike,
    **kw: Any,
) -> MyArray:
    return MyArray(
        lax.scatter_add_p.bind(
            operand.array, unwrap(scatter_indices), unwrap(updates), **kw
        ),
    )


@register(lax.scatter_add_p)
def scatter_add_p_ama(
    operand: ArrayLike,
    scatter_indices: MyArray,
    updates: ArrayLike,
    **kw: Any,
) -> MyArray:
    return MyArray(
        lax.scatter_add_p.bind(operand, scatter_indices.array, updates, **kw),
    )


# ==============================================================================


@register(lax.scatter_max_p)
def scatter_max_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_min_p)
def scatter_min_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_mul_p)
def scatter_mul_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_p)
def _(arg0: ArrayLike, arg1: MyArray, arg2: ArrayLike, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0, arg1.array, arg2, **kw))


@register(lax.scatter_p)
def _(arg0: ArrayLike, arg1: ArrayLike, arg2: MyArray, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0, arg1, arg2.array, **kw))


@register(lax.scatter_p)
def _(arg0: MyArray, arg1: ArrayLike, arg2: ArrayLike, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0.array, arg1, arg2, **kw))


@register(lax.scatter_p)
def _(arg0: MyArray, arg1: MyArray, arg2: ArrayLike, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0.array, arg1.array, arg2, **kw))


@register(lax.scatter_p)
def _(arg0: ArrayLike, arg1: MyArray, arg2: MyArray, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0, arg1.array, arg2.array, **kw))


@register(lax.scatter_p)
def _(arg0: MyArray, arg1: ArrayLike, arg2: MyArray, /, **kw: Any) -> MyArray:
    return MyArray(lax.scatter_p.bind(arg0.array, arg1, arg2.array, **kw))


# ==============================================================================


@register(lax.select_and_gather_add_p)
def select_and_gather_add_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_add_p)
def select_and_scatter_add_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_p)
def select_and_scatter_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_n_p)
def select_n_p_m(which: MyArray | ArrayLike, *cases: MyArray) -> MyArray:
    return MyArray(lax.select_n(unwrap(which), *[c.array for c in cases]))


@register(lax.select_n_p)
def _(which: ArrayLike, case0: ArrayLike, case1: MyArray) -> MyArray:
    return MyArray(lax.select_n(which, case0, case1.array))


@register(lax.select_n_p)
def _(which: ArrayLike, case0: MyArray, case1: ArrayLike) -> MyArray:
    return MyArray(lax.select_n(which, case0.array, case1))


@register(lax.select_n_p)
def _(which: MyArray, case0: MyArray, case1: ArrayLike) -> MyArray:
    return MyArray(lax.select_n(which.array, case0.array, case1))


@register(lax.select_n_p)
def _(which: MyArray, case0: ArrayLike, case1: ArrayLike) -> MyArray:
    return MyArray(lax.select_n(which.array, case0, case1))


@register(lax.select_n_p)
def _(which: MyArray, case0: ArrayLike, case1: MyArray) -> MyArray:
    return MyArray(lax.select_n(which.array, case0, case1.array))


# ==============================================================================


@register(lax.sharding_constraint_p)
def sharding_constraint_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_left_p)
def shift_left_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.shift_left_p.bind(x.array, unwrap(y)))


# ==============================================================================


@register(lax.shift_right_arithmetic_p)
def shift_right_arithmetic_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.shift_right_arithmetic_p.bind(x.array, unwrap(y)))


# ==============================================================================


@register(lax.shift_right_logical_p)
def shift_right_logical_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.shift_right_logical_p.bind(x.array, y))


# ==============================================================================


@register(lax.sign_p)
def sign_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sign(x.array))


# ==============================================================================


@register(lax.sin_p)
def sin_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.sin(x.array, **kw))


# ==============================================================================


@register(lax.sinh_p)
def sinh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sinh(x.array))


# ==============================================================================


@register(lax.slice_p)
def slice_p(
    operand: MyArray,
    *,
    start_indices: Any,
    limit_indices: Any,
    strides: Any,
) -> MyArray:
    return replace(
        operand,
        array=lax.slice_p.bind(
            operand.array,
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        ),
    )


@register(lax.split_p)
def split_p(x: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.split_p.bind(x.array, **kw)]


# ==============================================================================


@register(lax.sort_p)
def sort_p_m(*args: MyArray, **kw: Any) -> list[MyArray]:
    args = [arg.array for arg in args]
    return [MyArray(x) for x in lax.sort_p.bind(*args, **kw)]


@register(lax.sort_p)
def sort_p_ma(arg0: MyArray, arg1: ArrayLike, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.sort_p.bind(arg0.array, arg1, **kw)]


@register(lax.sort_p)
def sort_p_mma(
    arg0: MyArray, arg1: MyArray, arg2: ArrayLike, /, **kw: Any
) -> list[MyArray]:
    return [MyArray(x) for x in lax.sort_p.bind(arg0.array, arg1.array, arg2, **kw)]


# ==============================================================================


@register(lax.square_p)
def square(x: MyArray) -> MyArray:
    return replace(x, array=lax.square_p.bind(x.array))


# ==============================================================================


@register(lax.sqrt_p)
def sqrt_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.sqrt_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.squeeze_p)
def squeeze_p(x: MyArray, **kw: Any) -> MyArray:
    return replace(x, array=lax.squeeze_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.stop_gradient_p)
def stop_gradient_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.stop_gradient_p.bind(x.array))


# ==============================================================================
# Subtraction


@register(lax.sub_p)
def sub_p_m(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.sub_p.bind(x.array, unwrap(y)))


@register(lax.sub_p)
def sub_p_am(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.sub_p.bind(x, y.array))


# ==============================================================================


@register(lax.tan_p)
def tan_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.tan_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.tanh_p)
def tanh_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.tanh_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.tile_p)
def tile_p(x: MyArray, /, **kw: Any) -> MyArray:
    return replace(x, array=lax.tile_p.bind(x.array, **kw))


# ==============================================================================

if JAX_VERSION >= (0, 9, 0):

    @register(lax.top_k_p)
    def top_k_p(operand: MyArray, /, **kw: Any) -> MyArray:
        return [MyArray(x) for x in lax.top_k(operand.array, **kw)]


# ==============================================================================


@register(lax.transpose_p)
def transpose_p(operand: MyArray, /, **kw: Any) -> MyArray:
    return replace(operand, array=lax.transpose_p.bind(operand.array, **kw))


# ==============================================================================


@register(lax.while_p)
def while_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.xor_p)
def xor_p(x: MyArray, y: MyArray | ArrayLike) -> MyArray:
    return MyArray(lax.xor_p.bind(x.array, y.array if isinstance(y, MyArray) else y))


# ==============================================================================


@register(lax.zeta_p)
def zeta_p(x: MyArray, q: ArrayLike) -> MyArray:
    return replace(x, array=lax.zeta_p.bind(x.array, q))


# ==============================================================================


@register(lax.linalg.cholesky_p)
def cholesky_p(x: MyArray, **kw: Any) -> MyArray:
    return replace(x, array=lax.linalg.cholesky_p.bind(x.array, **kw))


# ==============================================================================


@register(lax.linalg.eig_p)
def eig_p(x: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.linalg.eig_p.bind(x.array, **kw)]


# ==============================================================================


@register(lax.linalg.eigh_p)
def eigh_p(x: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.linalg.eigh_p.bind(x.array, **kw)]


# ==============================================================================


@register(lax.linalg.hessenberg_p)
def hessenberg_p(x: MyArray, /) -> MyArray:
    return [MyArray(x) for x in lax.linalg.hessenberg_p.bind(x.array)]


# ==============================================================================


@register(lax.linalg.lu_p)
def lu(x: MyArray, /) -> MyArray:
    return [MyArray(x) for x in lax.linalg.lu_p.bind(x.array)]


# ==============================================================================


@register(lax.linalg.householder_product_p)
def householder_product_p(a: MyArray, taus: MyArray, /) -> MyArray:
    return MyArray(lax.linalg.householder_product_p.bind(a.array, taus.array))


# ==============================================================================


@register(lax.linalg.triangular_solve_p)
def triangular_solve_p(arg0: MyArray, arg1: MyArray, /, **kw: Any) -> MyArray:
    return MyArray(lax.linalg.triangular_solve_p.bind(arg0.array, arg1.array, **kw))


# ==============================================================================


@register(lax.linalg.qr_p)
def qr_p(arg: MyArray, /, **kw: Any) -> MyArray:
    return [MyArray(x) for x in lax.linalg.qr_p.bind(arg.array, **kw)]


# ==============================================================================


@register(lax.linalg.schur_p)
def schur_p(arg: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.linalg.schur_p.bind(arg.array, **kw)]


# ==============================================================================


@register(lax.linalg.svd_p)
def svd_p(arg: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.linalg.svd_p.bind(arg.array, **kw)]


# ==============================================================================


@register(lax.linalg.tridiagonal_p)
def tridiagonal_p(arg: MyArray, /, **kw: Any) -> list[MyArray]:
    return [MyArray(x) for x in lax.linalg.tridiagonal_p.bind(arg.array, **kw)]
