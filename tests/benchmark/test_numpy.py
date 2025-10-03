"""Benchmark tests for quaxed functions on quantities."""

from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict
from typing_extensions import Unpack

import jax
import jax.numpy as jnp
import pytest
from jax._src.stages import Compiled

import quax

from ..myarray import MyArray


Args: TypeAlias = tuple[Any, ...]

x = jnp.linspace(0, 1, 1000)
xm = MyArray(x)


def process_func(func: Callable[..., Any], args: Args) -> tuple[Compiled, Args]:
    """JIT and compile the function."""
    return jax.jit(quax.quaxify(func)), args


class ParameterizationKWArgs(TypedDict):
    """Keyword arguments for a pytest parameterization."""

    argvalues: list[tuple[Callable[..., Any], Args]]
    ids: list[str]


def process_pytest_argvalues(
    process_fn: Callable[[Callable[..., Any], Args], tuple[Callable[..., Any], Args]],
    argvalues: list[tuple[Callable[..., Any], Unpack[tuple[Args, ...]]]],
) -> ParameterizationKWArgs:
    """Process the argvalues."""
    # Get the ID for each parameterization
    get_types = lambda args: tuple(str(type(a)) for a in args)
    ids: list[str] = []
    processed_argvalues: list[tuple[Compiled, Args]] = []

    for func, *many_args in argvalues:
        for args in many_args:
            ids.append(f"{func.__name__}-{get_types(args)}")
            processed_argvalues.append(process_fn(func, args))

    # Process the argvalues and return the parameterization, with IDs
    return {"argvalues": processed_argvalues, "ids": ids}


funcs_and_args: list[tuple[Callable[..., Any], Unpack[tuple[Args, ...]]]] = [
    (jnp.abs, (xm,)),
    (jnp.acos, (xm,)),
    (jnp.acosh, (xm,)),
    (jnp.add, (xm, xm)),
    (jnp.asin, (xm,)),
    (jnp.asinh, (xm,)),
    (jnp.atan, (xm,)),
    (jnp.atan2, (xm, xm)),
    (jnp.atanh, (xm,)),
    # bitwise_and
    # bitwise_left_shift
    # bitwise_invert
    # bitwise_or
    # bitwise_right_shift
    # bitwise_xor
    (jnp.ceil, (xm,)),
    (jnp.conj, (xm,)),
    (jnp.cos, (xm,)),
    (jnp.cosh, (xm,)),
    (jnp.divide, (xm, xm)),
    (jnp.equal, (xm, xm)),
    (jnp.exp, (xm,)),
    (jnp.expm1, (xm,)),
    (jnp.floor, (xm,)),
    (jnp.floor_divide, (xm, xm)),
    (jnp.greater, (xm, xm)),
    (jnp.greater_equal, (xm, xm)),
    (jnp.imag, (xm,)),
    (jnp.isfinite, (xm,)),
    (jnp.isinf, (xm,)),
    (jnp.isnan, (xm,)),
    (jnp.less, (xm, xm)),
    (jnp.less_equal, (xm, xm)),
    (jnp.log, (xm,)),
    (jnp.log1p, (xm,)),
    (jnp.log2, (xm,)),
    (jnp.log10, (xm,)),
    (jnp.logaddexp, (xm, xm)),
    (jnp.logical_and, (xm, xm)),
    (jnp.logical_not, (xm,)),
    (jnp.logical_or, (xm, xm)),
    (jnp.logical_xor, (xm, xm)),
    (jnp.multiply, (xm, xm)),
    (jnp.negative, (xm,)),
    (jnp.not_equal, (xm, xm)),
    (jnp.positive, (xm,)),
    (jnp.power, (xm, 2.0)),
    (jnp.real, (xm,)),
    (jnp.remainder, (xm, xm)),
    (jnp.round, (xm,)),
    (jnp.sign, (xm,)),
    (jnp.sin, (xm,)),
    (jnp.sinh, (xm,)),
    (jnp.square, (xm,)),
    (jnp.sqrt, (xm,)),
    (jnp.subtract, (xm, xm)),
    (jnp.tan, (xm,)),
    (jnp.tanh, (xm,)),
    (jnp.trunc, (xm,)),
]


# =============================================================================


@pytest.mark.parametrize(
    ("func", "args"), **process_pytest_argvalues(process_func, funcs_and_args)
)
@pytest.mark.benchmark(group="quaxed", max_time=1.0, warmup=False)
def test_jit_compile(func, args):
    """Test the speed of jitting a function."""
    _ = func.lower(*args).compile()


@pytest.mark.parametrize(
    ("func", "args"), **process_pytest_argvalues(process_func, funcs_and_args)
)
@pytest.mark.benchmark(
    group="quaxed",
    max_time=1.0,  # NOTE: max_time is ignored
    warmup=True,
)
def test_execute(func, args):
    """Test the speed of calling the function."""
    _ = jax.block_until_ready(func(*args))
