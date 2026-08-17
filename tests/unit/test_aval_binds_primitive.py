"""`Value.aval` is user code, and may bind JAX primitives.

`quax.examples.lora.LoraArray.aval` does exactly that (via `lax.stop_gradient`), and
quax evaluates `aval()` while building tracers -- which happens with the current
trace taken. This module pins that down with a minimal `Value`.
"""

from typing import Any, cast

import jax
import jax.core
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

import quax
from quax._compat import typeof


class StopGradArray(quax.ArrayValue):
    """Like `lora.LoraArray`: computing its `aval` binds a primitive."""

    array: Array

    def materialise(self) -> Array:
        return self.array

    def aval(self) -> jax.core.ShapedArray:
        return cast(jax.core.ShapedArray, typeof(lax.stop_gradient(self.array)))


@quax.register(lax.mul_p)
def mul_stop_grad_array_stop_grad_array(
    x: StopGradArray, y: StopGradArray, /, **kw: Any
) -> StopGradArray:
    return StopGradArray(lax.mul_p.bind(x.array, y.array, **kw))


def test_wrapped_as_input():
    """The input wrapping in `_Quaxify.__call__` computes `aval()`."""
    x = StopGradArray(jnp.arange(3.0))
    out = quax.quaxify(lambda a: a + 1)(x)
    assert jnp.array_equal(out, jnp.arange(3.0) + 1)


def test_returned_from_rule():
    """Wrapping a rule's output back into a tracer computes `aval()` too."""
    x = StopGradArray(jnp.arange(3.0))
    out = quax.quaxify(lambda a: a * a)(x)
    assert isinstance(out, StopGradArray)
    assert jnp.array_equal(out.array, jnp.arange(3.0) ** 2)


def test_under_jit():
    """Same, with a non-trivial parent trace."""
    x = StopGradArray(jnp.arange(3.0))
    out = jax.jit(quax.quaxify(lambda a: a * a))(x)
    assert isinstance(out, StopGradArray)
    assert jnp.array_equal(out.array, jnp.arange(3.0) ** 2)
