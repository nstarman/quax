"""Multiple dispatch machinery for Quax."""

__all__ = ("register",)

from collections.abc import Callable, Sequence
from typing import Any, cast, TypeAlias

import equinox as eqx
import jax.extend.core as jexc
import plum
from jaxtyping import ArrayLike

from ._values import _dense, CT, Value, ValueLike


_rules: dict[jexc.Primitive, plum.Function] = {}

# Cache resolved dispatch methods keyed by (primitive, arg-types).
# Sentinel for "no matching rule; fall through to _default_process".
_DISPATCH_MISS = object()
_dispatch_cache: dict[tuple, Any] = {}


def register(
    primitive: jexc.Primitive, /, *, precedence: int = 0
) -> Callable[[CT], CT]:
    """Registers a multiple dispatch implementation for this JAX primitive.

    !!! Example

        Used as decorator, and requires type annotations to perform multiple dispatch:
        ```python
        @quax.register(jax.lax.add_p)
        def _(x: SomeValue, y: SomeValue):
            return ...  # some implementation
        ```

    All positional arguments will be (subclasses of) [`quax.Value`][] -- these are the
    set of types that Quax will attempt to perform multiple dispatch with.

    All keyword arguments will be the parameters for this primitive, as passed to
    `prim.bind(... **params)`.

    **Arguments:**

    - `primitive`: The `jax.extend.core.Primitive` to provide a multiple
      dispatch implementation for.

    - `precedence`: The precedence of this rule.
        See `plum.Dispatcher.dispatch` for details.

    **Returns:**

    A decorator for registering a multiple dispatch rule with the specified primitive.
    """

    def _register(rule: CT, /) -> CT:
        try:
            existing_rule = _rules[primitive]
        except KeyError:

            def new_rule():
                raise NotImplementedError("Abstract primitive")  # pragma: no cover

            new_rule.__name__ = f"{primitive}_dispatcher"
            new_rule.__qualname__ = f"{primitive}_dispatcher"
            existing_rule = plum.Dispatcher().abstract(new_rule)

            _rules[primitive] = existing_rule
        existing_rule.dispatch(rule, precedence=precedence)
        # Invalidate all cached dispatch decisions so newly registered rules
        # are picked up on the next call.  Registration is rare (import-time)
        # so clearing the whole cache is cheaper than scanning for matching
        # keys and keeps the hot-path lookup at a single dict.get.
        _dispatch_cache.clear()
        return rule

    return _register


DefaultCallable: TypeAlias = Callable[
    [jexc.Primitive, Sequence[ValueLike], dict[str, Any]],
    "ValueLike | Sequence[ValueLike]",
]


def _default_process(
    primitive: jexc.Primitive, values: Sequence[ValueLike], params: dict[str, Any]
) -> "ValueLike | Sequence[ValueLike]":
    # Fast path: find first non-default, then check if all match
    default: DefaultCallable | None = None
    value_default = Value.default  # Cache attribute lookup
    for x in values:
        if isinstance(x, Value):
            x_default = type(x).default
            if x_default is value_default:
                continue
            if default is None:
                default = x_default
            elif default is not x_default:
                # Multiple different defaults - slow path
                types = {type(v) for v in values if isinstance(v, Value)}
                raise TypeError(
                    f"Multiple array-ish types {types} are specifying default "
                    f"process rules."
                )
        elif not eqx.is_array_like(x):
            assert False

    if default is None:
        default = value_default

    return default(primitive, values, params)


def _wrap_if_array(x: Any, /) -> Value:
    return _dense(cast(ArrayLike, x)) if eqx.is_array_like(x) else cast(Value, x)
