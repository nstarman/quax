"""Multiple dispatch machinery for Quax."""

__all__ = ("register",)

from collections.abc import Callable, Sequence
from typing import Any, cast, TypeAlias

import equinox as eqx
import jax
import jax.extend.core as jexc
import plum
from jaxtyping import ArrayLike

from ._compat import fun_sourceinfo
from ._values import _dense, _is_value, CT, Value, ValueLike


# Keyed by primitive, or -- for a `jax.custom_jvp` -- by its source-info string.
_rules: dict[jexc.Primitive | str, plum.Function] = {}

# Cache resolved dispatch methods keyed by (primitive, arg-types).
# Sentinel for "no matching rule; fall through to _default_process".
_DISPATCH_MISS = object()
_dispatch_cache: dict[tuple, Any] = {}


def register(
    primitive: jexc.Primitive | jax.custom_jvp, /, *, precedence: int = 0
) -> Callable[[CT], CT]:
    """Registers a multiple dispatch implementation for this JAX primitive, or for
    a function decorated with `jax.custom_jvp`.

    !!! Example

        Used as decorator, and requires type annotations to perform multiple dispatch:
        ```python
        @quax.register(jax.lax.add_p)
        def add_somevalue_somevalue(x: SomeValue, y: SomeValue):
            return ...  # some implementation
        ```

    All positional arguments will be (subclasses of) [`quax.Value`][] -- these are the
    set of types that Quax will attempt to perform multiple dispatch with.

    All keyword arguments will be the parameters for this primitive, as passed to
    `prim.bind(... **params)`.

    **Arguments:**

    - `primitive`: The `jax.extend.core.Primitive` to provide a multiple
      dispatch implementation for. Alternatively a `jax.custom_jvp` function, whose
      body and derivative rule the registered rule then replaces (see below).

    - `precedence`: The precedence of this rule.
        See `plum.Dispatcher.dispatch` for details.

    **Returns:**

    A decorator for registering a multiple dispatch rule with the specified primitive.

    !!! Info "Registering a `jax.custom_jvp`"

        Normally Quax runs a `custom_jvp` function's body, dispatching each primitive
        in it. Registering the function instead replaces it, for the types the rule
        dispatches on, wherever it is called under [`quax.quaxify`][] -- eagerly or
        already staged inside a `jax.jit`:

        - The rule takes the function's inputs, flattened to leaves, positionally.
          Arguments in `nondiff_argnums` are not passed. No keyword arguments.
        - It returns a list or tuple of the function's outputs, flattened to leaves
          (`[out]` for a single array).
        - It replaces the derivative too: under an outer `jax.grad`, the rule itself
          is differentiated. Write it with JAX operations on plain arrays to keep
          JAX's own derivative rules.
        - A `jax.jvp` or `jax.grad` taken *inside* `quaxify` never reaches the rule:
          JAX differentiates the function before Quax sees it.
        - Rules are matched by the function's source location, so `custom_jvp`s built
          by one factory share a rule.
    """
    if isinstance(primitive, jexc.Primitive):
        key: jexc.Primitive | str = primitive
        name = str(primitive)
    elif isinstance(primitive, jax.custom_jvp):
        name = getattr(primitive, "__name__", repr(primitive))
        src = fun_sourceinfo(primitive.fun)
        if not src:  # pragma: no cover -- JAX falls back to the function's name
            raise ValueError(
                f"Cannot register {name}: JAX has no source location for it."
            )
        key = src
    else:
        raise TypeError(
            "`quax.register` takes a `jax.extend.core.Primitive` or a "
            f"`jax.custom_jvp` function, not {type(primitive).__name__!r}."
        )

    def _register(rule: CT, /) -> CT:
        try:
            existing_rule = _rules[key]
        except KeyError:

            def new_rule():
                raise NotImplementedError("Abstract primitive")  # pragma: no cover

            new_rule.__name__ = f"{name}_dispatcher"
            new_rule.__qualname__ = f"{name}_dispatcher"
            existing_rule = plum.Dispatcher().abstract(new_rule)

            _rules[key] = existing_rule
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
        if _is_value(x):  # MRO check; avoids ABCMeta.__instancecheck__
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
            # Operand is neither a quax Value nor array-like: a genuine dispatch
            # miss (e.g. a str/None passed to an operator). Raise a proper
            # TypeError so callers can catch it and defer -- e.g. return
            # NotImplemented from an operator. A bare `assert False` here
            # surfaced as AssertionError on older jax (which routes such operands
            # through this path) and, under `python -O`, vanished entirely,
            # letting the bad operand fall through to the default rule.
            raise TypeError(
                f"Primitive {primitive} got an operand of type "
                f"{type(x).__name__!r} that is neither a quax.Value nor "
                f"array-like; quax has no rule for it."
            )

    if default is None:
        default = value_default

    return default(primitive, values, params)


def _wrap_if_array(x: Any, /) -> Value:
    return _dense(cast(ArrayLike, x)) if eqx.is_array_like(x) else cast(Value, x)
