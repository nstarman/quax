"""Multiple dispatch machinery for Quax."""

__all__ = ("register",)

from collections.abc import Callable, Sequence
from typing import Any, cast, TypeAlias

import equinox as eqx
import jax
import jax.extend.core as jexc
import plum
from jaxtyping import ArrayLike

from ._dispatchable import dispatchable, DispatchableFunction
from ._values import _dense, _is_value, CT, Value, ValueLike


# Keyed by primitive, or by a `quax.dispatchable` function.
_rules: dict[jexc.Primitive | DispatchableFunction, plum.Function] = {}

# Cache resolved dispatch methods keyed by (primitive, arg-types).
# Sentinel for "no matching rule; fall through to _default_process".
_DISPATCH_MISS = object()
_dispatch_cache: dict[tuple, Any] = {}


def register(
    primitive: jexc.Primitive | DispatchableFunction | jax.custom_jvp,
    /,
    *,
    precedence: int = 0,
) -> Callable[[CT], CT]:
    """Registers a multiple dispatch implementation for this JAX primitive, or for
    a whole [`quax.dispatchable`][] function or `jax.custom_jvp` function.

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
      dispatch implementation for. Alternatively a [`quax.dispatchable`][]
      function: the rule then replaces the whole function for the types it
      dispatches on, and has the function's signature. See
      [`quax.dispatchable`][] for the contract. Or a `jax.custom_jvp` function:
      see below.

    - `precedence`: The precedence of this rule.
        See `plum.Dispatcher.dispatch` for details.

    **Returns:**

    A decorator for registering a multiple dispatch rule with the specified primitive.

    !!! Info "Registering a `jax.custom_jvp`"

        The first registration for a `jax.custom_jvp` replaces its body with a
        [`quax.dispatchable`][] one (`f.fun = quax.dispatchable(f.fun)`), and then
        registers the rule for that, so the rule takes the function's positional
        arguments -- including any `nondiff_argnums` ones -- and returns its
        output. Its own JVP rule is kept, and keeps applying under `jax.grad` on
        either side of [`quax.quaxify`][]: the rule replaces the function's value,
        not its declared derivative.

        This changes the function globally: every later trace of it, with or
        without Quax, contains a `quax_dispatch_call`. Values, derivatives and
        lowering are unchanged. It also calls `jax.clear_caches()`, so that no
        `jax.jit` cache keeps a trace of the old body. For a private JAX function
        (such as `jax._src.numpy.ufuncs._hypot`, which `jnp.hypot` is built on from
        JAX 0.12), look it up with `getattr(module, name, None)`, as these come and
        go between JAX versions.
    """
    if isinstance(primitive, jax.custom_jvp):
        if not isinstance(primitive.fun, DispatchableFunction):
            primitive.fun = dispatchable(primitive.fun)
            jax.clear_caches()
        primitive = primitive.fun

    if not isinstance(primitive, (jexc.Primitive, DispatchableFunction)):
        raise TypeError(
            "`quax.register` takes a `jax.extend.core.Primitive`, a "
            "`quax.dispatchable` function or a `jax.custom_jvp` function, not "
            f"{type(primitive).__name__!r}."
        )
    name = getattr(primitive, "__name__", str(primitive))

    def _register(rule: CT, /) -> CT:
        try:
            existing_rule = _rules[primitive]
        except KeyError:

            def new_rule():
                raise NotImplementedError("Abstract primitive")  # pragma: no cover

            new_rule.__name__ = f"{name}_dispatcher"
            new_rule.__qualname__ = f"{name}_dispatcher"
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
