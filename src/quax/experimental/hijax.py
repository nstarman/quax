"""Build a `quax.Value` on top of a `jax.experimental.hijax` type.

Quax and hijax answer opposite questions -- Quax runs *existing* code on your
type, hijax gives a *new* type its own operations -- and the way to get both is
to put a hijax value in the single leaf of a `quax.ArrayValue`. Quax stays on
the outside dispatching unmodified `jnp` code; hijax owns the inside, where the
type lives and where a cotangent can carry different metadata than its primal.
[Quax and hijax](../hijax.md) explains when that is worth doing.

This module removes the boilerplate of that arrangement.
[`HiValue`][quax.experimental.hijax.HiValue] is the wrapper, and
[`register_rules`][quax.experimental.hijax.register_rules] wires `lax`
primitives to hijax functions. It also re-exports the hijax names it resolves,
so that a rename upstream is absorbed here rather than in your code.

!!! warning

    Experimental, and doubly so: `jax.experimental.hijax` is itself experimental
    and has renamed things in most recent releases. This module tracks it.
"""

__all__ = (
    "HIJAX_FLOOR",
    "HiPrim",
    "HiType",
    "HiValue",
    "MappingSpec",
    "ShapedArray",
    "instantiate_zeros",
    "register_hitype",
    "register_rules",
)

import inspect
from collections.abc import Callable, Mapping
from typing import Any, Final

import jax
import jax._src.core as jax_core
import jax.extend.core as jexc
from jaxtyping import ArrayLike

from .._compat import hi_aval, JAX_VERSION
from .._dispatch import register
from .._values import ArrayValue


# ---------------------------------------------------------------------------
# Resolving the hijax API
# ---------------------------------------------------------------------------
#
# Hijax's surface has moved every few releases: `HiType` and `register_hitype`
# appeared in JAX 0.8.2, `HiPspec` in 0.9.2, `MappingSpec` became public in
# 0.11.0, and the primitive base class `VJPHiPrimitive` is renamed `HiPrim`
# after 0.11.1. Resolve the names by probing rather than pinning a version, so
# that a rename is a no-op here and downstream code never spells either name.

HIJAX_FLOOR: Final = "0.10.2"
"""The oldest JAX this module is tested against.

Hijax exists earlier, but with a different enough surface -- notably no public
`MappingSpec`, so no `vmap` -- that claiming support would be a guess.
"""

_TOO_OLD = (
    f"quax.experimental.hijax needs jax>={HIJAX_FLOOR} for "
    "`jax.experimental.hijax` (found jax=={version}). Quax itself supports a "
    "much older JAX; only this module, which interoperates with an "
    "experimental JAX API, needs a recent one."
)

try:
    from jax.experimental import hijax as _hijax
except ImportError as err:  # pragma: no cover -- needs an old JAX to reach
    raise ImportError(_TOO_OLD.format(version=JAX_VERSION)) from err


def _resolve(*names: str) -> Any:
    """The first of `names` this JAX exports from hijax, else `None`.

    Several names are tried per object so that a rename does not break us: the
    first is the current one, later ones are historical aliases. Each is looked
    for in `jax.experimental.hijax` and then in `jax._src.hijax`, because the
    public re-exports have lagged the implementation.
    """
    from jax._src import hijax as _hijax_src

    for module in (_hijax, _hijax_src):
        for name in names:
            obj = getattr(module, name, None)
            if obj is not None:
                return obj
    return None


HiPrim: Any = _resolve("HiPrim", "VJPHiPrimitive")
"""The hijax primitive base class, under whichever name this JAX uses."""

HiType: Any = _resolve("HiType")
"""The hijax abstract-value base class: the type of a hi value."""

MappingSpec: Any = _resolve("MappingSpec")
"""Base class for a hi type's `vmap` mapping spec."""

ShapedArray: Any = _resolve("ShapedArray")
"""`jax.core.ShapedArray`, as hijax re-exports it."""

instantiate_zeros: Any = _resolve("instantiate_zeros")
"""Turn a symbolic-zero tangent or cotangent into a dense one."""

register_hitype: Any = _resolve("register_hitype")
"""Associate a value class with the hi type that describes it."""

_MISSING = [
    name
    for name, obj in [
        ("HiPrim/VJPHiPrimitive", HiPrim),
        ("HiType", HiType),
        ("MappingSpec", MappingSpec),
        ("ShapedArray", ShapedArray),
        ("instantiate_zeros", instantiate_zeros),
        ("register_hitype", register_hitype),
    ]
    if obj is None
]
if _MISSING:  # pragma: no cover -- needs an old JAX to reach
    raise ImportError(
        f"{_TOO_OLD.format(version=JAX_VERSION)} Missing: {', '.join(_MISSING)}."
    )


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


class HiValue(ArrayValue):
    """A `quax.ArrayValue` whose single pytree leaf is a hijax value.

    Subclass this, then wire up primitives with
    [`register_rules`][quax.experimental.hijax.register_rules]. The subclass
    normally adds a friendlier `__init__` and some read-only properties, and
    nothing else:

    ```python
    class Unitful(HiValue):
        def __init__(self, array, units=()):
            self.leaf = array if isinstance(array, Quantity) else wrap(array, units)

        @property
        def units(self):
            return jax.typeof(self.leaf).units
    ```

    Two constraints this class exists to enforce, both of which are easy to get
    wrong by hand:

    - `aval()` reports a plain `ShapedArray` built *from* the leaf's hijax type,
      never the hijax type itself. `jax.Array`'s `isinstance` check reads a
      tracer's aval and every `jnp` function gates on it, so a `HiType` aval
      would make `jnp` reject your tracers -- defeating the point of using Quax.
    - The leaf must not be re-wrapped. A hijax tracer forwards `.shape` and
      `.dtype` from its type, so it passes `equinox.is_array_like`; an
      `__init__` that tests for array-likeness will wrap a traced value twice.
      Test for your hi value class instead.
    """

    leaf: Any
    """The hijax value, or -- under a trace -- a tracer of its hi type."""

    def __init__(self, leaf: Any, /) -> None:
        """**Arguments:**

        - `leaf`: the hijax value to wrap.
        """
        self.leaf = leaf

    def aval(self) -> jax_core.ShapedArray:
        """The shape and dtype of the leaf's hijax type, as a `ShapedArray`."""
        ty = jax.typeof(self.leaf)
        return jax_core.ShapedArray(ty.shape, ty.dtype)

    def materialise(self) -> Any:
        """Refuse: materialising would discard whatever the hi type carries.

        A primitive with no registered rule is therefore an error rather than a
        silent downgrade to a plain array. Override if your type would rather
        fall back than fail.
        """
        msg = (
            f"Refusing to materialise {type(self).__name__}, which would "
            "discard what its hijax type carries. Register a rule for this "
            "primitive, or override `materialise`."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Wiring primitives to hijax functions
# ---------------------------------------------------------------------------


def _arity(fn: Callable[..., Any], /) -> int:
    """How many positional arguments `fn` takes."""
    return sum(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        for p in inspect.signature(fn).parameters.values()
    )


def _forwarded_params(fn: Callable[..., Any], /) -> set[str] | None:
    """Which primitive params to pass to `fn`; `None` means all of them.

    A primitive passes every parameter it carries, and most hijax functions
    accept none of them -- `mul_p` alone passes `out_dtype`, which would be a
    `TypeError` for a two-argument `mul`. So forward only what the callable
    names as keyword-only, unless it takes `**kwargs` and can sort them out
    itself.
    """
    params = inspect.signature(fn).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params if p.kind is p.KEYWORD_ONLY}


def register_rules(
    cls: type[HiValue],
    rules: Mapping[jexc.Primitive, Callable[..., Any]],
    /,
) -> None:
    """Wire `lax` primitives to hijax functions for a `HiValue` subclass.

    Each `{primitive: function}` entry generates the Quax dispatch rules for
    that primitive: one per combination of `cls` and plain-array operands, since
    a binary primitive can be reached with your type on either side or both.
    Each generated rule unwraps the leaves, applies `function`, and wraps a
    hijax result back up in `cls`.

    The number of operands is read from `function`'s positional parameters, and
    the primitive's own parameters are forwarded to whichever keyword-only
    arguments `function` declares. Where the two do not line up -- because the
    hijax function takes them positionally, or names them differently -- adapt
    with a small lambda.

    !!! Example

        ```python
        register_rules(
            Unitful,
            {
                lax.mul_p: hi.mul,  # (x, y), no params
                lax.add_p: hi.add,
                lax.integer_pow_p: lambda q, *, y, **kw: hi.int_pow(q, y),
                lax.reduce_sum_p: lambda q, *, axes, **kw: hi.sum(q, axes),
            },
        )
        ```

    **Arguments:**

    - `cls`: the [`HiValue`][quax.experimental.hijax.HiValue] subclass to
        register against.
    - `rules`: a mapping from `jax.extend.core.Primitive` to the hijax function
        implementing it.

    **Returns:**

    Nothing; the rules are registered as a side effect, exactly as
    [`quax.register`][] does.

    **Raises:**

    `NotImplementedError` for a primitive with multiple results, which this
    helper cannot wrap. Write the rule by hand with `quax.register`.
    """
    for primitive, fn in rules.items():
        if primitive.multiple_results:
            msg = (
                f"`register_rules` cannot wrap {primitive}, which has multiple "
                "results. Register a rule for it directly with `quax.register`."
            )
            raise NotImplementedError(msg)
        arity = _arity(fn)
        # Every combination with at least one `cls` operand; a rule with none
        # would claim primitives that have nothing to do with this type.
        for mask in range(1, 2**arity):
            types = [cls if (mask >> i) & 1 else ArrayLike for i in range(arity)]
            _register_one(cls, primitive, fn, types)


def _register_one(
    cls: type[HiValue],
    primitive: jexc.Primitive,
    fn: Callable[..., Any],
    types: list[Any],
    /,
) -> None:
    """Register one generated rule, for one combination of operand types."""
    forwarded = _forwarded_params(fn)

    def rule(*args: Any, **params: Any) -> Any:
        leaves = [a.leaf if isinstance(a, cls) else a for a in args]
        if forwarded is not None:
            params = {k: v for k, v in params.items() if k in forwarded}
        out = fn(*leaves, **params)
        # A rule may legitimately return something that is not a hi value --
        # a comparison returns plain booleans -- so only re-wrap when the hijax
        # layer handed a hi value back.
        return cls(out) if hi_aval(out) is not None else out

    # Named after the primitive and the types it dispatches on, because plum's
    # ambiguity errors, tracebacks and profiles all show this name.
    operands = "_".join(
        "arraylike" if t is ArrayLike else t.__name__.lower() for t in types
    )
    rule.__name__ = rule.__qualname__ = f"{primitive.name}_{operands}"
    rule.__doc__ = (
        f"`{primitive.name}` for ({operands}), generated by "
        "`quax.experimental.hijax.register_rules`."
    )
    # plum dispatches on the signature, so give the generated function one.
    parameters = [
        inspect.Parameter(f"x{i}", inspect.Parameter.POSITIONAL_ONLY, annotation=t)
        for i, t in enumerate(types)
    ]
    parameters.append(inspect.Parameter("params", inspect.Parameter.VAR_KEYWORD))
    rule.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    rule.__annotations__ = {f"x{i}": t for i, t in enumerate(types)}
    register(primitive)(rule)
