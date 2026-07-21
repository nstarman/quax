"""A faster construction path for Quax's `equinox.Module` subclasses.

`equinox`'s `Module` is optimised for models that are built a handful of times, not
for hot-loop allocation. Quax, however, wraps every primitive input and output in a
[`quax.Value`][] during a trace, so `Module` construction is on the hottest path.

`equinox._module._module._ModuleMeta.__call__` runs a batch of per-instance safety
checks on *every* construction:

- `dir(self)` plus a missing-field set-difference,
- a per-field loop scanning for converters, static-array warnings, and
  `init=False` warnings,
- an MRO walk collecting `__check_init__` methods,
- (for dataclass-`__init__` classes) a `jax.tree_util.tree_leaves` scan via
  `_warn_jax_transformed_function`.

For quax these are almost pure overhead. [`_FastModuleMeta`][] precomputes, once at
class-creation time, everything the fast path needs, then reproduces only the
*essential* parts of construction — the initialisation guard, field converters, and
`__check_init__` — while skipping the validation and warnings. The real correctness
hooks (`__check_init__` and converters) are always preserved; only non-fatal warnings
and the "field not initialised" `TypeError` (which fires only on an already-buggy
`__init__`) are dropped.

Should a future `equinox` release rename or remove the internals this relies on, the
fast path disables itself and every construction falls through to `equinox`'s own,
correct `__call__` — costing the speedup but never correctness.
"""

__all__ = ("FastPathUnavailableWarning",)

import dataclasses
import warnings
from typing import Any

import equinox as eqx


class FastPathUnavailableWarning(UserWarning):
    """Warned once, at import, when Quax's fast `Module` construction is disabled.

    The fast path relies on a small set of `equinox` internals. If they are missing
    (typically because the installed `equinox` is newer than Quax has been updated
    for), Quax stays fully correct but falls back to `equinox`'s slower per-instance
    construction, which noticeably slows `Value`-heavy tracing. This warning exists
    so that regression is never *silent*; filter it with
    `warnings.filterwarnings("ignore", category=quax.FastPathUnavailableWarning)`.
    """


_EqxModuleMeta = type(eqx.Module)

# Internals the fast path depends on. Kept behind a guarded import so that a change
# in `equinox` degrades to the slow-but-correct path rather than breaking at import.
# Degradation is deliberately *not silent*: we warn here, and a CI canary test
# (`test_fast_path_available`) asserts the fast path stays live on supported
# `equinox` versions so a breaking upgrade turns CI red rather than quietly halving
# throughput.
#
# The fallbacks are typed callables (not `None`) so the fast path stays type-clean;
# they are never reached, because `_FASTPATH_AVAILABLE is False` forces every class's
# `__quax_fast__` flag off and construction goes through `super().__call__`.
try:
    from equinox._module._module import _currently_initialising, is_abstract_module

    _FASTPATH_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - only hit on an incompatible equinox

    class _UnavailableInitGuard:
        def add(self, obj: Any, /) -> None: ...
        def remove(self, obj: Any, /) -> None: ...

    _currently_initialising = _UnavailableInitGuard()

    def is_abstract_module(cls: Any, /) -> bool:
        return True

    _FASTPATH_AVAILABLE = False
    warnings.warn(
        "Quax's fast equinox.Module construction is disabled: could not access the "
        f"equinox internals it relies on (equinox {eqx.__version__}: {_exc!r}). "
        "Quax remains correct but Value-heavy tracing will be slower. This usually "
        "means the installed equinox is newer than this version of Quax supports.",
        FastPathUnavailableWarning,
        stacklevel=2,
    )


class _FastModuleMeta(_EqxModuleMeta):
    """Metaclass that skips `equinox.Module`'s per-instance validation where safe.

    Applied to [`quax.Value`][], so every Quax value type inherits the fast
    construction path. See the module docstring for the rationale and the exact set
    of checks that are (and are not) preserved.

    The per-class ``__quax_fast__`` / ``__quax_converters__`` / ``__quax_checks__``
    metadata is precomputed in ``__new__`` and read in ``__call__``; it is set
    dynamically (annotating it on the metaclass would make Equinox's
    ``dataclass_transform`` synthesise an ``__init__``), hence the read-site ignores.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        converters: tuple[tuple[str, Any], ...] = ()
        checks: tuple[Any, ...] = ()
        fast = False
        if _FASTPATH_AVAILABLE:
            try:
                fields = dataclasses.fields(cls)  # type: ignore[arg-type]
                # Converters must still be applied post-init (equinox does this
                # regardless of whether __init__ is dataclass-generated or custom).
                converters = tuple(
                    (f.name, c)
                    for f in fields
                    if (c := f.metadata.get("converter")) is not None
                )
                # __check_init__ hooks, outermost-class-first, exactly as equinox
                # walks them. These enforce user invariants and must always run.
                checks = tuple(
                    k.__dict__["__check_init__"]
                    for k in cls.__mro__
                    if "__check_init__" in k.__dict__
                )
                fast = True
            except Exception:  # pragma: no cover - defensive
                fast = False
        # Dunder names are never treated as dataclass fields (fields come from
        # annotations), so these are safe to stash on the class.
        cls.__quax_fast__ = fast
        cls.__quax_converters__ = converters
        cls.__quax_checks__ = checks
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Abstract instantiation, and any class the fast path could not analyse,
        # go through equinox's own __call__ (which raises the right errors and
        # runs the full validation).
        if not cls.__quax_fast__ or is_abstract_module(cls):  # pyright: ignore[reportAttributeAccessIssue]
            return super().__call__(*args, **kwargs)

        # `self` is a freshly-allocated instance of a dynamically-determined class;
        # `Any` is the honest type and avoids metaclass-`Self` typing friction on the
        # `_currently_initialising` / `object.__setattr__` calls below.
        self: Any = cls.__new__(cls)  # pyright: ignore[reportArgumentType]
        # Register with equinox's init guard so that a *custom* __init__'s
        # `self.x = ...` assignments are permitted on the frozen dataclass (and so
        # equinox's own __setattr__ warnings still fire for those assignments). A
        # dataclass-generated __init__ uses object.__setattr__ and does not need
        # this, but registering unconditionally is cheap and uniform.
        _currently_initialising.add(self)
        try:
            cls.__init__(self, *args, **kwargs)  # pyright: ignore[reportCallIssue]
        finally:
            _currently_initialising.remove(self)

        # Converters first, then __check_init__ — the same order equinox uses.
        for fname, converter in cls.__quax_converters__:  # pyright: ignore[reportAttributeAccessIssue]
            object.__setattr__(
                self, fname, converter(object.__getattribute__(self, fname))
            )
        for check in cls.__quax_checks__:  # pyright: ignore[reportAttributeAccessIssue]
            check(self)
        return self
