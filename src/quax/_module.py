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
class-creation time, everything the fast path needs, then reproduces the *essential*
parts of construction — the initialisation guard, field converters, `__check_init__`,
and the "field not initialised" check — while skipping only the non-fatal warnings and
the expensive `dir(self)` scan (replaced by a cheaper `hasattr` check).

Should a future `equinox` release rename or remove the internals this relies on, the
fast path disables itself and every construction falls through to `equinox`'s own,
correct `__call__` — costing the speedup but never correctness.

Known limitations (all arise from precomputing metadata at class-creation time, and
are narrow enough not to matter in practice):

- Converters and `__check_init__` hooks are captured when the class is created.
  Monkeypatching a *new* `__check_init__` (or converter) onto the class afterwards is
  not observed by the fast path, whereas equinox — which re-walks the MRO on every
  construction — would pick it up.
- The internal `type(x).aval(x)` / `type(x).materialise(x)` calls (in `_values` /
  `_trace`) assume these are ordinary instance methods. A subclass that implements
  them as a `@classmethod` would misbind; such an implementation is already
  semantically wrong (they depend on instance data) and fails loudly.
"""

__all__ = ()

import dataclasses
import warnings
import weakref
from typing import Any, NamedTuple

import equinox as eqx


_EqxModuleMeta = type(eqx.Module)

# Internals the fast path depends on. Kept behind a guarded import so that a change
# in `equinox` degrades to the slow-but-correct path rather than breaking at import.
# Degradation is deliberately *not silent*: we warn here, and a CI canary test
# (`test_fast_path_available`) asserts the fast path stays live on supported
# `equinox` versions so a breaking upgrade turns CI red rather than quietly halving
# throughput.
#
# The fallbacks are typed callables (not `None`) so the fast path stays type-clean;
# they are never reached, because `_FASTPATH_AVAILABLE is False` keeps every class out
# of `_fast_specs` (below), so construction goes through `super().__call__`.
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
        RuntimeWarning,
        stacklevel=2,
    )


class _FastSpec(NamedTuple):
    """Precomputed data the fast path needs to construct a class.

    ``field_names`` are all dataclass field names, used to detect a field left
    unset by a buggy ``__init__``; ``converters`` are ``(field_name, converter)``
    pairs applied after ``__init__``; ``checks`` are the class's ``__check_init__``
    hooks in equinox's MRO order.
    """

    field_names: tuple[str, ...]
    converters: tuple[tuple[str, Any], ...]
    checks: tuple[Any, ...]


# Per-class fast-path metadata, keyed off the class rather than stored as attributes
# on it (the same weak-collection idiom equinox uses for `_has_dataclass_init` /
# `_currently_initialising`). A class is present iff it qualifies for the fast path;
# absence means "fall back to equinox's construction", so no separate flag is needed.
# Weak keys let entries for dynamically-created classes be collected with the class.
_fast_specs: "weakref.WeakKeyDictionary[type, _FastSpec]" = weakref.WeakKeyDictionary()


class _FastModuleMeta(_EqxModuleMeta):
    """Metaclass that skips `equinox.Module`'s per-instance validation where safe.

    Applied to [`quax.Value`][], so every Quax value type inherits the fast
    construction path. See the module docstring for the rationale and the exact set
    of checks that are (and are not) preserved. Per-class metadata lives in the
    module-level `_fast_specs` registry rather than as attributes on the class.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if _FASTPATH_AVAILABLE:
            # `cls` is an equinox.Module, which equinox has just turned into a frozen
            # dataclass in `super().__new__`, so `dataclasses.fields` is always valid.
            fields = dataclasses.fields(cls)  # type: ignore[arg-type]
            # Converters must still be applied post-init (equinox does this
            # regardless of whether __init__ is dataclass-generated or custom).
            converters = tuple(
                (f.name, c)
                for f in fields
                if (c := f.metadata.get("converter")) is not None
            )
            # __check_init__ hooks, outermost-class-first, exactly as equinox walks
            # them. These enforce user invariants and must always run.
            checks = tuple(
                k.__dict__["__check_init__"]
                for k in cls.__mro__
                if "__check_init__" in k.__dict__
            )
            field_names = tuple(f.name for f in fields)
            _fast_specs[cls] = _FastSpec(field_names, converters, checks)
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # A class the fast path could not analyse is absent from `_fast_specs`;
        # abstract instantiation is still deferred to equinox's own __call__ (which
        # raises the right errors and runs the full validation).
        spec = _fast_specs.get(cls)
        if spec is None or is_abstract_module(cls):
            return super().__call__(*args, **kwargs)

        # `self` is a freshly-allocated instance of a dynamically-determined class;
        # `Any` is the honest type and avoids metaclass-`Self` typing friction on the
        # `_currently_initialising` / `object.__setattr__` calls below.
        #
        # Forward the constructor arguments to `__new__`, matching `type.__call__`
        # (and equinox's slow path): a `Value` may override `__new__` to consume
        # them. For the common case where `__new__` is `object.__new__`, the extras
        # are ignored because every Module overrides `__init__`.
        self: Any = cls.__new__(cls, *args, **kwargs)  # pyright: ignore[reportArgumentType]
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

        # Reject a field left unset by a buggy __init__. equinox raises the same
        # error via a `dir(self)` scan; without it the instance would flatten to a
        # divergent pytree treedef and fail confusingly far from the cause. Checking
        # the precomputed field names with `hasattr` is much cheaper than `dir`.
        missing = {name for name in spec.field_names if not hasattr(self, name)}
        if missing:
            raise TypeError(
                f"The following fields were not initialised during __init__: {missing}"
            )

        # Converters first, then __check_init__ — the same order equinox uses.
        for fname, converter in spec.converters:
            object.__setattr__(
                self, fname, converter(object.__getattribute__(self, fname))
            )
        for check in spec.checks:
            check(self)
        return self
