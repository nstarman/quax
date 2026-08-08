import abc
from collections.abc import Callable, Sequence
from typing import Any, cast, final, TypeAlias, TypeGuard, TypeVar, Union

import equinox as eqx
import jax._src.core as core
import jax.extend.core as jexc
from jax.typing import ArrayLike

from ._compat import typeof
from ._module import _FastModuleMeta


T = TypeVar("T")
CT = TypeVar("CT", bound=Callable)
ValueLike: TypeAlias = Union[ArrayLike, "Value"]


class Value(eqx.Module, metaclass=_FastModuleMeta):
    """Represents an object which Quax can perform multiple dispatch with.

    In practice you will almost always want to inherit from [`quax.ArrayValue`][]
    instead, which represents specifically an array-ish object that can be used for
    multiple dispatch.
    """

    @abc.abstractmethod
    def aval(self) -> core.AbstractValue:
        """All concrete subclasses must implement this method, specifying the abstract
        value seen by JAX.

        **This method must be pure**: it must return the same `AbstractValue` every
        time it is called on the same instance. In practice this means that any
        Python metadata that influences the abstract value should be static (for
        example via `eqx.field(static=True)`), whilst shape or dtype may also be
        derived from dynamic array fields. Quax caches the result of `aval()` at
        tracer-construction time and will not observe later changes.

        **Arguments:**

        Nothing.

        **Returns:**

        Any subclass of `jax.core.AbstractValue`. Typically a `jax.core.ShapedArray`.
        """

    @staticmethod
    def default(
        primitive: jexc.Primitive, values: Sequence[ValueLike], params: dict[str, Any]
    ) -> "ValueLike | Sequence[ValueLike]":
        """This is the default rule for when no rule has been [`quax.register`][]'d for
        a primitive.

        When performing multiple dispatch `primitive.bind(value1, value2, value3)`,
        then:

        1. If there is a dispatch rule matching the types of `value1`, `value2`, and
            `value3`, then that will be used.
        2. If precisely one of the types of `value{1,2,3}` overloads this method, then
            that default rule will be used.
        3. If precisely zero of the types of `value{1,2,3}` overloads this method, then
            all values are [`quax.Value.materialise`][]d, and the usual JAX
            implementation is called.
        4. If multiple of the types of `value{1,2,3}` overload this method, then a
            trace-time error will be raised.

        **Arguments:**

        - `primitive`: the `jax.extend.core.Primitive` being considered.
        - `values`: a sequence of what values this primitive is being called with. Each
            value can either be [`quax.Value`][]s, or a normal JAX arraylike (i.e.
            `bool`/`int`/`float`/`complex`/NumPy scalar/NumPy array/JAX array).
        - `params`: the keyword parameters to the primitive.

        **Returns:**

        The result of binding this primitive against these types. If
        `primitive.multiple_results is False` then this should be a single `quax.Value`
        or JAX arraylike. If `primitive.multiple_results is True`, then this should be
        a tuple/list of such values.

        !!! Example

            The default implementation discussed above performs the following:
            ```python
            @staticmethod
            def default(primitive, values, params):
                arrays = [
                    x if equinox.is_array_like(x) else x.materialise() for x in values
                ]
                return primitive.bind(*arrays, **params)
            ```
            (Using the [Equinox](https://github.com/patrick-kidger/equinox) library that
            underlies much of the JAX ecosystem.)
        """
        # Call materialise unbound (via the type) rather than `x.materialise()`:
        # for user Value types the latter goes through equinox's
        # Module.__getattribute__, which wraps the method in a fresh BoundMethod
        # (a Module allocation) so jax.jit(x.method) works — pure overhead on
        # this fallback path, which runs once per non-dense primitive input.
        arrays = [
            type(x).materialise(x) if _is_value(x) else cast(ArrayLike, x)
            for x in values
        ]
        return primitive.bind(*arrays, **params)

    @abc.abstractmethod
    def materialise(self) -> Any:
        """All concrete subclasses must implement this method, specifying how to
        materialise this object into a JAX type (i.e. almost always a JAX array, unless
        you're doing something obscure using tokens or refs).

        !!! Example

            For example, a LoRA array consists of three arrays `(W, A, B)`, combined as
            `W + AB`. [`quax.examples.lora.LoraArray`] leaves these as three separate
            arrays for efficiency, but calling `lora_array.materialise()` will evaluate
            `W + AB` and return a normal JAX array.

        This is so that the usual JAX primitive implementations can be applied as a
        fallback: the array-ish object is materialised, and then the usual JAX
        implementation called on it. (See [`quax.Value.default`][].)

        !!! Info

            It is acceptable for this function to just raise an error -- in this case
            the error will be surfaced to the end user, indicating that an operation is
            not supported for this array-ish object.

        **Arguments:**

        Nothing.

        **Returns:**

        A JAX type; typically a JAX array.
        """


def _is_value(x: object) -> TypeGuard[Value]:
    # `Value` is an ABC, so `isinstance(x, Value)` routes through the (comparatively
    # slow) `ABCMeta.__instancecheck__`. Every Quax value type is a *real* subclass of
    # `Value` (Quax never uses ABC virtual registration), so a plain MRO-membership
    # test is equivalent and ~3x faster — and this predicate is on the hot leaf path
    # (it's the `is_leaf` for the wrap/unwrap tree_maps run on every quaxify call).
    return Value in type(x).__mro__


class ArrayValue(Value):
    """A subclass [`quax.Value`][] for specifically array-like types. If you are
    creating a custom array-ish object then you should typically inherit from this.

    Provides the properties `.shape`, `.dtype`, `.ndim`, `.size`, each as a shortcut for
    `self.aval().shape` etc.
    """

    @abc.abstractmethod
    def materialise(self) -> ArrayLike:
        pass

    @abc.abstractmethod
    def aval(self) -> core.ShapedArray:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.aval().shape

    @property
    def dtype(self) -> Any:  # jax.numpy.dtype or ExtendedDType
        return self.aval().dtype

    @property
    def ndim(self) -> int:
        return self.aval().ndim

    @property
    def size(self) -> int:
        return self.aval().size


@final
class _DenseArrayValue(ArrayValue):
    """Internal type used to wrap up a JAX arraylike into Quax's `Value` system.

    Hot paths test membership with ``type(x) is _DenseArrayValue`` (cheaper than an
    ABC ``isinstance``), which is exact only because this type is never subclassed.
    `@final` documents that for type checkers; `__init_subclass__` enforces it at
    runtime so a stray subclass can't silently fall off the dense fast paths.

    This is an implementation detail hidden from the user! It is unwrapped straight
    before calling a dispatch rule, and re-wrapped immediately afterwards.
    """

    array: ArrayLike

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError(
            "_DenseArrayValue is internal and final; the trace hot paths rely on "
            "`type(x) is _DenseArrayValue`, so it must not be subclassed."
        )

    def __init__(self, array: ArrayLike, /) -> None:
        # Bypass equinox's Module.__setattr__, which on every field assignment
        # checks whether the module is frozen and performs pytree-leaf
        # bookkeeping via dataclasses.fields(). _DenseArrayValue is constructed
        # on every primitive output in the O1 fast path and for every input
        # passed to to_value(), so this is on the hottest call site in the
        # trace. The bypass is safe: this class is internal-only, has exactly
        # one field that is set once at construction and never mutated, and
        # is never exposed to user code or equinox transforms.
        object.__setattr__(self, "array", array)

    def materialise(self) -> ArrayLike:
        return self.array

    def aval(self) -> core.ShapedArray:
        return typeof(self.array)

    def __getattribute__(self, name: str, /) -> Any:
        # Bypass equinox's Module.__getattribute__, which on every non-magic
        # attribute access wraps the result in a BoundMethod(func, self) — a
        # fresh eqx.Module allocation — so that bound methods are themselves
        # valid pytrees (needed for jax.jit(module.method)). That allocation
        # is pure overhead here: _DenseArrayValue methods are called directly
        # by _QuaxTracer and _QuaxTrace internals, never passed to jax.jit.
        # Benchmarks show .aval() drops from ~38 µs to ~1.6 µs and
        # .materialise() from ~35 µs to ~0.3 µs with this override.
        return object.__getattribute__(self, name)


# Bind the allocation primitives once at import; these are the entirety of
# _dense()'s body, so avoiding the repeated global/attribute lookups matters on
# the hottest allocation site in the trace.
_object_new = object.__new__
_object_setattr = object.__setattr__


def _dense(array: ArrayLike, /) -> _DenseArrayValue:
    """Construct a `_DenseArrayValue` while bypassing `equinox.Module` entirely.

    `_DenseArrayValue` is allocated on *every* primitive input and output during a
    quaxified trace, making it the single hottest allocation site. Going through
    `_ModuleMeta.__call__` costs ~13 µs/call — almost all of it per-instance
    validation (`dir(self)`, `dataclasses.fields`, converter/static/`init` scans,
    the `__check_init__` MRO walk) that is meaningless for this type. This factory
    drops that to ~0.2 µs (a ~57× speedup) by allocating directly.

    The result is a genuine `_DenseArrayValue`: the class is already registered as
    a pytree at class-creation time, so flatten/unflatten and every JAX/equinox
    transform behave identically. The bypass is safe because this type is
    internal-only, has exactly one field set once and never mutated, and is never
    exposed to user code or handed to `jax.jit`.
    """
    obj = _object_new(_DenseArrayValue)
    _object_setattr(obj, "array", array)
    return obj


def _make_cache_finalizer(cache: dict, key: tuple) -> Callable[[Any], None]:
    """Return a weakref finalizer that removes *key* from *cache* when called."""

    def _fin(_ref: Any) -> None:
        # _ref is the now-dead weakref passed by the weakref machinery; unused
        # here because we only need the pre-captured key.
        cache.pop(key, None)  # no-op if already evicted

    return _fin  # caller passes this as the `callback` arg to weakref.ref()
