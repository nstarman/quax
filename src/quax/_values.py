import abc
from collections.abc import Callable, Sequence
from typing import Any, cast, TypeAlias, TypeGuard, TypeVar, Union

import equinox as eqx
import jax._src.core as core
import jax.extend.core as jexc
from jaxtyping import ArrayLike

from ._compat import typeof


T = TypeVar("T")
CT = TypeVar("CT", bound=Callable)
ValueLike: TypeAlias = Union[ArrayLike, "Value"]


class Value(eqx.Module):
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
        # Use list comprehension for better performance
        arrays = [
            x.materialise() if _is_value(x) else cast(ArrayLike, x) for x in values
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
    return isinstance(x, Value)


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


class _DenseArrayValue(ArrayValue):
    """Internal type used to wrap up a JAX arraylike into Quax's `Value` system.

    This is an implementation detail hidded from the user! It is unwrapped straight
    before calling a dispatch rule, and re-wrapped immediately afterwards.
    """

    array: ArrayLike

    def materialise(self) -> ArrayLike:
        return self.array

    def aval(self) -> core.ShapedArray:
        return typeof(self.array)


def _make_cache_finalizer(cache: dict, key: tuple) -> Callable[[Any], None]:
    """Return a weakref finalizer that removes *key* from *cache* when called."""

    def _fin(_ref: Any) -> None:
        # _ref is the now-dead weakref passed by the weakref machinery; unused
        # here because we only need the pre-captured key.
        cache.pop(key, None)  # no-op if already evicted

    return _fin  # caller passes this as the `callback` arg to weakref.ref()
