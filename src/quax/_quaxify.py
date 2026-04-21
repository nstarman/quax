import functools as ft
from collections.abc import Callable
from typing import Any, cast, Generic, Union

import equinox as eqx
import jax._src.core as core
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._trace import _QuaxTrace, _unwrap_tracer, _wrap_tracer
from ._values import _is_value, CT


class _Quaxify(eqx.Module, Generic[CT]):
    fn: CT
    filter_spec: PyTree[bool | Callable[[Any], bool]]
    dynamic: bool = eqx.field(static=True)

    @property
    def __wrapped__(self) -> CT:
        return self.fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        dynamic, static = eqx.partition(
            (self.fn, args, kwargs), self.filter_spec, is_leaf=_is_value
        )
        tag = core.TraceTag()
        with core.take_current_trace() as parent_trace:
            trace = _QuaxTrace(parent_trace, tag)
            # Cache partial functions to avoid repeated closure creation
            dynamic = jtu.tree_map(
                ft.partial(_wrap_tracer, trace), dynamic, is_leaf=_is_value
            )
            fn, args, kwargs = eqx.combine(dynamic, static)
            with core.set_current_trace(trace):
                out = fn(*args, **kwargs)
            out = jtu.tree_map(ft.partial(_unwrap_tracer, trace), out)
            return out

    def __get__(
        self, instance: object | None, owner: Any
    ) -> Union["_Quaxify[CT]", eqx.Partial["_Quaxify[CT]"]]:
        # Getting from a class
        if instance is None:
            return self
        # Getting from an instance
        return eqx.Partial(self, instance)


def quaxify(
    fn: CT,
    filter_spec: PyTree[bool | Callable[[Any], bool]] = True,
) -> CT:
    """'Quaxifies' a function, so that it understands custom array-ish objects like
    [`quax.examples.lora.LoraArray`][]. When this function is called, multiple dispatch
    will be performed against the types it is called with.

    **Arguments:**

    - `fn`: the function to wrap.
    - `filter_spec`: which arguments to quaxify. Advanced usage, see tip below.

    **Returns:**

    A copy of `fn`, that understands all Quax types.

    !!! Warning "Performance: pair with jax.jit"

        Calling `quaxify(fn)(*args)` **without** an outer `jax.jit` is 50–100× slower
        than the JIT path for small operations. Every call pays for Python-level trace
        setup, jaxpr interpretation, and equinox module overhead.

        Generally prefer:

        ```python
        jit_fn = jax.jit(quax.quaxify(fn))
        jit_fn(*args)  # fast warm-call path
        ```

    !!! Tip "Only quaxifying some arguments"

        Calling `quax.quaxify(fn, filter_spec)(*args, **kwargs)` will under-the-hood run
        `dynamic, static = eqx.partition((fn, args, kwargs), filter_spec)`, and then
        only quaxify those arguments in `dynamic`. This allows for passing through some
        [`quax.Value`][]s into the function unchanged, typically so that they can hit a
        nested `quax.quaxify`. See the
        [advanced tutorial](../examples/redispatch.ipynb).
    """
    return cast(CT, eqx.module_update_wrapper(_Quaxify(fn, filter_spec, dynamic=False)))
