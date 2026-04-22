import functools as ft
from collections.abc import Callable
from typing import Any, cast, Generic, Union

import equinox as eqx
import jax._src.core as core
import jax.tree_util as jtu
from jaxtyping import PyTree

from ._trace import _QuaxTrace, _unwrap_tracer, _wrap_tracer
from ._values import _is_value, CT


def _partition_and_wrap(tree: Any, filter_spec: Any, trace: _QuaxTrace) -> Any:
    """Wrap the dynamic leaves of ``tree`` in ``_QuaxTracer`` for the given trace.

    Called once per ``_Quaxify.__call__`` invocation to prepare ``(fn, args,
    kwargs)`` for dispatch.  ``filter_spec`` mirrors the ``eqx.partition``
    contract:

    - ``True``  — all leaves are dynamic (the default for :func:`quaxify`).
    - A callable or nested bool pytree — only leaves selected by the spec are
      wrapped; the rest are left as plain Python/JAX objects so they pass through
      any nested :func:`quaxify` call unchanged (see the redispatch tutorial).

    The return value has the same pytree structure as ``tree`` with selected leaves
    replaced by ``_QuaxTracer`` instances.
    """
    if filter_spec is True:
        # Fast path: every leaf is dynamic, so partition+combine is the identity
        # transformation. Skip both calls and apply _wrap_tracer in a single
        # tree_map pass.
        return jtu.tree_map(ft.partial(_wrap_tracer, trace), tree, is_leaf=_is_value)

    # General path: split tree into dynamic (to be traced) and static (passed
    # through unchanged), wrap only the dynamic half, then recombine.
    dynamic, static = eqx.partition(tree, filter_spec, is_leaf=_is_value)
    dynamic = jtu.tree_map(ft.partial(_wrap_tracer, trace), dynamic, is_leaf=_is_value)
    return eqx.combine(dynamic, static, is_leaf=_is_value)


class _Quaxify(eqx.Module, Generic[CT]):
    fn: CT
    filter_spec: PyTree[bool | Callable[[Any], bool]]
    dynamic: bool = eqx.field(static=True)

    @property
    def __wrapped__(self) -> CT:
        return self.fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tag = core.TraceTag()
        with core.take_current_trace() as parent_trace:
            trace = _QuaxTrace(parent_trace, tag)
            fn, args, kwargs = _partition_and_wrap(
                (self.fn, args, kwargs), self.filter_spec, trace
            )
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
    /,
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
