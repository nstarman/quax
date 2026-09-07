# quax.experimental

Experimental APIs, which may change or be removed without notice.

## quax.experimental.hijax

Plumbing for building a `quax.Value` around a
[hijax](https://docs.jax.dev/en/latest/301/hijax-types.html) type.
[Quax and hijax](../hijax.md#which-should-you-use) covers when to reach for it,
[the how-to](../how-to/transform-metadata-under-autodiff.md) covers how, and
[`quax.examples.hijax`](hijax.md) is a worked type built on this module.

Doubly experimental: `jax.experimental.hijax` is itself experimental and has
renamed things in most recent releases. This module resolves those names by
probing, so a rename upstream is absorbed here rather than in your code.

| | |
|---|---|
| Minimum JAX | 0.10.2 (`quax.experimental.hijax.HIJAX_FLOOR`) |
| Import failure | `ImportError` naming the missing hijax objects |
| Re-exported hijax names | `HiPrim`, `HiType`, `MappingSpec`, `ShapedArray`, `instantiate_zeros`, `register_hitype` |

`HiPrim` is `VJPHiPrimitive` up to JAX 0.11.1 and `HiPrim` after it; importing
it from here means never spelling either name.

---

::: quax.experimental.hijax.HiValue
    options:
        members:
            - __init__
            - aval
            - materialise

::: quax.experimental.hijax.register_rules
