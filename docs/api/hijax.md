# quax.examples.hijax

An array with physical units whose cotangent carries the *inverse* units. Built
on [`quax.experimental.hijax`](experimental.md); see
[Quax and hijax](../hijax.md) for when to reach for it, and
[`quax.examples.unitful`](unitful.md) for the same physics as a plain Quax type.

| | |
|---|---|
| Minimum JAX | 0.10.2 ([`quax.experimental.hijax.HIJAX_FLOOR`](experimental.md)) |
| Import failure | `ImportError` naming the missing hijax objects |
| Dimensions | shared with `quax.examples.unitful`: import `meters`, `kilograms`, `seconds`, `Dimension` from there |

## Supported operations

A rule is registered for each `lax` primitive below. Any other primitive falls
through to [`quax.Value.materialise`][], which raises, so an unsupported
operation is an error rather than a silent loss of units.

| Primitive | Operands | Units of the result |
|---|---|---|
| `mul_p` | two `Unitful`s, or a `Unitful` and an array | product of the operands' |
| `add_p` | two `Unitful`s | the operands', which must be equal |
| `integer_pow_p` | one `Unitful` | the operand's, exponents multiplied |
| `reduce_sum_p` | one `Unitful` | the operand's |
| `broadcast_in_dim_p` | one `Unitful` | the operand's |

`jnp.sum`, `**`, `*` and `+` therefore work; `-`, `/`, `<`, `jnp.where`,
`jnp.copy` and `jnp.mean` raise.

## Supported transformations

| Transformation | Notes |
|---|---|
| `jax.jit` | Supported. On JAX 0.10.2 and 0.11.0 only, trips `JAX_CHECK_TRACER_LEAKS` with a false report; see Limitations |
| `jax.grad`, `jax.vjp` | Cotangent units are the inverse of the primal's |
| `jax.jvp` | Tangent units are the primal's |
| `jax.vmap` | Requires [`MAPPED`][quax.examples.hijax.MAPPED] as the `in_axes`/`out_axes` entry, and `axis_size` when no plain-array argument is mapped |
| `jax.lax.scan` | Works with a `Unitful` carry; no spec needed |
| `jax.shard_map` | Not supported: `QuantityTy` implements no `shard`, `unshard` or `nospec` |

## Limitations

- **Broadcasting.** `mul` and `add` broadcast a scalar operand against an array
  one. Any other broadcast raises `TypeError`; use
  [`broadcast_in_dim`][quax.examples.hijax.broadcast_in_dim] explicitly.
- **Transposing a broadcast.** `broadcast_in_dim` has no transpose rule for a
  broadcast that stretches a size-1 axis, so differentiating through one raises
  `NotImplementedError`.
- **`JAX_CHECK_TRACER_LEAKS` on JAX 0.10.2 and 0.11.0.** On those two releases,
  consuming a hijax value under `jit` reports a leaked tracer that is not one.
  It is a JAX regression rather than anything to fix here, reproducible with one
  `HiPrim` and no Quax: 0.10.1 and earlier are unaffected, and 0.11.1 fixed it.
  Quax's test suite disables the check on the affected versions only, so the
  report is noticed rather than tolerated if it returns.
- **Dimension identity.** As in `quax.examples.unitful`, two `Dimension`
  instances with the same name are different dimensions.

---

## The Quax layer

Built on [`quax.experimental.hijax`](experimental.md), which supplies the
wrapper base class and generates the dispatch rules.

`Unitful` is a [`quax.experimental.hijax.HiValue`][], so it inherits `aval`,
`materialise`, and the `leaf` field holding the `Quantity`.

::: quax.examples.hijax.Unitful
    options:
        members:
            - __init__
            - units
            - array

::: quax.examples.hijax.MAPPED

## The hijax layer

Values of these types are produced and consumed only by the functions below.
See [the how-to](../how-to/transform-metadata-under-autodiff.md#avoid-two-traps)
for the two rules that govern using them in traced code.

::: quax.examples.hijax.Quantity

::: quax.examples.hijax.QuantityTy

::: quax.examples.hijax.QuantitySpec

`Units` is the canonical representation of physical units used throughout this
module: a tuple of `(Dimension, exponent)` pairs, sorted by dimension name, with
zero exponents dropped. It is a tuple rather than the `dict` that
`quax.examples.unitful` uses because a `HiType` must be hashable.
[`to_units`][quax.examples.hijax.to_units] builds one from a `Dimension` or a
`dict`.

### Applying the primitives

::: quax.examples.hijax.wrap

::: quax.examples.hijax.unwrap

::: quax.examples.hijax.add

::: quax.examples.hijax.mul

::: quax.examples.hijax.int_pow

::: quax.examples.hijax.sum

::: quax.examples.hijax.broadcast_in_dim

::: quax.examples.hijax.to_units
