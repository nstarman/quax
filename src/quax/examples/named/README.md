# Named arrays

These are arrays with named dimensions. We can then use these to specify which dimensions we'd like to reduce down, or e.g. to check that we only perform matmuls down axes with matching semantics.

## Example

```python
import equinox as eqx
import jax.random as jr
import quax
import quax.examples.named as named

# Existing program
linear = eqx.nn.Linear(3, 4, key=jr.PRNGKey(0))

# Wrap our desired inputs into NamedArrays
In = named.Axis(3)
Out = named.Axis(4)
named_bias = named.NamedArray(linear.bias, (Out,))
named_weight = named.NamedArray(linear.weight, (Out, In))
named_linear = eqx.tree_at(lambda l: (l.bias, l.weight), linear, (named_bias, named_weight))
vector = named.NamedArray(jr.normal(jr.PRNGKey(1), (3,)), (In,))

# Wrap function (here using matrix-vector multiplication) with quaxify. Output will be
# a NamedArray!
out = quax.quaxify(named_linear)(vector)
print(out)  # NamedArray(array=f32[4], axes=(Axis(size=4),))
```

## API

```python
named.NamedArray  # The star of the show.
named.Axis        # How each axis is named.
named.trace       # Trace down two named axes.
```

The usual JAX addition, subtraction, multiplication, and contraction (matrix-vector multiplication; matrix-matrix multiplication `jnp.tensordot` etc.) are also supported.

## Semantics

Names **validate**; they do not **align**. Operations run positionally, exactly as
the usual JAX operations do, and the names are checked to line up with that
positional pairing (and used to label the output):

- For an elementwise op, the operands must have the *same axes in the same
  order* — `(A, B) + (B, A)` is rejected, not silently transposed and added.
- For a contraction, each contracted (and batched) axis must share a name with
  the axis it is *positionally* paired against — `dot_general` contracts
  `lhs_contract[k]` with `rhs_contract[k]`, so those two axes must match.

This deliberately differs from a fully name-driven system (e.g. `xarray`), which
would *reorder* operands to align shared names. Here a mismatch is an error,
surfacing a likely bug rather than quietly reinterpreting the operation.

Axes are matched by **identity**: two axes are "the same name" only if they are
the same `Axis` object (so re-using an `Axis` instance is how you express that
two arrays share a dimension). Two separately constructed `Axis(3)`s are
distinct names.
