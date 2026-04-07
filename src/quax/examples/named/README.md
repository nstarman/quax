# Named arrays

These are arrays with named dimensions. We can then use these to specify which dimensions we'd like to reduce down, or e.g. to check that we only perform matmuls down axes with matchign semantics.

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
