# Intervals

An array carrying a lower and an upper bound on each element. Operations map the
bounds forward, so the result brackets the result of the same computation on any
values inside the input bounds. Put the uncertainty on your inputs in, read the
width of the output.

## Example

```python
import quax
import quax.examples.interval as interval

# Existing program. It knows nothing about bounds.
def polynomial(x):
    return x**3 - 2.0 * x + 1.0

# A number known only to lie in [0.9, 1.1].
x = interval.Interval(0.9, 1.1)

out = quax.quaxify(polynomial)(x)
print(float(out.lo), float(out.hi))  # -0.471 0.531
print(float(out.width))              # 1.002
```

## The catch

Each operation treats its operands as independent, so a value used twice is
treated as two unrelated values. That makes the bounds wider than they need to
be — still correct, but pessimistic, and increasingly so along a computation:

```python
x = interval.Interval(-1.0, 2.0)

print(quax.quaxify(lambda a: a * a)(x).lo)   # -2.0, but x*x is never negative
print(quax.quaxify(lambda a: a**2)(x).lo)    # 0.0, the true bound
```

`mul` cannot see that both its operands are the same `x`; `integer_pow` sees one
operand and knows it is squaring it. This is the *dependency problem*, inherent
to interval arithmetic rather than a defect of this implementation. Affine
arithmetic is the standard remedy: it carries a linear form in shared unknowns
so that correlations cancel, which is a different representation rather than a
better rule set.

Bounds here are also approximate rather than certified, because JAX exposes no
control over floating-point rounding mode. Sound interval arithmetic rounds the
lower bound down and the upper bound up at every step; this cannot.
