# quax.examples.unitful

A worked example of a `quax.Value` that carries metadata and refuses to give it up:
an array tagged with physical units. It is the type most of these docs use for
demonstrations, because unit errors are easy to recognise as errors.

```python
import jax.numpy as jnp
import quax
from quax.examples.unitful import meters, seconds, Unitful

distance = Unitful(jnp.asarray(100.0), meters)
time = Unitful(jnp.asarray(9.58), seconds)
speed = quax.quaxify(lambda d, t: d * t**-1)(distance, time)
print(speed.units)  # {m: 1, s: -1}
```

Rules are registered for `add`, `mul`, `integer_pow`, `lt`, `broadcast_in_dim`,
`copy` and `select_n` -- enough to differentiate through, and to run a `lax` loop
or a `diffrax` solve. Anything else raises rather than dropping the units.

---

::: quax.examples.unitful.Unitful
    options:
        members:
            - __init__

::: quax.examples.unitful.Dimension
