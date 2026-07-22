# quax

An end user of a library built on Quax needs only one thing from this section: the [`quax.quaxify`][] function.

::: quax.quaxify

---

A developer of a library built on Quax (e.g. if you wanted to write your own libary analogous to `quax.examples.lora`) should additionally know about the following functionality.

!!! Info

    See also the [tutorials](../examples/custom_rules.ipynb) for creating your own array-ish Quax types.

::: quax.register

::: quax.Value
    options:
        members:
            - aval
            - default
            - materialise

::: quax.ArrayValue
    options:
        members:
            - aval
            - materialise
            - shape
            - dtype
            - ndim
            - size
