# quax.examples.lora

As a (actually quite useful) tech-demo, Quax provides an implementation of [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685), which is a popular fine-tuning method for large neural network models.

Most of the time you will just need the [`quax.examples.lora.loraify`][] function, which transforms an existing [Equinox](https://github.com/patrick-kidger/equinox) model.

For a user who only wants to LoRA'ify only part of their model, the underlying [`quax.examples.lora.LoraArray`][] array-ish object (which subclasses [`quax.ArrayValue`][]) is also available.

---

::: quax.examples.lora.loraify

::: quax.examples.lora.LoraArray
    options:
        members:
            - __init__

::: quax.examples.lora.trainable_filter

## Example

Here's a copy of the LoRA example from the README again:

--8<-- ".lora-example.md"
