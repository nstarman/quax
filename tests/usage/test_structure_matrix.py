import jax.numpy as jnp
import jax.random as jr

import quax
import quax.examples.structured_matrices as structured_matrices


def test_matmul(getkey):
    lower_diag = jnp.arange(3)
    main_diag = jnp.arange(4)
    upper_diag = jnp.arange(3) + 5
    x = structured_matrices.TridiagonalMatrix(lower_diag, main_diag, upper_diag)
    y = jnp.array([[0, 5, 0, 0], [0, 1, 6, 0], [0, 1, 2, 7], [0, 0, 2, 3]])
    v = jr.normal(getkey(), (4,))
    matmul = quax.quaxify(lambda a, b: a @ b)
    out = matmul(x, v)
    out2 = matmul(y, v)
    assert jnp.allclose(out, out2)


def test_materialise():
    lower_diag = jnp.arange(3)
    main_diag = jnp.arange(4)
    upper_diag = jnp.arange(3) + 5
    x = structured_matrices.TridiagonalMatrix(
        lower_diag, main_diag, upper_diag, allow_materialise=True
    )
    y = jnp.array([[0, 5, 0, 0], [0, 1, 6, 0], [0, 1, 2, 7], [0, 0, 2, 3]])
    assert jnp.array_equal(x.materialise(), y)
