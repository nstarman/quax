import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

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
    # Scale atol by |y| @ |v| (the float32 rounding bound): a row that nearly
    # cancels differs by ~1 ulp of its operands between the two paths.
    atol = 1e-5 * jnp.max(jnp.abs(y) @ jnp.abs(v))
    assert jnp.allclose(out, out2, atol=atol)


def test_materialise():
    lower_diag = jnp.arange(3)
    main_diag = jnp.arange(4)
    upper_diag = jnp.arange(3) + 5
    x = structured_matrices.TridiagonalMatrix(
        lower_diag, main_diag, upper_diag, allow_materialise=True
    )
    y = jnp.array([[0, 5, 0, 0], [0, 1, 6, 0], [0, 1, 2, 7], [0, 0, 2, 3]])
    assert jnp.array_equal(x.materialise(), y)


def _tridiag(key, batch, n):
    k1, k2, k3 = jr.split(key, 3)
    lower = jr.normal(k1, (*batch, n - 1))
    main = jr.normal(k2, (*batch, n))
    upper = jr.normal(k3, (*batch, n - 1))
    return lower, main, upper


@pytest.mark.parametrize(
    ("batch", "rhs_shape", "dn"),
    [
        # (contract, batch) dimension numbers; tridiag matrix is (*batch, n, n).
        ((), (4,), (((1,), (0,)), ((), ()))),  # pure matvec
        ((), (4, 3), (((1,), (0,)), ((), ()))),  # matvec + rhs free
        ((2,), (2, 4), (((2,), (1,)), ((0,), (0,)))),  # 1 batch
        ((2, 3), (2, 4), (((3,), (1,)), ((0,), (0,)))),  # batch + lhs free
        ((2, 3), (2, 4, 5), (((3,), (1,)), ((0,), (0,)))),  # batch + lhs + rhs free
        ((2, 3), (2, 3, 4), (((3,), (2,)), ((0, 1), (0, 1)))),  # 2 batch
        ((2, 3, 4), (2, 3, 4, 4), (((4,), (3,)), ((0, 1, 2), (0, 1, 2)))),  # 3 batch
    ],
)
def test_dot_general_batched_axis_order(batch, rhs_shape, dn):
    """`dot_general` output must be in `(batch, lhs_free, rhs_free)` order.

    Regression: the nested-vmap fast path emitted axes in the order the vmaps
    happened to produce them (lhs-free and batch groups leading in reverse),
    so any dot_general mixing a batch dim with a non-batch free dim returned a
    silently axis-permuted result.
    """
    n = 4
    lower, main, upper = _tridiag(jr.PRNGKey(0), batch, n)
    rhs = jr.normal(jr.PRNGKey(1), rhs_shape)

    tri = structured_matrices.TridiagonalMatrix(lower, main, upper)
    out = quax.quaxify(lambda A, x: jax.lax.dot_general(A, x, dn))(tri, rhs)

    dense = structured_matrices.TridiagonalMatrix(
        lower, main, upper, allow_materialise=True
    ).materialise()
    ref = jax.lax.dot_general(dense, rhs, dn)

    assert out.shape == ref.shape
    atol = 1e-5 * jnp.max(jax.lax.dot_general(jnp.abs(dense), jnp.abs(rhs), dn))
    assert jnp.allclose(out, ref, atol=atol)


@pytest.mark.parametrize("transform", ["eager", "jit", "grad"])
@pytest.mark.parametrize(
    ("batch", "rhs_shape", "dn"),
    [
        ((2, 3), (3, 4), (((3,), (1,)), ((1,), (0,)))),
        ((2, 3), (4, 3, 5), (((3,), (0,)), ((1,), (1,)))),
        ((2, 3), (3, 2, 4), (((3,), (2,)), ((1, 0), (0, 1)))),
        ((2, 3), (3, 2, 4), (((3,), (2,)), ((0, 1), (1, 0)))),
        ((2, 3, 5), (7, 5, 4, 2), (((4,), (2,)), ((2, 0), (1, 3)))),
        ((2, 3, 5), (7, 2, 4, 5), (((4,), (2,)), ((2, 0), (3, 1)))),
        ((2, 3), (3, 5, 4), (((3,), (2,)), ((), ()))),
        ((2, 3), (2, 3, 4), (((3,), (2,)), ((0, 1), (0, 1)))),
    ],
)
def test_dot_general_interleaved_batch_axes(batch, rhs_shape, dn, transform):
    """Mapped axes retain their meaning after enclosing vmaps remove other axes."""
    lower, main, upper = _tridiag(jr.PRNGKey(12), batch, 4)
    rhs = jr.normal(jr.PRNGKey(13), rhs_shape)

    def structured(lower, main, upper, rhs):
        tri = structured_matrices.TridiagonalMatrix(lower, main, upper)
        return quax.quaxify(lambda a, b: jax.lax.dot_general(a, b, dn))(tri, rhs)

    def dense(lower, main, upper, rhs):
        matrix = structured_matrices.TridiagonalMatrix(
            lower, main, upper, allow_materialise=True
        ).materialise()
        return jax.lax.dot_general(matrix, rhs, dn)

    if transform == "jit":
        structured = jax.jit(structured)
        dense = jax.jit(dense)
    elif transform == "grad":
        structured_fn, dense_fn = structured, dense
        structured = jax.grad(
            lambda *args: jnp.sum(structured_fn(*args) ** 2), argnums=(0, 1, 2, 3)
        )
        dense = jax.grad(
            lambda *args: jnp.sum(dense_fn(*args) ** 2), argnums=(0, 1, 2, 3)
        )

    actual = structured(lower, main, upper, rhs)
    expected = dense(lower, main, upper, rhs)
    for result, reference in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        assert jnp.allclose(result, reference, rtol=2e-5, atol=2e-5)
