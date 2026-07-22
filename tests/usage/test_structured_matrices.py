import jax
import jax.numpy as jnp
import pytest

import quax
import quax.examples.structured_matrices as sm


def _tridiag(key, batch, n):
    k1, k2, k3 = jax.random.split(key, 3)
    low = jax.random.normal(k1, (*batch, n - 1))
    main = jax.random.normal(k2, (*batch, n))
    up = jax.random.normal(k3, (*batch, n - 1))
    return low, main, up


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
    low, main, up = _tridiag(jax.random.PRNGKey(0), batch, n)
    rhs = jax.random.normal(jax.random.PRNGKey(1), rhs_shape)

    tri = sm.TridiagonalMatrix(low, main, up)
    out = quax.quaxify(lambda A, x: jax.lax.dot_general(A, x, dn))(tri, rhs)

    dense = sm.TridiagonalMatrix(low, main, up, allow_materialise=True).materialise()
    ref = jax.lax.dot_general(dense, rhs, dn)

    assert out.shape == ref.shape
    assert jnp.allclose(out, ref)
