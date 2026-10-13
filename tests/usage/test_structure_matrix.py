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


@pytest.mark.parametrize(
    ("dtype", "preferred"),
    [
        (jnp.float16, jnp.float32),
        (jnp.bfloat16, jnp.float32),
        (jnp.float16, None),
        (jnp.float32, None),
        (jnp.float32, jnp.float16),
        (jnp.float32, jnp.int32),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_preferred_element_type(dtype, preferred, batched, jit):
    batch = (2,) if batched else ()
    scale = (
        1.5
        if dtype == jnp.float32 and preferred is not None
        else (400.0 if preferred is not None else 1.0)
    )
    lower = jnp.full((*batch, 3), scale, dtype)
    main = jnp.full((*batch, 4), scale, dtype)
    upper = jnp.full((*batch, 3), scale, dtype)
    rhs = jnp.full(
        (*batch, 4),
        1.5
        if dtype == jnp.float32 and preferred is not None
        else (500.0 if preferred is not None else 2.0),
        dtype,
    )
    dn = (((2,), (1,)), ((0,), (0,))) if batched else (((1,), (0,)), ((), ()))

    def run(lower, main, upper, rhs):
        tri = structured_matrices.TridiagonalMatrix(lower, main, upper)
        return quax.quaxify(
            lambda A, x: jax.lax.dot_general(A, x, dn, preferred_element_type=preferred)
        )(tri, rhs)

    def reference(lower, main, upper, rhs):
        make_dense = lambda l, m, u: jnp.diag(m) + jnp.diag(l, -1) + jnp.diag(u, 1)
        dense = (
            jax.vmap(make_dense)(lower, main, upper)
            if batched
            else make_dense(lower, main, upper)
        )
        return jax.lax.dot_general(dense, rhs, dn, preferred_element_type=preferred)

    if jit:
        run = jax.jit(run)
    args = (lower, main, upper, rhs)
    actual, expected = run(*args), reference(*args)
    assert actual.dtype == expected.dtype
    assert jnp.array_equal(actual, expected)
    if not jnp.issubdtype(actual.dtype, jnp.inexact):
        return
    actual_grads = jax.grad(lambda *xs: run(*xs).sum(), argnums=(0, 1, 2, 3))(*args)
    expected_grads = jax.grad(lambda *xs: reference(*xs).sum(), argnums=(0, 1, 2, 3))(
        *args
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert jnp.array_equal(actual_grad, expected_grad)
