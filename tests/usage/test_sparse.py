import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import quax
import quax.examples.sparse as sparse


def _make_sparse_example(getkey):
    data = jr.normal(getkey(), (2, 4))
    indices0 = jnp.array([2, 0, 3])
    indices1 = jnp.array([0, 0, 3])
    indices2 = jnp.array([1, 0, 0])
    indices3 = jnp.array([2, 0, 1])
    indices_batch0 = jnp.stack([indices0, indices1, indices2, indices3])
    indices_batch1 = jnp.stack([indices1, indices0, indices3, indices3])
    indices = jnp.stack([indices_batch0, indices_batch1])
    shape = (2, 5, 1, 4)
    x = sparse.BCOO(data, indices, shape)
    return x


def _make_sparse_example2(getkey):
    data = jr.normal(getkey(), (3, 2, 4))
    indices0 = jnp.array([2, 0, 0])
    indices1 = jnp.array([0, 0, 0])
    indices2 = jnp.array([1, 0, 0])
    indices3 = jnp.array([4, 0, 0])
    indices_batch00 = jnp.stack([indices0, indices1, indices2, indices3])
    indices_batch01 = jnp.stack([indices1, indices0, indices3, indices3])
    indices_batch10 = jnp.stack([indices3, indices1, indices2, indices3])
    indices_batch11 = jnp.stack([indices1, indices3, indices3, indices3])
    indices_batch20 = jnp.stack([indices1, indices1, indices0, indices3])
    indices_batch21 = jnp.stack([indices0, indices0, indices3, indices3])
    indices_batch0 = jnp.stack([indices_batch00, indices_batch01])
    indices_batch1 = jnp.stack([indices_batch10, indices_batch11])
    indices_batch2 = jnp.stack([indices_batch20, indices_batch21])
    indices = jnp.stack([indices_batch0, indices_batch1, indices_batch2])
    # TODO: change to (3, 2, 5, 1, 1) to test broadcasting
    shape = (3, 2, 5, 1, 4)
    x = sparse.BCOO(data, indices, shape)
    return x


def test_add_dense(getkey):
    x = _make_sparse_example(getkey)
    x_mat = x.enable_materialise().materialise()
    y0 = 1
    y1 = jr.normal(getkey(), (1, 4))
    # TODO: change to (3, 1, 5, 1, 4) to test broadcasting
    y2 = jr.normal(getkey(), (3, 2, 5, 1, 4))
    add = quax.quaxify(lambda a, b: a + b)
    out0 = add(x, y0)
    out1 = add(x, y1)
    out2 = add(x, y2)

    @jax.vmap
    def _add0(i, d):
        assert i.shape == (4, 3)
        assert d.shape == (4,)
        idx = tuple(jnp.moveaxis(i, -1, 0))
        return jnp.ones((5, 1, 4)).at[idx].add(d)

    true_out0 = _add0(x.indices, x.data)
    assert jnp.array_equal(out0, true_out0)
    assert jnp.allclose(out0, x_mat + y0)

    @jax.vmap
    def _add1(i, d):
        assert i.shape == (4, 3)
        assert d.shape == (4,)
        idx = tuple(jnp.moveaxis(i, -1, 0))
        base = jnp.broadcast_to(y1, (5, 1, 4))
        return base.at[idx].add(d)

    true_out1 = _add1(x.indices, x.data)
    assert jnp.array_equal(out1, true_out1)
    assert jnp.allclose(out1, x_mat + y1)

    assert jnp.allclose(out2, x_mat + y2)


def test_add_sparse(getkey):
    x = _make_sparse_example(getkey)
    y = _make_sparse_example2(getkey)
    add = quax.quaxify(lambda a, b: a + b)
    z = add(x, y)
    w = add(x, x)
    x_mat = x.enable_materialise().materialise()
    y_mat = y.enable_materialise().materialise()
    z_mat = z.enable_materialise().materialise()
    w_mat = w.enable_materialise().materialise()
    assert z.shape == (3, 2, 5, 1, 4)
    assert z_mat.shape == (3, 2, 5, 1, 4)
    assert w.shape == (2, 5, 1, 4)
    assert w_mat.shape == (2, 5, 1, 4)
    assert jnp.allclose(w_mat, x_mat * 2)
    assert jnp.allclose(z_mat, x_mat + y_mat)


def test_mul(getkey):
    data = jr.normal(getkey(), (2, 4))
    indices0 = jnp.array([2, 0, 3])
    indices1 = jnp.array([0, 0, 3])
    indices2 = jnp.array([1, 0, 0])
    indices3 = jnp.array([2, 0, 1])
    indices_batch0 = jnp.stack([indices0, indices1, indices2, indices3])
    indices_batch1 = jnp.stack([indices1, indices0, indices3, indices3])
    indices = jnp.stack([indices_batch0, indices_batch1])
    shape = (2, 5, 1, 4)
    x = sparse.BCOO(data, indices, shape)
    y0 = 3
    y1 = jr.normal(getkey(), (5, 1, 4))
    y2 = jr.normal(getkey(), (2, 5, 1, 4))
    mul = quax.quaxify(lambda a, b: a * b)
    out0 = mul(x, y0)
    out1 = mul(x, y1)
    out2 = mul(x, y2)

    true_out0 = sparse.BCOO(data * 3, indices, shape)
    assert eqx.tree_equal(out0, true_out0)

    tuple_indices = tuple(jnp.moveaxis(indices, -1, 0))
    y1_at_indices = jax.vmap(lambda a: y1[a])(tuple_indices)
    true_out1 = sparse.BCOO(data * y1_at_indices, indices, shape)
    assert eqx.tree_equal(out1, true_out1)

    y2_at_indices = jax.vmap(lambda a, b: a[b])(y2, tuple_indices)
    true_out2 = sparse.BCOO(data * y2_at_indices, indices, shape)
    assert eqx.tree_equal(out2, true_out2)
