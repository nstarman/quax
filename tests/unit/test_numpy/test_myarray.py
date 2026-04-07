"""Test with JAX inputs."""

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jtu
import pytest

import quax

from ..myarray import is_myarray, MyArray, unwrap
from ..test_lax.test_myarray import _unwrap_myarray


xfail_quax58 = pytest.mark.xfail(
    reason="https://github.com/patrick-kidger/quax/issues/58"
)
mark_todo = pytest.mark.skip("TODO")
mark_nomd = pytest.mark.xfail(reason="Can't be supported with MD on primitives")

x = MyArray(jnp.array([[1, 2], [3, 4]], dtype=float))
y = MyArray(jnp.array([[5, 6], [7, 8]], dtype=float))
xtrig = MyArray(jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float))
xbool = MyArray(jnp.array([True, False, True], dtype=bool))


@pytest.mark.parametrize(
    ("func_name", "args", "kw", "expect_myarray"),
    [
        ("abs", (x,), {}, True),
        ("absolute", (x,), {}, True),
        ("acos", (xtrig,), {}, True),
        ("acosh", (x,), {}, True),
        ("add", (x, y), {}, True),
        ("all", (xbool,), {}, False),
        *[
            ("allclose", (x, x), {}, False),
            ("allclose", (x, y), {}, False),
        ],
        ("amax", (x,), {}, True),
        ("amin", (x,), {}, True),
        ("angle", (x,), {}, True),
        ("any", (xbool,), {}, True),
        ("append", (x, 4), {}, True),
        pytest.param(
            "apply_along_axis", (lambda x: x + 1, 0, x), {}, True, marks=mark_todo
        ),
        pytest.param(
            "apply_over_axes",
            (lambda x, _: x + 1, x, [0, 1]),
            {},
            True,
            marks=mark_todo,
        ),
        pytest.param("arange", (MyArray(0), MyArray(3)), {}, True, marks=xfail_quax58),
        ("arccos", (xtrig,), {}, True),
        ("arccosh", (x,), {}, True),
        ("arcsin", (xtrig,), {}, True),
        ("arcsinh", (xtrig,), {}, True),
        ("arctan", (xtrig,), {}, True),
        ("arctan2", (x, y), {}, True),
        ("arctanh", (xtrig,), {}, True),
        ("argmax", (x,), {}, True),
        ("argmin", (x,), {}, True),
        ("argpartition", (x, 1), {}, True),
        ("argsort", (x,), {}, True),
        *(
            pytest.param("argwhere", (x,), {}, True, marks=xfail_quax58),
            ("argwhere", (x,), {"size": x.size}, True),
        ),
        ("around", (x,), {}, True),
        ("array", (x,), {}, True),
        ("array_equal", (x, y), {}, False),
        ("array_equiv", (x, y), {}, False),
        pytest.param("array_str", (x,), {}, True, marks=mark_todo),
        ("asarray", (x,), {}, True),
        ("asin", (xtrig,), {}, True),
        ("asinh", (xtrig,), {}, True),
        ("astype", (x, int), {}, True),
        ("atan", (xtrig,), {}, True),
        ("atan2", (x, y), {}, True),
        ("atanh", (xtrig,), {}, True),
        ("atleast_1d", (x,), {}, True),
        ("atleast_2d", (x,), {}, True),
        ("atleast_3d", (x,), {}, True),
        ("average", (x,), {}, True),
        *(
            pytest.param(
                "bincount",
                (MyArray(jnp.asarray([0, 1, 1, 2, 2, 2])),),
                {},
                True,
                marks=xfail_quax58,
            ),
            (
                "bincount",
                (MyArray(jnp.asarray([0, 1, 1, 2, 2, 2])),),
                {"length": 3},
                True,
            ),
        ),
        ("bitwise_and", (xbool, xbool), {}, True),
        ("bitwise_count", (xbool,), {}, True),
        ("bitwise_invert", (xbool,), {}, True),
        ("bitwise_left_shift", (xbool, 1), {}, True),
        ("bitwise_not", (xbool,), {}, True),
        ("bitwise_or", (xbool, xbool), {}, True),
        ("bitwise_right_shift", (xbool, 1), {}, True),
        ("bitwise_xor", (xbool, xbool), {}, True),
        ("block", ([x, x],), {}, True),
        ("broadcast_arrays", (x, y), {}, True),
        ("broadcast_to", (x, (2, 2)), {}, True),
        ("cbrt", (x,), {}, True),
        ("ceil", (x,), {}, True),
        pytest.param("choose", (0, [x, x]), {}, True, marks=xfail_quax58),
        ("clip", (x, 1, 2), {}, True),
        ("column_stack", ([x, x],), {}, True),
        pytest.param("complex128", (x,), {}, True, marks=pytest.mark.xfail),
        ("complex64", (x,), {}, True),
        pytest.param("complex_", (x,), {}, True, marks=pytest.mark.xfail),
        pytest.param("complexfloating", (1,), {}, True, marks=pytest.mark.xfail),
        pytest.param("compress", (xbool, x), {}, True, marks=xfail_quax58),
        ("concat", ([x, x],), {}, True),
        ("concatenate", ([x, x],), {}, True),
        ("conj", (x,), {}, True),
        ("conjugate", (x,), {}, True),
        ("convolve", (x[:, 0], y[:, 0]), {}, True),
        ("copy", (x,), {}, True),
        ("copysign", (x, y), {}, True),
        ("corrcoef", (x,), {}, True),
        ("correlate", (x[:, 0], y[:, 0]), {}, True),
        ("cos", (x,), {}, True),
        ("cosh", (x,), {}, True),
        ("count_nonzero", (x,), {}, True),
        ("cov", (x,), {}, True),
        ("cross", (x, y), {}, True),
        ("csingle", (x,), {}, True),
        ("cumprod", (x,), {}, True),
        ("cumsum", (x,), {}, True),
        ("deg2rad", (x,), {}, True),
        ("degrees", (x,), {}, True),
        ("delete", (x, 1), {}, True),
        ("diag", (x,), {}, True),
        ("diag_indices_from", (x,), {}, False),
        ("diagflat", (x,), {}, True),
        ("diagonal", (x,), {}, True),
        ("diff", (x,), {}, True),
        ("digitize", (x, x[:, 0]), {}, True),
        ("divide", (x, y), {}, True),
        ("divmod", (x, 2), {}, True),
        ("dot", (x, y), {}, True),
        pytest.param("double", (x,), {}, True, marks=pytest.mark.xfail),
        ("dsplit", (MyArray(jnp.arange(16.0).reshape(2, 2, 4)), 2), {}, True),
        ("dstack", (x,), {}, True),
        ("ediff1d", (x,), {}, True),
        ("einsum", ("ij,jk->ik", x, y), {}, True),
        # ("einsum_path", ("ij,jk,kl->il", rand1, rand2, rand3), {"optimize": "greedy"}),  # TODO: replace independent test with this  # noqa: E501
        pytest.param("empty_like", (x,), {}, True, marks=mark_nomd),
        ("equal", (x, y), {}, True),
        ("exp", (x,), {}, True),
        ("exp2", (x,), {}, True),
        ("expand_dims", (x, 0), {}, True),
        ("expm1", (x,), {}, True),
        pytest.param("extract", (jnp.array([True]), x), {}, True, marks=xfail_quax58),
        ("fabs", (x,), {}, True),
        ("fill_diagonal", (x, 2), {"inplace": False}, True),
        ("fix", (x,), {}, True),
        *(
            pytest.param("flatnonzero", (x,), {}, True, marks=xfail_quax58),
            ("flatnonzero", (x,), {"size": x.size}, True),
        ),
        ("flip", (x,), {}, True),
        ("fliplr", (x,), {}, True),
        ("flipud", (x,), {}, True),
        ("float8_e4m3b11fnuz", (x,), {}, True),
        ("float8_e4m3fn", (x,), {}, True),
        ("float8_e4m3fnuz", (x,), {}, True),
        ("float8_e5m2", (x,), {}, True),
        ("float8_e5m2fnuz", (x,), {}, True),
        pytest.param(
            "float_",
            (x,),
            {},
            True,
            marks=pytest.mark.skipif(hasattr(jnp, "float_"), reason="not available"),
        ),
        ("float_power", (x, y), {}, True),
        ("floor", (x,), {}, True),
        ("floor_divide", (x, y), {}, True),
        ("fmax", (x, y), {}, True),
        ("fmin", (x, y), {}, True),
        ("fmod", (x, y), {}, True),
        ("frexp", (x,), {}, True),
        pytest.param("from_dlpack", (x,), {}, True, marks=mark_todo),
        pytest.param(
            "frombuffer",
            (b"\x01\x02",),
            {"dtype": jnp.uint8},
            True,
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            "fromfunction",
            ((lambda i, _: i), (2, 2)),
            {"dtype": float},
            True,
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            "fromstring",
            ("1 2",),
            {"dtype": int, "sep": " "},
            True,
            marks=pytest.mark.xfail,
        ),
        pytest.param("full", ((2, 2), 4.0), {}, True, marks=mark_nomd),
        pytest.param("full_like", (x, 4.0), {}, True, marks=mark_nomd),
        ("gcd", (x[:, 0].astype(int), y[:, 0].astype(int)), {}, True),
        ("geomspace", (MyArray(1.0), MyArray(100.0)), {}, True),
        ("gradient", (x,), {}, True),
        ("greater", (x, y), {}, True),
        ("greater_equal", (x, y), {}, True),
        pytest.param("hamming", (2,), {}, True, marks=pytest.mark.xfail),
        pytest.param("hanning", (2,), {}, True, marks=pytest.mark.xfail),
        ("heaviside", (x, y), {}, True),
        ("histogram", (x,), {}, True),
        ("histogram2d", (x[:, 0], x[:, 1]), {}, True),
        ("histogram_bin_edges", (x,), {"bins": 3}, True),
        pytest.param("histogramdd", (x,), {"bins": 2}, (True, False), marks=mark_todo),
        ("hsplit", (x, 2), {}, True),
        ("hstack", ([x, x],), {}, True),
        ("hypot", (x, y), {}, True),
        ("i0", (x,), {}, True),
        pytest.param("identity", (4,), {}, True, marks=pytest.mark.xfail),
        ("imag", (x,), {}, False),  # TODO: why not MyArray?
        ("inner", (x, y), {}, True),
        ("insert", (x, 1, 2.0), {}, True),
        pytest.param(
            "interp", (x[:, 0], x[:, 0], 2 * x[:, 0]), {}, True, marks=mark_todo
        ),
        *(
            pytest.param(
                "intersect1d", (x[:, 0], x[:, 0]), {}, True, marks=xfail_quax58
            ),
            ("intersect1d", (x[:, 0], x[:, 0]), {"size": x[:, 0].size}, True),
        ),
        ("invert", (x.astype(int),), {}, True),
        ("isclose", (x, y), {}, True),
        ("iscomplex", (x,), {}, False),
        ("iscomplexobj", (x,), {}, False),
        ("isdtype", (x, "real floating"), {}, False),
        ("isfinite", (x,), {}, True),
        (
            "isin",
            (
                2 * MyArray(jnp.arange(4).reshape((2, 2))),
                MyArray(jnp.asarray([1, 2, 4, 8])),
            ),
            {},
            True,
        ),
        ("isinf", (x,), {}, True),
        ("isnan", (x,), {}, True),
        ("isneginf", (x,), {}, True),
        ("isposinf", (x,), {}, True),
        ("isreal", (x,), {}, False),
        ("isrealobj", (x,), {}, False),
        ("isscalar", (x,), {}, False),
        ("iterable", (x,), {}, False),
        ("ix_", (x[:, 0], y[:, 0]), {}, True),
        ("kaiser", (4, MyArray(jnp.asarray(3.0))), {}, True),
        ("kron", (x, x), {}, True),
        ("lcm", (x.astype(int), y.astype(int)), {}, True),
        ("ldexp", (x, y.astype(int)), {}, True),
        ("left_shift", (x.astype(int), y.astype(int)), {}, True),
        ("less", (x, y), {}, True),
        ("less_equal", (x, y), {}, True),
        ("lexsort", (x,), {}, True),
        ("linspace", (MyArray(2.0), MyArray(3.0), 5), {}, True),
        ("log", (x,), {}, True),
        ("log10", (x,), {}, True),
        ("log1p", (x,), {}, True),
        ("log2", (x,), {}, True),
        ("logaddexp", (x, y), {}, True),
        ("logaddexp2", (x, y), {}, True),
        ("logical_and", (x, y), {}, True),
        ("logical_not", (x,), {}, True),
        ("logical_or", (x, y), {}, True),
        ("logical_xor", (x, y), {}, True),
        ("logspace", (MyArray(0.0), MyArray(10.0)), {}, True),
        ("matmul", (x, y), {}, True),
        ("matrix_transpose", (x,), {}, True),
        ("max", (x,), {}, True),
        ("maximum", (x, y), {}, True),
        ("mean", (x,), {}, True),
        ("median", (x,), {}, True),
        ("meshgrid", (x[:, 0], y[:, 0]), {}, True),
        ("min", (x,), {}, True),
        ("minimum", (x, y), {}, True),
        ("mod", (x, 2), {}, True),
        ("modf", (x,), {}, True),
        ("moveaxis", (x[None], 0, 1), {}, True),
        ("multiply", (x, y), {}, True),
        ("nan_to_num", (x,), {}, True),
        ("nanargmax", (x,), {}, True),
        ("nanargmin", (x,), {}, True),
        ("nancumprod", (x,), {}, True),
        ("nancumsum", (x,), {}, True),
        ("nanmax", (x,), {}, True),
        ("nanmean", (x,), {}, True),
        ("nanmedian", (x,), {}, True),
        ("nanmin", (x,), {}, True),
        ("nanpercentile", (x, 50), {}, True),
        ("nanprod", (x,), {}, True),
        ("nanquantile", (x, 0.5), {}, True),
        ("nanstd", (x,), {}, True),
        ("nansum", (x,), {}, True),
        ("nanvar", (x,), {}, True),
        ("ndim", (x,), {}, False),
        ("negative", (x,), {}, True),
        ("nextafter", (x, y), {}, True),
        *(
            pytest.param("nonzero", (x,), {}, True, marks=xfail_quax58),
            ("nonzero", (x,), {"size": x.size}, True),
        ),
        ("not_equal", (x, y), {}, True),
        pytest.param("ones_like", (x,), {}, True, marks=mark_nomd),
        ("outer", (x, y), {}, True),
        (
            "packbits",
            (MyArray(jnp.array([[[1, 0, 1], [0, 1, 0]]], dtype=jnp.uint8)),),
            {},
            True,
        ),
        ("pad", (x, 20), {}, True),
        ("partition", (x, 1), {}, True),
        ("percentile", (x, 50), {}, True),
        ("permute_dims", (x, (0, 1)), {}, True),
        ("piecewise", (x, [x < 0, x >= 0], [-1, 1]), {}, False),
        ("place", (x, x.array > jnp.mean(x.array), 0), {"inplace": False}, True),
        ("poly", (x,), {}, True),
        pytest.param("polyadd", (x,), {}, True, marks=mark_todo),
        pytest.param("polyder", (x,), {}, True, marks=mark_todo),
        pytest.param("polydiv", (x,), {}, True, marks=mark_todo),
        pytest.param("polyfit", (x,), {}, True, marks=mark_todo),
        pytest.param("polyint", (x,), {}, True, marks=mark_todo),
        pytest.param("polymul", (x,), {}, True, marks=mark_todo),
        pytest.param("polysub", (x,), {}, True, marks=mark_todo),
        pytest.param("polyval", (x,), {}, True, marks=mark_todo),
        ("positive", (x,), {}, True),
        ("pow", (x, 2), {}, True),
        ("power", (x, 2.5), {}, True),
        ("prod", (x,), {}, True),
        ("ptp", (x,), {}, True),
        ("put", (x, 0, 2), {"inplace": False}, True),
        ("quantile", (x,), {"q": 0.5}, True),
        ("rad2deg", (x,), {}, True),
        ("radians", (x,), {}, True),
        ("ravel", (x,), {}, True),
        pytest.param(
            "ravel_multi_index",
            (jnp.array([[0, 1], [0, 1]]), (2, 2)),
            {},
            True,
            marks=xfail_quax58,
        ),
        ("real", (x,), {}, True),
        ("reciprocal", (x,), {}, True),
        ("remainder", (x, y), {}, True),
        ("repeat", (x, 3), {}, True),
        ("reshape", (x, (1, -1)), {}, True),
        ("resize", (x, (1, len(x))), {}, True),
        # ("result_type", (3, x), {}, True),
        ("right_shift", (xbool, xbool), {}, True),
        ("rint", (x,), {}, True),
        ("roll", (x, 4), {}, True),
        ("rollaxis", (x, -1), {}, True),
        pytest.param("roots", (x[:, 0],), {}, True, marks=xfail_quax58),
        ("rot90", (x,), {}, True),
        ("round", (x,), {}, True),
        # pytest.param("round_", (x,), {}, True, marks=pytest.mark.deprecated),
        pytest.param("save", (x,), {}, True, marks=pytest.mark.xfail),
        pytest.param("savez", (x,), {}, True, marks=pytest.mark.xfail),
        pytest.param("searchsorted", (x,), {}, True, marks=mark_todo),
        pytest.param("select", (x,), {}, True, marks=mark_todo),
        *(
            pytest.param("setdiff1d", (x[:, 0], y[:, 0]), {}, True, marks=xfail_quax58),
            ("setdiff1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}, True),
        ),
        *(
            pytest.param("setxor1d", (x[:, 0], y[:, 0]), {}, True, marks=xfail_quax58),
            ("setxor1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}, True),
        ),
        ("shape", (x,), {}, False),
        ("sign", (x,), {}, True),
        ("signbit", (x,), {}, True),
        ("sin", (x,), {}, True),
        ("sinc", (x,), {}, True),
        ("sinh", (x,), {}, True),
        ("size", (x,), {}, False),
        ("sort", (x,), {}, True),
        ("sort_complex", (x,), {}, True),
        ("split", (x, 2), {}, True),
        ("sqrt", (x,), {}, True),
        ("square", (x,), {}, True),
        ("squeeze", (x[None],), {}, True),
        ("stack", ([x, x],), {}, True),
        ("std", (x,), {}, True),
        ("subtract", (x, y), {}, True),
        ("sum", (x,), {}, True),
        pytest.param("swapaxes", (x, 0, 1), {}, True, marks=mark_todo),
        pytest.param("take", (), {}, True, marks=mark_todo),
        pytest.param("take_along_axis", (), {}, True, marks=mark_todo),
        ("tan", (x,), {}, True),
        ("tanh", (x,), {}, True),
        ("tensordot", (x, y), {}, True),
        ("tile", (x, 3), {}, True),
        ("trace", (x,), {}, True),
        ("transpose", (x,), {}, True),
        ("tril", (x,), {}, True),
        ("tril_indices_from", (x,), {}, False),
        pytest.param(
            "trim_zeros",
            (
                MyArray(
                    jnp.concat(
                        (jnp.array([0.0, 0, 0]), x.array[:, 0], jnp.array([0.0, 0, 0]))
                    )
                ),
            ),
            {},
            True,
            marks=xfail_quax58,
        ),
        ("triu", (x,), {}, True),
        ("triu_indices_from", (x,), {}, False),
        ("true_divide", (x, y), {}, True),
        ("trunc", (x,), {}, True),
        pytest.param("ufunc", (x,), {}, True, marks=mark_todo),
        *(
            pytest.param("union1d", (x[:, 0], y[:, 0]), {}, True, marks=xfail_quax58),
            ("union1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}, True),
        ),
        *(
            pytest.param("unique", (x,), {}, True, marks=xfail_quax58),
            ("unique", (x,), {"size": x.size}, True),
        ),
        *(
            pytest.param("unique_all", (x,), {}, True, marks=xfail_quax58),
            ("unique_all", (x,), {"size": x.size}, True),
        ),
        *(
            pytest.param("unique_counts", (x,), {}, True, marks=xfail_quax58),
            ("unique_counts", (x,), {"size": x.size}, True),
        ),
        *(
            pytest.param("unique_inverse", (x,), {}, True, marks=xfail_quax58),
            ("unique_inverse", (x,), {"size": x.size}, True),
        ),
        *(
            pytest.param("unique_values", (x,), {}, True, marks=xfail_quax58),
            ("unique_values", (x,), {"size": x.size}, True),
        ),
        pytest.param(
            "unpackbits",
            (MyArray(jnp.array([[[1, 0, 1], [0, 1, 0]]], dtype=jnp.uint8)),),
            {},
            True,
            marks=mark_todo,
        ),
        ("unravel_index", (x,), {"shape": (1, 4)}, True),
        ("unwrap", (x,), {}, True),
        pytest.param("vander", (x[:, 0],), {}, True, marks=mark_todo),
        ("var", (x,), {}, True),
        ("vdot", (x, y), {}, True),
        ("vecdot", (x, y), {}, True),
        ("vsplit", (x, 2), {}, True),
        ("vstack", ([x, y],), {}, True),
        ("where", (jnp.ones_like(x.array, dtype=bool), x, y), {}, True),
        pytest.param("zeros_like", (x,), {}, True, marks=mark_nomd),
    ],
)
def test_numpy_functions(func_name, args, kw, expect_myarray):
    """Test lax vs qlax functions."""
    # Jax
    func = getattr(jnp, func_name)
    jax_args, jax_kw = jtu.map(unwrap, (args, kw), is_leaf=is_myarray)
    exp = func(*jax_args, **jax_kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed
    got = quax.quaxify(func)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)
    got = _unwrap_myarray(got, expect_myarray)

    assert jtu.all(jtu.map(jnp.allclose, got, exp))


# ###############################################################################


def test_einsum_path():
    """Test `quaxed.numpy.einsum_path`."""
    key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
    a = jr.uniform(key1, (2, 2))
    b = jr.uniform(key2, (2, 5))
    c = jr.uniform(key3, (5, 2))

    exp = jnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")
    got = quax.quaxify(jnp.einsum_path)("ij,jk,kl->il", a, b, c, optimize="greedy")

    assert jnp.array_equal(got[0], exp[0])


def test_vectorize():
    """Test `quaxed.numpy.vectorize`."""

    @quax.quaxify
    @jnp.vectorize
    def f(x):
        return jnp.add(x, 1)

    got = f(x)
    assert isinstance(got, MyArray)

    exp = f(x.array)
    assert jnp.array_equal(got.array, exp)


###############################################################################
# Linalg

x1225 = MyArray(jnp.array([[1, 2], [2, 5]], dtype=float))
xN3 = MyArray(jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("cholesky", (x1225,), {}),
        ("cross", (xN3, xN3), {}),
        ("det", (x1225,), {}),
        ("diagonal", (xN3,), {}),
        ("eig", (xN3,), {}),
        ("eigvals", (xN3,), {}),
        ("eigh", (xN3,), {}),
        ("eigvalsh", (xN3,), {}),
        pytest.param(
            "inv", (x1225,), {}, marks=pytest.mark.xfail(reason="FIXME: tracer leak")
        ),
        ("matmul", (xN3, xN3), {}),
        ("matrix_norm", (xN3,), {"ord": 2}),
        ("matrix_power", (xN3, 2), {}),
        ("matrix_rank", (xN3,), {}),
        ("matrix_transpose", (xN3,), {}),
        ("outer", (xN3[:, 0], xN3[:, 1]), {}),
        ("pinv", (xN3,), {}),
        ("qr", (xN3,), {}),
        ("slogdet", (x1225,), {}),
        pytest.param(
            "solve",
            (x1225, jnp.array([1, 2])),
            {},
            marks=pytest.mark.xfail(reason="FIXME: tracer leak"),
        ),
        ("svd", (xN3,), {}),
        ("svdvals", (xN3,), {}),
        ("tensordot", (xN3, xN3), {"axes": 1}),
        ("trace", (xN3,), {}),
        ("vecdot", (xN3[:, 0], xN3[:, 1]), {}),
        ("vector_norm", (xN3,), {"ord": 2}),
        ("norm", (xN3,), {"ord": 2}),
    ],
)
def test_linalg_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    # Jax
    func = getattr(jnp.linalg, func_name)
    jax_args, jax_kw = jtu.map(unwrap, (args, kw), is_leaf=is_myarray)
    exp = func(*jax_args, **jax_kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed
    got = quax.quaxify(func)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)
    got = jtu.map(unwrap, got, is_leaf=is_myarray)

    assert jtu.all(jtu.map(jnp.allclose, got, exp))
