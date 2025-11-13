"""Tests for support for diffrax."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.tree as jtu
import pytest

import quax

from .myarray import MyArray


@pytest.mark.parametrize(
    "solver",
    [
        dfx.Dopri5(),
        dfx.Dopri8(),
        dfx.Tsit5(),
        dfx.Euler(),
        dfx.Heun(),
        dfx.Midpoint(),
        dfx.Ralston(),
    ],
)
@pytest.mark.parametrize(
    ("sign", "kwargs"),
    [
        (1, {}),
        (1, {"saveat": dfx.SaveAt(ts=[0.0, 1.0, 2.0, 3.0])}),
        (1, {"saveat": dfx.SaveAt(steps=True)}),
        (1, {"saveat": dfx.SaveAt(dense=True)}),
        (1, {"stepsize_controller": dfx.PIDController(rtol=1e-5, atol=1e-5)}),
        (1, {"stepsize_controller": dfx.ConstantStepSize()}),
        (1, {"adjoint": dfx.RecursiveCheckpointAdjoint()}),
        (1, {"adjoint": dfx.DirectAdjoint()}),
        (1, {"max_steps": 1000}),
        (1, {"saveat": dfx.SaveAt(t0=True, t1=True)}),
        (1, {"saveat": dfx.SaveAt(subs=dfx.SubSaveAt(t0=True, t1=True))}),
        (
            1,
            {
                "event": dfx.Event(dfx.steady_state_event(rtol=1e-3, atol=1e-3)),
                "stepsize_controller": dfx.PIDController(rtol=1e-5, atol=1e-5),
            },
        ),
        (-1, {}),  # backward integration
    ],
)
def test_diffrax_integration(solver, sign, kwargs):
    """Integration test: custom VJP works with diffrax ODE solver."""

    # Skipping incompatible Euler test cases
    if isinstance(solver, dfx.Euler) and not isinstance(
        kwargs.get("stepsize_controller", None), dfx.ConstantStepSize
    ):
        pytest.skip("Euler solver requires ConstantStepSize controller.")

    # Define vector field with args
    def vector_field(t, y, args):
        decay_rate = args["decay"]
        return -decay_rate * y

    term = dfx.ODETerm(vector_field)
    y0 = jnp.array([2.0, 3.0])
    kwargs.setdefault("args", {"decay": 1.0})

    sol_jax = dfx.diffeqsolve(
        term, solver, t0=0, t1=sign * 3, dt0=sign * 0.1, y0=y0, **kwargs
    )

    sol_quax = quax.quaxify(dfx.diffeqsolve)(
        term, solver, t0=0, t1=sign * 3, dt0=sign * 0.1, y0=y0, **kwargs
    )

    # Check solutions match
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ts, sol_jax.ts))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys, sol_jax.ys))


def test_diffrax_gradient():
    """Test that gradients work through diffrax with quax."""

    def vector_field(t, y, args):
        return -args * y

    def loss(decay_rate):
        term = dfx.ODETerm(vector_field)
        solver = dfx.Dopri5()
        y0 = jnp.array([2.0])
        sol = quax.quaxify(dfx.diffeqsolve)(
            term, solver, t0=0, t1=1, dt0=0.1, y0=y0, args=decay_rate
        )
        return jnp.sum(sol.ys[-1] ** 2)

    decay_rate = 1.0
    grad_fn = jax.grad(loss)
    grad = grad_fn(decay_rate)

    # Just check that gradient computation doesn't crash and produces a result
    assert jnp.isfinite(grad)


def test_diffrax_multiple_terms():
    """Test diffrax with multiple terms (MultiTerm with WeaklyDiagonalControlTerm)."""

    def drift(t, y, args):
        return -y

    def diffusion(t, y, args):
        # Return 1D array for WeaklyDiagonalControlTerm
        return 0.1 * jnp.ones(2)

    key = jax.random.PRNGKey(0)
    brownian = dfx.VirtualBrownianTree(t0=0, t1=1, tol=1e-3, shape=(2,), key=key)
    terms = dfx.MultiTerm(
        dfx.ODETerm(drift), dfx.WeaklyDiagonalControlTerm(diffusion, brownian)
    )
    solver = dfx.Euler()
    y0 = jnp.array([2.0, 3.0])

    sol_jax = dfx.diffeqsolve(terms, solver, t0=0, t1=1, dt0=0.01, y0=y0)

    sol_quax = quax.quaxify(dfx.diffeqsolve)(terms, solver, t0=0, t1=1, dt0=0.01, y0=y0)

    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ts, sol_jax.ts))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys, sol_jax.ys))


def test_diffrax_implicit_solver():
    """Test implicit solver with quax."""

    def vector_field(t, y, args):
        # Stiff ODE: dy/dt = -100*y
        return -100.0 * y

    term = dfx.ODETerm(vector_field)
    solver = dfx.Kvaerno5()  # Implicit solver for stiff problems
    y0 = jnp.array([1.0])
    # Need adaptive controller with tolerances for implicit solver
    stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    kwargs = dict(
        t0=0, t1=0.1, dt0=0.01, y0=y0, stepsize_controller=stepsize_controller
    )
    sol_jax = dfx.diffeqsolve(term, solver, **kwargs)
    sol_quax = quax.quaxify(dfx.diffeqsolve)(term, solver, **kwargs)

    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ts, sol_jax.ts))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys, sol_jax.ys))


def test_diffrax_pytree_state():
    """Test diffrax with PyTree state (dict of arrays)."""

    def vector_field(t, y, args):
        return {"x": -y["x"], "v": -2 * y["v"]}

    term = dfx.ODETerm(vector_field)
    solver = dfx.Dopri5()
    y0 = {"x": jnp.array([1.0]), "v": jnp.array([2.0])}

    sol_jax = dfx.diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)

    sol_quax = quax.quaxify(dfx.diffeqsolve)(term, solver, t0=0, t1=1, dt0=0.1, y0=y0)

    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ts, sol_jax.ts))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys["x"], sol_jax.ys["x"]))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys["v"], sol_jax.ys["v"]))


def test_diffrax_integration_myarray():
    """Integration test: custom VJP works with diffrax ODE solver with MyArray.

    Note: throw=True is not compatible with custom ArrayValue types like MyArray
    due to how equinox.error_if constructs conditional branches. This is a known
    limitation.
    """

    # Define vector field with args
    def vector_field(t, y, args):
        decay_rate = args["decay"]
        return -decay_rate * y

    solver = dfx.Dopri5()
    term = dfx.ODETerm(vector_field)
    y0 = MyArray(jnp.array([2.0, 3.0]))
    kwargs = {"args": {"decay": 1.0}}

    sol_jax = dfx.diffeqsolve(term, solver, t0=0, t1=3, dt0=0.1, y0=y0.array, **kwargs)

    # TODO: get throw=True to work
    # Note: throw=False because throw=True with custom ArrayValue types
    # hits PyTree structure mismatch in error_if branches
    sol_quax = quax.quaxify(dfx.diffeqsolve)(
        term, solver, t0=0, t1=3, dt0=0.1, y0=y0, throw=False, **kwargs
    )

    # Check solutions match
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ts, sol_jax.ts))
    assert jtu.all(jtu.map(jnp.allclose, sol_quax.ys.array, sol_jax.ys))
