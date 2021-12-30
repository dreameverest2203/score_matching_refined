from scipy import integrate
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from jax import jit

# from UNet import marginal_prob_std


def ode_sampler(
    f,
    params,
    state,
    init_x,
    global_sigma,
    eps=1e-3,
    chain_length=1,
    error_tol=1e-4,
    num_samples=1,
):
    time_shape = (chain_length,)
    sample_shape = (chain_length, 28, 28, num_samples)

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        out = f.apply(params, state, sample, time_steps, global_sigma, False)
        score = out[0]
        # score = (
        #     jnp.concatenate([out[0]] * conf.num_samples, axis=-1) - sample
        # ) / marginal_prob_std(time_steps, conf.sigma) ** 2
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def jax_score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = sample.reshape(sample_shape)
        time_steps = time_steps.reshape(time_shape)
        out = f.apply(params, state, sample, time_steps, global_sigma, False)
        score = out[0]
        # score = (
        #     jnp.concatenate([out[0]] * conf.num_samples, axis=-1) - sample
        # ) / marginal_prob_std(time_steps, conf.sigma) ** 2
        return jnp.asarray(score).reshape((-1,)).astype(jnp.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones(time_shape) * t
        g = global_sigma ** t
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    @jit
    def jax_ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones(time_shape) * t
        g = global_sigma ** t
        return -0.5 * (g ** 2) * jax_score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    jax_odeint_fn = lambda y, t: jax_ode_func(t, y)

    # need to use rtol instead of atol really
    result_0 = odeint(
        jax_odeint_fn,
        jnp.array(init_x).reshape(-1),
        jnp.array([1.0, eps]),
        atol=error_tol,
    )
    result = integrate.solve_ivp(
        ode_func,
        (1.0, eps),
        np.asarray(init_x).reshape(-1),
        # rtol=conf.error_tolerance,
        atol=error_tol,
        method="RK45",
    )
    assert np.linalg.norm(result_0[0] - result.y[:, 0], 1) < 0.05
    print(f"Number of function evaluations: {result.nfev}")
    # x = jnp.asarray(result.y[:, -1]).reshape(shape)

    return result.y
