from scipy import integrate
import jax
import jax.numpy as jnp
from train import f
from config import get_config
import numpy as np
from UNet import marginal_prob_std
import pdb

conf = get_config()


def ode_sampler(
    params,
    state,
    init_x,
    eps=1e-3,
):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      params: A dictionary that contains model parameters.
      state: State of the model
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      eps: The smallest time step for numerical stability.
    """
    time_shape = (conf.chain_length,)
    sample_shape = (conf.chain_length, 28, 28, conf.num_samples)

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        out = f.apply(params, state, sample, time_steps, conf.sigma, False)
        score = out[0]
        # score = (
        #     jnp.concatenate([out[0]] * conf.num_samples, axis=-1) - sample
        # ) / marginal_prob_std(time_steps, conf.sigma) ** 2
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones(time_shape) * t
        g = conf.sigma ** t
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(
        ode_func,
        (1.0, eps),
        np.asarray(init_x).reshape(-1),
        rtol=conf.error_tolerance,
        atol=conf.error_tolerance,
        method="RK45",
    )
    print(f"Number of function evaluations: {res.nfev}")
    x = jnp.asarray(res.y[:, -1]).reshape(shape)

    return x
