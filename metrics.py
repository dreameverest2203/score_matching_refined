from config import get_config
from train import f
import jax
import jax.numpy as jnp
from NCSN import marginal_prob_std
import haiku as hk
from typing import Tuple, Callable

conf = get_config()


def eval(params, state, x):
    key = jax.random.PRNGKey(100)
    ts = jax.random.uniform(key, (x.shape[0],), minval=1e-5, maxval=1.0)
    z = jax.random.normal(key, (x.shape[0], conf.num_samples * conf.data_dim))
    std = marginal_prob_std(ts, conf.sigma)
    x = jnp.concatenate(conf.num_samples * [x], axis=-1)
    perturbed_x = x + z * std[:, None]
    score, state = f.apply(params, state, perturbed_x, ts[:, None], conf.sigma)
    ground_truth = -z / std[:, None]
    return jnp.mean(jnp.sum((ground_truth - score) ** 2, axis=-1)) / conf.num_samples
    # return jnp.mean(jnp.linalg.norm(ground_truth - score, ord = conf.num_samples, axis = -1))
    # change this to 2-norm and then rescale by num of samples


def numerical_exact_density(
    score: Callable[[jnp.ndarray], jnp.ndarray],
    lims: Tuple[float, float],
    dx,
    data_dim: int,
):
    """Given score function f, compute density at a grid of points from -lim to lim with spacing dx

    Using numerical integration and approximation of the partition function, we
    compute the density induced by the score function f. Note that this the
    complexity of this will be exponential in the dimension

    Args:
        f: A function that takes in xs and gives the score function grad[log(x)] at that point

    Returns:
        A grid of densities at points
    """
    # First compute along axis 0
    x_0s = jnp.arange(lims[0], lims[1], dx)[..., None]
    if data_dim != 1:
        x_0s = jnp.vstack((x_0s, jnp.ones((len(x_0s), data_dim - 1)) * lims[0]))
    scores = score(x_0s)
    unnormalized_log_probs = jnp.cumsum(dx * scores)

    # TODO: Integrate along remaining axes

    return unnormalized_log_probs
