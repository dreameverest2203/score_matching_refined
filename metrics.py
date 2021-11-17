from config import get_config
from train import f
import jax
import jax.numpy as jnp
from NCSN import marginal_prob_std

conf = get_config()


def eval(params, state, x):
    key = jax.random.PRNGKey(100)
    dummy_t = jax.random.uniform(key, (x.shape[0],), minval=1e-5, maxval=1.0)
    z = jax.random.normal(key, (x.shape[0], conf.num_samples * conf.data_dim))
    std = marginal_prob_std(dummy_t, conf.sigma)
    x = jnp.concatenate(conf.num_samples * [x], axis=-1)
    perturbed_x = x + z * std[:, None]
    score, state = f.apply(params, state, perturbed_x, dummy_t[:, None], conf.sigma)
    score = jnp.concatenate(conf.num_samples * [score], axis=-1)
    ground_truth = -z / std[:, None]
    return jnp.mean(jnp.sum((ground_truth - score) ** 2, axis=-1)) / conf.num_samples
    # return jnp.mean(jnp.linalg.norm(ground_truth - score, ord = conf.num_samples, axis = -1))
    # change this to 2-norm and then rescale by num of samples
