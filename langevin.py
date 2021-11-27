import jax
import jax.numpy as jnp
from functools import partial
from jax import random, vmap
from train import f
from NCSN import marginal_prob_std
from config import get_config
import pdb
import warnings

warnings.filterwarnings("ignore")


conf = get_config()


def itemwise_f(params, state, x, std):
    """Calls f with a single x, std, handling reshaping
    which is necesary for the batchnorm"""
    assert len(x.shape) == 3
    assert len(std.shape) == 1 or len(std.shape) == 0
    expanded_x = jnp.expand_dims(x, axis=0)
    expanded_std = jnp.expand_dims(std, axis=(0, 1, 2, 3))
    expanded_std = jnp.tile(expanded_std, (1, 1, 1, conf.num_samples))
    return f.apply(
        params, state, expanded_x, expanded_std, conf.sigma, is_training=False
    )


@partial(jax.jit, static_argnames=("n_iterations", "n_burn_in"))
def annealed_langevin(
    params, state, key, x_0, std_array, epsilon, n_iterations, n_burn_in
):
    def langevin(carry, std_array):
        x_0, state, key = carry
        std = std_array
        key, _ = random.split(key)

        def inner_fn(carry, key):
            x, state = carry
            key, _ = random.split(key)
            out, state = itemwise_f(params, state, x, std)
            out = jnp.concatenate(conf.num_samples * [out.squeeze(axis=0)], axis=-1)
            score = (out - x) / marginal_prob_std(std, conf.sigma) ** 2
            x = (
                x
                + epsilon * score
                + jnp.sqrt(2 * epsilon) * random.normal(key, shape=x.shape)
            )
            x = jnp.clip(x, 0.0, 1.0)
            return (x, state), x

        (x, state), xs = jax.lax.scan(
            inner_fn, (x_0, state), random.split(key, n_iterations)
        )
        return (x, state, key), xs

    (x, state, key), xs = jax.lax.scan(langevin, (x_0, state, key), std_array)
    return xs[-1, n_burn_in:, :]


chain_langevin = vmap(annealed_langevin, (None, None, 0, 0, None, None, None, None))
