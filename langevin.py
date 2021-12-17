import jax
import jax.numpy as jnp
from functools import partial
from jax import random, vmap
from train import f
from NCSN import marginal_prob_std
from config import get_config
import pdb


conf = get_config()


def itemwise_f(params, state, x, t):
    """Calls f with a single x, t, handling reshaping
    which is necesary for the batchnorm"""
    assert len(x.shape) == 3
    assert len(t.shape) == 1 or len(t.shape) == 0
    expanded_x = jnp.expand_dims(x, axis=0)
    expanded_t = jnp.expand_dims(t, axis=0)
    return f.apply(params, state, expanded_x, expanded_t, conf.sigma, is_training=False)


@partial(jax.jit, static_argnames=("n_iterations", "n_burn_in"))
def annealed_langevin(
    params, state, key, x_0, t_array, epsilon, n_iterations, n_burn_in
):
    def langevin(carry, t_array):
        x_0, state, key = carry
        t = t_array
        key, _ = random.split(key)

        def inner_fn(carry, key):
            x, state = carry
            key, _ = random.split(key)
            out, state = itemwise_f(params, state, x, t)
            # out = jnp.concatenate(conf.num_samples * [out.squeeze(axis=0)], axis=-1)
            # score = (out - x) / marginal_prob_std(t, conf.sigma) ** 2
            # pdb.set_trace()
            score = out.squeeze(axis=0)
            x = (
                x
                + epsilon * score
                + jnp.sqrt(2 * epsilon) * random.normal(key, shape=x.shape)
            )
            # x = jnp.clip(x, 0.0, 1.0)
            return (x, state), x

        (x, state), xs = jax.lax.scan(
            inner_fn, (x_0, state), random.split(key, n_iterations)
        )
        return (x, state, key), xs

    (x, state, key), xs = jax.lax.scan(langevin, (x_0, state, key), t_array)
    return xs[-1, n_burn_in:, :]


chain_langevin = vmap(annealed_langevin, (None, None, 0, 0, None, None, None, None))
