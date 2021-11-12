import jax
import jax.numpy as jnp
from functools import partial
from jax import random, vmap
from train import f
from config import get_config

conf = get_config()


def itemwise_f(params, state, x, std):
    """Calls f with a single x, std, handling reshaping 
    which is necesary for the batchnorm"""
    assert len(x.shape) == 1
    assert len(std.shape) == 1 or len(std.shape) == 0
    return f.apply(params, state, x.reshape(1, -1), std.reshape(1, -1), conf.sigma,)


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
            x = (
                x
                + epsilon * out.squeeze()
                + jnp.sqrt(2 * epsilon) * random.normal(key, shape=x.shape)
            )
            return (x, state), x

        (x, state), xs = jax.lax.scan(
            inner_fn, (x_0, state), random.split(key, n_iterations)
        )
        return (x, state, key), xs

    (_, state, key), xs = jax.lax.scan(langevin, (x_0, state, key), std_array)
    return xs[-1, n_burn_in:, :]


chain_langevin = vmap(annealed_langevin, (None, None, 0, 0, None, None, None, None))
