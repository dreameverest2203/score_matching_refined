import jax.numpy as jnp
from jax import random as rnd
import haiku as hk
from UNet import DAE


def get_model(cfg):
    def forward_new(perturbed_x, t, sigma, is_training):
        score_model = DAE(cfg)
        return score_model(perturbed_x, t, sigma, is_training)

    f = hk.without_apply_rng(hk.transform_with_state(forward_new))

    dummy_xs, dummy_t = (
        jnp.ones((cfg.batch_size, 28, 28, cfg.num_samples)),
        jnp.zeros((cfg.batch_size,)),
    )
    init_params, init_state = f.init(
        rnd.PRNGKey(cfg.random_seed), dummy_xs, dummy_t, 0.0, True
    )
    # out, state = f.apply(params, state, dummy_xs, dummy_t, 0.0, True)
    return init_params, init_state, f
