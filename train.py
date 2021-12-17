import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import random as rnd
import haiku as hk
import optax
from NCSN import marginal_prob_std
from UNet import DAE
from config import get_config
import pdb
from tqdm import trange


conf = get_config()


def forward_new(perturbed_x, t, sigma, is_training):
    score_model = DAE()
    return score_model(perturbed_x, t, sigma, is_training)


f = hk.without_apply_rng(hk.transform_with_state(forward_new))

dummy_xs, dummy_t = jnp.ones((conf.batch_size, 28, 28, conf.num_samples)), jnp.zeros(
    (conf.batch_size,)
)
params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma, True)
out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma, True)


@jit
def full_loss(params, rng, state, x, sigma):
    rng_1, rng_2 = rnd.split(rng, 2)
    random_t = rnd.uniform(rng_1, (x.shape[0],), minval=1e-5, maxval=1)
    std = marginal_prob_std(random_t, sigma)
    std = std[:, None, None, None]
    x_stacked = jnp.concatenate(conf.num_samples * [x], axis=-1)
    z = rnd.normal(rng_2, x_stacked.shape)
    perturbed_x = x + z * std

    # perturbed_x = jnp.clip(perturbed_x, 0.0, 1.0)
    # x_est, new_state = f.apply(
    #     params, state, perturbed_x, random_t, sigma, is_training=True
    # )
    score, new_state = f.apply(
        params, state, perturbed_x, random_t, sigma, is_training=True
    )
    loss = jnp.mean(jnp.sum((score * std + z) ** 2, axis=(1, 2, 3)))
    # loss = jnp.mean(jnp.sum((x - x_est) ** 2, axis=(1, 2, 3)))
    return loss, new_state


@jit
def step(params, state, opt_state, xs, rng_key, sigma):
    """Compute the gradient for a batch and update the parameters"""
    rng_key_1, rng_key_2, rng_key_3 = rnd.split(rng_key, 3)
    # xs = xs[rnd.choice(rng_key_1, len(xs), shape=(conf.batch_size,))]
    (loss_value, new_state), grads = value_and_grad(full_loss, has_aux=True)(
        params, rng_key_2, state, xs, sigma
    )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_state, opt_state, loss_value, rng_key_3


opt = optax.adam(conf.lr)
opt_state = opt.init(params)
data_shape = (-1, 28, 28, 1)


def training_loop(dataloader, params, state, num_epochs, opt_state, rng_key):
    train_loss = []
    with trange(num_epochs) as t:
        for i in t:
            for x, y in dataloader:
                x = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
                rng_key, rng = rnd.split(rng_key, 2)
                params, state, opt_state, loss_value, rng_key = step(
                    params, state, opt_state, x, rng_key, conf.sigma
                )
                t.set_description("Training Loss: {:3f}".format(loss_value))
    return train_loss, params, state


def train_wrapper(dataloader, params, state, num_epochs):
    train_loss, params, state = training_loop(
        dataloader, params, state, num_epochs, opt_state, rnd.PRNGKey(0)
    )
    return train_loss, params, state
