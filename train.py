import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import random as rnd
import haiku as hk
import optax

# from NCSN import NCSN
from NCSN import marginal_prob_std
from UNet import DAE
from config import get_config
import pdb


conf = get_config()


def forward_new(perturbed_x, t, sigma):
    # score_model = NCSN(conf.layer_sizes, conf.embed_dim)
    score_model = DAE()
    return score_model(perturbed_x, t, sigma)


f = hk.without_apply_rng(hk.transform_with_state(forward_new))
dummy_xs, dummy_t = jnp.ones((conf.batch_size, 28, 28, conf.num_samples)), jnp.zeros(
    (conf.batch_size, 1, 1, conf.num_samples)
)
# dummy_xs, dummy_t = jnp.ones(
#     (conf.batch_size, conf.data_dim * conf.num_samples)
# ), jnp.ones(conf.batch_size)


params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma)
out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma)


@jit
def full_loss(params, rng, state, x, sigma):
    rng_1, rng_2 = rnd.split(rng, 2)
    # random_t = rnd.uniform(rng_1, (x.shape[0],), minval=1e-5, maxval=1.0)
    random_t = rnd.uniform(
        rng_1, (x.shape[0], 1, 1, conf.num_samples), minval=1e-5, maxval=1.0
    )
    std = marginal_prob_std(random_t, sigma)
    # perturbed_x = jnp.concatenate(conf.num_samples * [x], axis=-1)
    perturbed_x = jnp.concatenate(conf.num_samples * [x], axis=-1)
    z = rnd.normal(rng_2, perturbed_x.shape)
    perturbed_x = perturbed_x + z * std
    # pdb.set_trace()
    output, new_state = f.apply(params, state, perturbed_x, random_t, conf.sigma)
    loss = jnp.mean(jnp.mean(jnp.sum((output - x) ** 2, axis=(1, 2, 3)), axis=-1))
    # loss = jnp.mean(jnp.sum((output - x) ** 2, axis=1), axis=-1)
    return loss, new_state


@jit
def step(params, state, opt_state, xs, rng_key, sigma):
    """Compute the gradient for a batch and update the parameters"""
    rng_key_1, rng_key_2, rng_key_3 = rnd.split(rng_key, 3)
    xs = xs[rnd.choice(rng_key_1, len(xs), shape=(conf.batch_size,))]
    (loss_value, new_state), grads = value_and_grad(full_loss, has_aux=True)(
        params, rng_key_2, state, xs, sigma
    )
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_state, opt_state, loss_value, rng_key_3


opt = optax.adam(conf.lr)
opt_state = opt.init(params)


def training_loop(params, state, num_epochs, opt_state, xs, rng_key):
    # Get the initial set of parameters
    train_loss = []
    log_period = 1000
    for i in range(num_epochs):
        # start_time = time.time()
        params, state, opt_state, loss_value, rng_key = step(
            params, state, opt_state, xs, rng_key, conf.sigma
        )
        # epoch_time = time.time() - start_time
        if i % log_period == 0:
            print(f"Epoch {i}: Training Loss: {loss_value}\n")
    fullloss = full_loss(params, rng_key, state, xs, conf.sigma)
    return fullloss, train_loss, params, state


def train_wrapper(params, state, num_epochs, x):
    fullloss, train_loss, params, state = training_loop(
        params, state, num_epochs, opt_state, x, rnd.PRNGKey(0)
    )
    return fullloss, train_loss, params, state
