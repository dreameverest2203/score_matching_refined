import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
from config import get_config
import pdb
from UNet import marginal_prob_std
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pickle
import tqdm

conf = get_config()


def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
    standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[2:])
    return -N / 2.0 * jnp.log(2 * np.pi * sigma ** 2) - jnp.sum(
        z ** 2, axis=(2, 3, 4)
    ) / (2 * sigma ** 2)


def score_fn(f, params, state, x, t):
    perturbed_x = jnp.concatenate([x] * conf.num_samples, axis=-1)
    t = jnp.squeeze(t, axis=1)
    out = f.apply(params, state, perturbed_x, t, conf.sigma, True)
    # score = (out - x) / marginal_prob_std(t, conf.sigma) ** 2
    den = marginal_prob_std(t, conf.sigma) ** 2
    # score = (jnp.concatenate([out[0]] * conf.num_samples, axis=-1) - perturbed_x) / den[
    #     :, None, None, None
    # ]
    # score = jnp.mean(score, axis=-1)  # this line won't work for RGB images
    score = out[0]
    # return score[:, :, :, None]
    return score


def ode_likelihood(
    f, rng, x, params, state, batch_size, eps=1e-5,
):
    """Compute the likelihood with probability flow ODE.

    Args:
      rng: A JAX random state.
      x: Input data.
      score_model: A `flax.linen.Module` instance that represents the architecture
        of the score-based model.
      params: A dictionary that contains model parameters.
      eps: A `float` number. The smallest time step for numerical stability.

    Returns:
      z: The latent code for `x`.
      bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    rng, step_rng = jax.random.split(rng)

    def divergence_eval(sample, time_steps):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        epsilon = jax.random.normal(rng, sample.shape)
        score_e_fn = lambda x: jnp.sum(
            score_fn(f, params, state, x, time_steps) * epsilon
        )
        grad_score_e = jax.grad(score_e_fn)(sample)
        return jnp.sum(grad_score_e * epsilon, axis=(1, 2, 3))

    shape = x.shape
    time_shape = (shape[0], shape[1])

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        sample = jnp.squeeze(sample, axis=1)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        score = score_fn(f, params, state, sample, time_steps)
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        # Obtain x(t) by solving the probability flow ODE.
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(shape)
        sample = jnp.squeeze(sample, axis=1)
        time_steps = jnp.asarray(time_steps, dtype=jnp.float32).reshape(time_shape)
        # Compute likelihood.
        div = divergence_eval(sample, time_steps)
        return np.asarray(div).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = np.ones(time_shape) * t
        sample = x[: -shape[0] * shape[1]]
        logp = x[-shape[0] * shape[1] :]
        g = conf.sigma ** t
        sample_grad = -0.5 * g ** 2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g ** 2 * divergence_eval_wrapper(sample, time_steps)
        return np.concatenate([sample_grad, logp_grad], axis=0)

    init = jnp.concatenate(
        [x.reshape((-1,)), jnp.zeros((shape[0] * shape[1],))], axis=0
    )
    # Black-box ODE solver
    res = integrate.solve_ivp(
        ode_func, (eps, 1.0), np.asarray(init), rtol=1e-5, atol=1e-5, method="RK45"
    )
    zp = jnp.asarray(res.y[:, -1])
    z = zp[: -shape[0] * shape[1]].reshape(shape)
    delta_logp = zp[-shape[0] * shape[1] :].reshape((shape[0], shape[1]))
    sigma_max = marginal_prob_std(1.0, conf.sigma)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[2:])
    bpd = bpd / N + 8.0
    return z, bpd


# with open(f"model_weights/dae_model_{conf.num_samples}samples.pickle", "rb") as handle:
#     params, state = pickle.load(handle)

# sample_batch_size = 64
# dataset = MNIST(".", train=False, transform=transforms.ToTensor(), download=True)
# data_loader = DataLoader(
#     dataset,
#     batch_size=sample_batch_size,
#     shuffle=True,
# )
# all_bpds = 0.0
# all_items = 0
# rng = jax.random.PRNGKey(100)
# tqdm_data = tqdm.tqdm(data_loader)
# for x, _ in tqdm_data:
#     x = x.permute(0, 2, 3, 1).cpu().numpy().reshape((-1, 1, 28, 28, 1))
#     rng, step_rng = jax.random.split(rng)
#     z = jax.random.uniform(step_rng, x.shape)
#     x = (x * 255.0 + z) / 256.0
#     _, bpd = ode_likelihood(step_rng, x, params, state, sample_batch_size, eps=1e-5)
#     all_bpds += bpd.sum()
#     all_items += bpd.shape[0] * bpd.shape[1]
#     tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))