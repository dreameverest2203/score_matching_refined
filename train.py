from models import get_model
from ode_sampler import ode_sampler
import jax.numpy as jnp
from jax import jit, value_and_grad
from jax import random as rnd
import haiku as hk
import optax
from NCSN import marginal_prob_std
from UNet import DAE
from config import get_config
from tqdm import trange
from likelihood import ode_likelihood
import matplotlib.pyplot as plt

# from langevin import chain_langevin
from jax import vmap
from jax._src.prng import PRNGKeyArray
from typing import cast
import wandb
import numpy as np
from torchvision.transforms import RandAugment
import pdb
import torch
from ode_sampler import ode_sampler

sigma = 25.0


def train_wrapper(train_dataloader, val_dataloader, cfg):

    init_params, init_state, f = get_model(cfg)
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(init_params)
    data_shape = (-1, 28, 28, 1)

    @jit
    def full_loss(params, rng, state, x):
        rng_1, rng_2 = rnd.split(rng, 2)
        random_t = rnd.uniform(rng_1, (x.shape[0],), minval=1e-5, maxval=1)
        std = marginal_prob_std(random_t, sigma)
        std = std[:, None, None, None]
        x_stacked = jnp.concatenate(cfg.num_samples * [x], axis=-1)
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
        # 1) plot noisy inp, inp, denoised inp
        # denoised_img = score + perturbed_x

        # fig,ax = plt.subplots(16,1)

        # pdb.set_trace()
        # loss = jnp.mean(jnp.sum((x - x_est) ** 2, axis=(1, 2, 3)))
        return loss, (new_state, x, score * std ** 2, perturbed_x)

    def eval_val_callback(
        params: hk.Params,
        state: hk.State,
        rng: PRNGKeyArray,
        val_dataloader,
        sigma: float,
        n_samples: int,
        verbose: bool = False,
    ):
        """Callback to eval on a batch of data from the test set"""
        rng_1, rng_2, rng_3 = rnd.split(rng, 3)

        val_loss = 0
        for i, (xs, _) in enumerate(val_dataloader):
            xs = xs.permute(0, 2, 3, 1).numpy().reshape(data_shape)
            val_loss += full_loss(params, rng, state, xs)[0]
        val_loss /= i + 1  # type: ignore
        xs = cast(jnp.ndarray, xs)  # type: ignore
        samples_shape = tuple([n_samples] + list(xs.shape[1:]))

        # if verbose:
        #     val_likelihood = ode_likelihood(
        #         f, rng_1, xs, params, state, len(xs), eps=1e-5
        #     )
        #     init_xs = rnd.normal(rng_2, shape=samples_shape)
        #     langevin_samples = chain_langevin(
        #         f,
        #         params,
        #         state,
        #         rnd.split(rng_2, len(init_xs)),
        #         xs,
        #         jnp.array([1.0, 0.5, 0.2, 0.1, 0.01]),
        #         cfg.langevin_stepsize,
        #         10_000,
        #         2_000,
        #     )
        #     ode_samples = vmap(ode_sampler, in_axes=(None, None, None, 0, None))(
        #         f, params, state, rnd.normal(rng_3, samples_shape), 1e-3
        #     )
        #     return (val_loss, val_likelihood, langevin_samples, ode_samples)
        # else:
        #     return val_loss

    @jit
    def step(params, state, opt_state, xs, aug_xs, rng_key):
        """Compute the gradient for a batch and update the parameters"""
        rng_key_1, rng_key_2 = rnd.split(rng_key, 2)
        # rng_key_1 = rnd.PRNGKey(0)
        # xs = xs[rnd.choice(rng_key_1, len(xs), shape=(cfg.batch_size,))]
        # aux_loss = full_loss(params, rng_key, state, xs)
        # pdb.set_trace
        (
            (loss_value, (new_state, x, denoised_diff, perturbed_x)),
            grads,
        ) = value_and_grad(full_loss, has_aux=True)(params, rng_key_1, state, xs)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        extra_stuff = x, denoised_diff, perturbed_x
        return params, new_state, opt_state, loss_value, rng_key_2, extra_stuff

    def augmentation(x):
        x = torch.tensor(255 * x, dtype=torch.uint8)
        x = torch.permute(x, (0, 3, 1, 2))
        aux = RandAugment(3, 5, 31).forward(x)
        aug_x = torch.tensor(aux / 255.0, dtype=torch.float16)
        return aug_x

    def training_loop(
        train_dataloader,
        val_dataloader,
        params: hk.Params,
        state: hk.State,
        num_epochs: int,
        opt_state,
        rng_key: PRNGKeyArray,
        sampling_log_interval: int = 500,
    ):
        train_loss = []
        # x0, _ = next(train_dataloader)
        # for x0, _ in train_dataloader:
        #     break
        with trange(num_epochs) as t:
            for i in t:
                for j, (x, _) in enumerate(train_dataloader):
                    # x = x0  # type: ignore
                    x = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
                    # pdb.set_trace()
                    aug_x = augmentation(x).numpy().reshape(data_shape)
                    params, state, opt_state, loss_value, rng_key, extras = step(
                        params, state, opt_state, x, aug_x, rng_key
                    )
                    if j % 50 == 0:
                        if cfg.use_wandb:
                            wandb.log(
                                {"Training Loss": float(loss_value)}
                            )  # Log occasionally
                        t.set_description("Training Loss: {:3f}".format(loss_value))

                if i % sampling_log_interval == 0 and i != 0:
                    (
                        val_loss,
                        val_likelihood,
                        langevin_samples,
                        ode_samples,
                    ) = eval_val_callback(
                        params,
                        state,
                        rng_key,
                        val_dataloader,
                        cfg.sigma,
                        n_samples=2,
                        verbose=True,
                    )
                    if cfg.use_wandb:
                        wandb.log(
                            data={
                                "Training Loss": float(loss_value),  # type: ignore
                                "Val Loss": float(val_loss),
                                "Langevin Samples": None,
                                "ODE Samples": None,
                            },
                        )
                else:
                    pass
                    # val_loss = eval_val_callback(
                    #     params,
                    #     state,
                    #     rng_key,
                    #     val_dataloader,
                    #     cfg.sigma,
                    #     n_samples=2,
                    #     verbose=False,
                    # )
                    # if cfg.use_wandb:
                    #     wandb.log(
                    #         data={
                    #             "Training Loss": float(loss_value),  # type: ignore
                    #             "Val Loss": float(val_loss),  # type: ignore
                    #         },
                    #     )
                x, denoised_diff, perturbed_x = extras  # type: ignore
                fig, axs = plt.subplots(3, 16)
                for im, ax in zip(x[:16], axs[0]):
                    ax.imshow(im)
                    ax.axis("off")
                for im, ax in zip(perturbed_x[:16], axs[1]):
                    ax.imshow(im)
                    ax.axis("off")
                for im, ax in zip(perturbed_x[:16] + denoised_diff[:16], axs[2]):
                    ax.imshow(im)
                    ax.axis("off")
                plt.savefig(f"model_outputs_epoch_{i}.png")  # type: ignore
                if cfg.use_wandb:
                    wandb.log(
                        {"training_figure": wandb.Image(f"model_outputs_epoch_{i}.png")}
                    )

        return train_loss, params, state

    if cfg.use_wandb:
        wandb.init(project="ncsnpp", config=cfg)
    train_loss, params, state = training_loop(
        train_dataloader,
        val_dataloader,
        init_params,
        init_state,
        cfg.num_epochs,
        opt_state,
        rnd.PRNGKey(0),
    )

    return train_loss, params, state
