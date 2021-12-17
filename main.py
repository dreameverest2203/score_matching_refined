import jax.numpy as jnp
from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt
from ode_sampler import ode_sampler
from train import f, train_wrapper
import numpy as np
import jax.random as rnd
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pickle
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image


if __name__ == "__main__":
    conf = get_config()

    train_dataset = MNIST(
        ".", train=True, transform=transforms.ToTensor(), download=True
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2
    )

    test_dataset = MNIST(
        ".", train=False, transform=transforms.ToTensor(), download=True
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2
    )

    # Load Saved Model
    # with open(f"model_weights/dae_model_{conf.num_samples}samples.pickle", "rb") as handle:
    #     params, state = pickle.load(handle)

    # # Initialization of Params
    dummy_xs, dummy_t = (
        jnp.ones((conf.batch_size, 28, 28, conf.num_samples)),
        jnp.zeros((conf.batch_size,)),
    )

    params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma, True)
    out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma, True)
    # Model Training
    train_loss, params, state = train_wrapper(
        train_data_loader, test_data_loader, params, state, conf.num_epochs
    )
    print("Model training done\n")

    with open(
        f"model_weights/dae_model_{conf.num_samples}samples.pickle", "wb"
    ) as handle:
        pickle.dump((params, state), handle)

    # # Langevin Chain
    # init_x = rnd.uniform(
    #     rnd.PRNGKey(4),
    #     (conf.chain_length, 28, 28, conf.num_samples),
    #     minval=0.0,
    #     maxval=1.0,
    # )
    # key_array = rnd.split(rnd.PRNGKey(15), init_x.shape[0])
    # out = chain_langevin(
    #     params,
    #     state,
    #     key_array,
    #     init_x,
    #     jnp.flip(jnp.array([1e-3, 5e-3, 25e-3, 125e-3, 625e-3, 1])),
    #     1e-3,
    #     50,
    #     0,
    # )

    for i in range(10):
        init_x = rnd.normal(rnd.PRNGKey(i), (1, 28, 28, conf.num_samples),)

        out = ode_sampler(f, params, state, init_x)

        # out = jnp.clip(out, 0.0, 1.0)
        grid_img = np.array(out)
        grid_img = torch.from_numpy(grid_img)
        indices = np.arange(0, 50, 10)
        # MNIST
        # grid_img = grid_img[:, :, :, denoised_sample - 1]
        # grid_img = grid_img[:, indices, :, :, denoised_sample - 1]  # comment out for flow ODE
        grid_img = torch.reshape(grid_img[:, :, None, :], (-1, 1, 28, 28))
        save_image(
            grid_img, fp=f"images/MNIST_{conf.num_samples}_{i}.png", nrow=len(indices)
        )

    # # CIFAR-10
    # grid_img = grid_img[:, indices, :, :, 3 * (denoised_sample - 1) : 3 * denoised_sample]
    # grid_img = torch.permute(grid_img, (0, 1, 4, 2, 3))
    # grid_img = torch.reshape(grid_img, (-1, 3, 32, 32))
    # save_image(grid_img, fp="CIFAR10.png", nrow=len(indices))
