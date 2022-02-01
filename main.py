import jax.numpy as jnp
from langevin import langevin_wrapper
import matplotlib.pyplot as plt
from ode_sampler import ode_sampler
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
import hydra
from omegaconf import DictConfig, OmegaConf
from train import train_wrapper
from models import get_model
import pdb
from likelihood import likelihood_wrapper


@hydra.main(config_path=".", config_name="debug_config")
def main(cfg: DictConfig) -> None:

    train_dataset = MNIST(
        "../../..", train=True, transform=transforms.ToTensor(), download=False
    )
    # idx = train_dataset.targets==2
    # train_dataset = train_dataset.data[idx]
    train_data_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    test_dataset = MNIST(
        "../../..", train=False, transform=transforms.ToTensor(), download=False
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2
    )

    if cfg.load_model:
        with open(
            f"../../../model_weights/dae_model_{cfg.num_samples}samples.pickle", "rb"
        ) as handle:
            params, state = pickle.load(handle)
    else:
        train_loss, params, state = train_wrapper(
            train_data_loader, test_data_loader, cfg
        )
        print("Model training done\n")

        with open(
            f"../../../model_weights/dae_model_{cfg.num_samples}samples.pickle", "wb"
        ) as handle:
            pickle.dump((params, state), handle)

    _, _, f = get_model(cfg)

    # Langevin Chain
    if cfg.langevin:
        init_x = rnd.uniform(
            rnd.PRNGKey(4),
            (cfg.chain_length, 28, 28, cfg.num_samples),
            minval=0.0,
            maxval=1.0,
        )
        key_array = rnd.split(rnd.PRNGKey(15), init_x.shape[0])
        out = langevin_wrapper(
            f,
            params,
            state,
            key_array,
            init_x,
            jnp.flip(jnp.array([1e-3, 5e-3, 25e-3, 125e-3, 625e-3, 1])),
        )
        out = jnp.clip(out, 0.0, 1.0)
        grid_img = np.array(out)
        grid_img = torch.from_numpy(grid_img)
        pdb.set_trace()

    if cfg.plot_points:
        for i in range(10):
            init_x = rnd.uniform(
                rnd.PRNGKey(i), (1, 28, 28, cfg.num_samples), minval=0.0, maxval=1.0
            )
            out = ode_sampler(f, params, state, init_x, cfg.sigma, 1e-3, chain_length=1)
            out = jnp.clip(out, 0.0, 1.0)
            grid_img = np.array(out)
            grid_img = torch.from_numpy(grid_img)
            grid_img = torch.permute(grid_img, (0, 3, 1, 2))
            save_image(
                grid_img,
                fp=f"../../../images/MNIST_{cfg.num_samples}_{i}.png",
                nrow=grid_img.shape[0],
            )
    if cfg.calculate_likelihood:
        bpd = likelihood_wrapper(f, cfg, params, state)
        print(f"Final bpd: {bpd}")


if __name__ == "__main__":
    main()
