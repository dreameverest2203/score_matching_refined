from ode_sampler import ode_sampler
import pickle
import torch
from torchvision.transforms import RandAugment
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import pdb
import matplotlib.pyplot as plt
import tqdm
import warnings
import wandb

warnings.filterwarnings("ignore")

torch.manual_seed(8)


def augmentation(x):
    x = torch.tensor(255 * x, dtype=torch.uint8)
    x = torch.permute(x, (0, 3, 1, 2))
    aux = RandAugment(3, 5, 31).forward(x)
    aug_x = torch.tensor(aux / 255.0, dtype=torch.float16)
    return aug_x


@hydra.main(config_path=".", config_name="debug_config")
def main(cfg: DictConfig) -> None:
    if cfg.use_wandb:
        wandb.init(project="ncsnpp", config=cfg)
    num_points = 2
    data_shape = (-1, 28, 28, 1)

    train_dataset = MNIST(
        "../../..", train=True, transform=transforms.ToTensor(), download=True
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=num_points, shuffle=True, num_workers=2
    )

    _, _, f = get_model(cfg)

    with open(
        f"../../../model_weights/dae_model_{cfg.num_samples}samples.pickle", "rb"
    ) as handle:
        params, state = pickle.load(handle)

    # for j, (x, _) in enumerate(train_data_loader):

    #     x0 = x.permute(0, 2, 3, 1).numpy().reshape(data_shape).astype(np.float32)
    #     aug0 = augmentation(x0).float().numpy()
    #     augT = ode_sampler(f, params, state, aug0, 25.0, 1e-3, chain_length=num_points)
    #     xT = ode_sampler(f, params, state, x0, 25.0, 1e-3, chain_length=num_points)

    # x0 = np.reshape(x0, (x0.shape[0], -1))
    # xT = np.reshape(xT, (xT.shape[0], -1))
    # aug0 = np.reshape(aug0, (aug0.shape[0], -1))
    # augT = np.reshape(augT, (augT.shape[0], -1))

    train_data_loader2 = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=2
    )
    num_comparisons, all_distorg, all_distaug = 0, 0, 0
    tqdm_data = tqdm.tqdm(train_data_loader2)
    for x, _ in tqdm_data:
        # for j, (x, _) in enumerate(train_data_loader2):
        x0 = x.permute(0, 2, 3, 1).numpy().reshape(data_shape).astype(np.float32)
        aug0 = augmentation(x0).float().numpy()
        augT = ode_sampler(f, params, state, aug0, 25.0, 1e-3, chain_length=num_points)
        xT = ode_sampler(f, params, state, x0, 25.0, 1e-3, chain_length=num_points)
        distorg = np.linalg.norm(xT[0] - xT[1])
        distaug = np.linalg.norm(augT[0] - augT[1])
        all_distorg += distorg
        all_distaug += distaug
        num_comparisons += 1

        wandb.log(
            data={
                "Mean distance between x(T) and y(T)": float(
                    all_distorg / num_comparisons
                )
            }
        )
        wandb.log(
            data={
                "Mean distance between xaug(0) and yaug(T)": float(
                    all_distaug / num_comparisons
                )
            }
        )
        tqdm_data.set_description(
            "Mean distance between x(T) and y(T): {:5f}".format(
                all_distorg / num_comparisons
            )
        )

        tqdm_data.set_description(
            "Mean distance between x_aug(T) and y_aug(T): {:5f}".format(
                all_distaug / num_comparisons
            )
        )


if __name__ == "__main__":
    main()
