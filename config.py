from jax import random
from typing import NamedTuple, List
from jax import numpy as jnp


# Changed this from ml_collections since ml_collections
# gave a load of errors on my type checker (pyright)


class Config(NamedTuple):
    data_dim: int
    num_samples: int
    n_data: int
    layer_sizes: List[int]
    embed_dim: int
    key: jnp.ndarray
    batch_size: int
    sigma: int
    num_epochs: int
    lr: float
    langevin_iterations: int
    langevin_burnin: int
    langevin_stepsize: float


def get_config():
    num_samples: int = 1
    data_dim: int = 2

    return Config(
        data_dim=data_dim,
        num_samples=num_samples,
        n_data=10_000,
        layer_sizes=[32, 32, 32, num_samples * data_dim],
        embed_dim=8,
        key=random.PRNGKey(1),
        batch_size=256,
        sigma=25,
        num_epochs=50_000,
        lr=1e-3,
        langevin_iterations=100_000,
        langevin_burnin=80_000,
        langevin_stepsize=1e-4,
    )
