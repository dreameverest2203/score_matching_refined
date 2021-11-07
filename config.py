import ml_collections
from jax import random

def get_config():
    config = ml_collections.ConfigDict()
    config.data_dim = 2
    config.num_samples = 4
    config.n_data = 10_000
    config.layer_sizes = [32, 32, config.num_samples*config.data_dim]
    config.embed_dim = 8
    config.key = random.PRNGKey(1)
    config.batch_size = 256
    config.sigma = 25
    config.num_epochs = 50_000
    config.lr = 1e-3
    config.langevin_iterations = 100_000
    config.langevin_burnin = 20_000
    config.langevin_stepsize = 1e-4
    return config
