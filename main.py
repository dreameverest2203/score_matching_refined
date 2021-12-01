import jax.numpy as jnp
from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt
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
from NCSN import marginal_prob_std
import pdb


conf = get_config()


dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)

data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
for i, l in data_loader:
    images, labels = i, l

x = images.permute(0, 2, 3, 1).numpy()


# # Load Saved Model
# with open(f"dae_model_{conf.num_samples}samples.pickle", "rb") as handle:
#     params, state = pickle.load(handle)


# Initialization of Params
dummy_xs, dummy_t = jnp.ones(
    (conf.batch_size, 32, 32, 3 * conf.num_samples)
), jnp.zeros((conf.batch_size,))

params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma, True)
out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma, True)
# # Model Training
fullloss, train_loss, params, state = train_wrapper(params, state, conf.num_epochs, x)
print("Model training done\n")
print(f"Loss over entire dataset: {fullloss}")

with open(f"dae_model_{conf.num_samples}samples.pickle", "wb") as handle:
    pickle.dump((params, state), handle)


# Langevin Chain
init_x = rnd.uniform(
    conf.key, (10, 32, 32, 3 * conf.num_samples), minval=0.0, maxval=1.0
)
# init_x = x[:10]
# init_x = jnp.concatenate([init_x] * conf.num_samples, axis=-1)
# z = rnd.normal(rnd.PRNGKey(23), init_x.shape)
# random_t = rnd.uniform(rnd.PRNGKey(23), (init_x.shape[0],), minval=0.1, maxval=0.5)
# std = marginal_prob_std(random_t, conf.sigma)
# std = std[:, None, None, None]
# init_x = init_x + std * z
key_array = rnd.split(rnd.PRNGKey(15), init_x.shape[0])

out = chain_langevin(
    params,
    state,
    key_array,
    init_x,
    jnp.flip(jnp.array([1e-3, 1e-1, 5e-1, 1])),
    1e-3,
    50,
    0,
)
out = jnp.clip(out, 0.0, 1.0)

chain_num = 3
denoised_sample = 1
grid_img = np.array(out)
grid_img = torch.from_numpy(grid_img)
indices = np.arange(0, 200, 40)
# # MNIST
# grid_img = grid_img[:, indices, :, :, denoised_sample - 1]
# grid_img = torch.reshape(grid_img[:, :, None, :], (-1, 3, 32, 32))
# save_image(grid_img, fp="CIFAR10.png", nrow=len(indices))

# CIFAR-10
grid_img = grid_img[:, indices, :, :, 3 * (denoised_sample - 1) : 3 * denoised_sample]
grid_img = torch.permute(grid_img, (0, 1, 4, 2, 3))
grid_img = torch.reshape(grid_img, (-1, 3, 32, 32))
save_image(grid_img, fp="CIFAR10.png", nrow=len(indices))
