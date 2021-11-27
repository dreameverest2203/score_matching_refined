import jax.numpy as jnp
from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt
from train import f, train_wrapper
import numpy as np
import jax.random as rnd
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pickle
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import pdb
import warnings
import tqdm

warnings.filterwarnings("ignore")


conf = get_config()


dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)

data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
for i, l in data_loader:
    images, labels = i, l

x = images.permute(0, 2, 3, 1).numpy()

# Load Saved Model
with open("dae_model_8samples.pickle", "rb") as handle:
    params, state = pickle.load(handle)


# # Initialization of Params
# dummy_xs, dummy_t = jnp.ones((conf.batch_size, 28, 28, conf.num_samples)), jnp.zeros(
#     (conf.batch_size, 1, 1, conf.num_samples)
# )


# params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma, True)
# out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma, True)

# # # Model Training
# fullloss, train_loss, params, state = train_wrapper(params, state, conf.num_epochs, x)
# print("Model training done\n")
# print(f"Loss over entire dataset: {fullloss}")

with open("dae_model_8samples.pickle", "wb") as handle:
    pickle.dump((params, state), handle)


# Langevin Chain
init_x = rnd.uniform(conf.key, (3, 28, 28, conf.num_samples), minval=0.0, maxval=1.0)
key_array = rnd.split(rnd.PRNGKey(15), init_x.shape[0])
key_array = rnd.split(rnd.PRNGKey(15), init_x.shape[0])

out = chain_langevin(
    params,
    state,
    key_array,
    init_x,
    jnp.flip(jnp.array([1e-3, 1e-1, 5e-1, 1])),
    1e-4,
    80,
    0,
)
out = jnp.clip(out, 0.0, 1.0)


chain_num = 3
denoised_sample = 1

grid_img = np.array(out)
grid_img = torch.from_numpy(grid_img)
indices = np.arange(0, 80, 10)
grid_img = grid_img[:, indices, :, :, denoised_sample - 1]
grid_img = torch.reshape(grid_img[:, :, None, :], (-1, 1, 28, 28))
save_image(grid_img, fp="MNIST.png", nrow=len(indices))


# grid_img = np.array(out)
# grid_img = torch.from_numpy(grid_img)
# indices = [
#     0,
#     5,
#     10,
#     15,
#     20,
#     25,
#     30,
#     35,
#     40,
#     45,
# ]  # , 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
# grid_img = grid_img[chain_num - 3, indices, :, :, denoised_sample - 1]
# grid_img = grid_img[:, None, :, :]
# grid_img = make_grid(grid_img, nrow=10)
# plt.rcParams["figure.figsize"] = [25, 5]
# plt.imshow(grid_img.permute(1, 2, 0), cmap="gray")
# plt.show()
# plt.savefig("anuj.png")
