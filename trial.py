import pickle
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import jax.numpy as jnp
from config import get_config
import pdb
import numpy as np
import torch
import jax.random as rnd
from UNet import marginal_prob_std


conf = get_config()


with open("dae_model_8samples.pickle", "rb") as handle:
    params, state = pickle.load(handle)


dataset = CIFAR10(".", train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
for i, l in data_loader:
    images, labels = i, l

x = images.permute(0, 2, 3, 1).numpy()

ip_x = jnp.concatenate(conf.num_samples * [x[:1]], axis=-1)
z = rnd.normal(rnd.PRNGKey(23), ip_x.shape)
ip_t = 1e-2 * jnp.ones((1,))
std = marginal_prob_std(ip_t, conf.sigma)[:, None, None, None]
ip_x = ip_x + std * z
out, state = f.apply(params, state, ip_x, ip_t, conf.sigma, True)
print(
    f"Distance between original and denoised version: {jnp.linalg.norm(out-ip_x[:,:,:,3:6])}"
)

grid_img = np.array(out)
grid_img = torch.from_numpy(grid_img)
model_input = torch.from_numpy(np.array(ip_x[:, :, :, :3]))
original_img = torch.tensor(x[:1])
pdb.set_trace()
grid_img = torch.permute(
    torch.cat((original_img, model_input, grid_img), axis=0), (0, 3, 1, 2)
)
save_image(grid_img, fp="recons.png", nrow=3)
