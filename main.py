import jax.numpy as jnp

# from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt
from train import f, train_wrapper

# from data import get_gaussian_mixture
from metrics import eval
import jax.random as rnd
from jax import jit
import pdb

# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
import tensorflow_datasets as tfds
import tensorflow as tf

conf = get_config()

num_chains = 10
noise_scales = jnp.array([10.0, 5.0, 1.0, 0.1])

# Make Dataset
# data_distribution = get_gaussian_mixture(
#     means=[-4.0 * jnp.ones(conf.data_dim), 4.0 * jnp.ones(conf.data_dim)],
#     std=1.0,
#     weights=[0.8, 0.2],
# )
# x = jit(data_distribution.sample, static_argnames=("sample_shape"))(
#     seed=rnd.PRNGKey(0), sample_shape=(conf.n_data,)
# )

ds = tfds.load("mnist", split="train", shuffle_files=True)
ds = ds.shuffle(60000).batch(60000).prefetch(tf.data.AUTOTUNE)
for example in ds.take(1):
    images, labels = example["image"], example["label"]

plt.imshow(images[5].numpy().squeeze(), cmap="gray")
x = images.numpy()

# Initialization of Params
# pdb.set_trace()
# dummy_xs, dummy_t = jnp.ones(
#     (conf.batch_size, conf.data_dim * conf.num_samples)
# ), jnp.ones(conf.batch_size)
dummy_xs, dummy_t = jnp.ones((conf.batch_size, 28, 28, conf.num_samples)), jnp.zeros(
    (conf.batch_size, 1, 1, conf.num_samples)
)
params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_t, conf.sigma)
out, state = f.apply(params, state, dummy_xs, dummy_t, conf.sigma)
# pdb.set_trace()

# Model Training
fullloss, train_loss, params, state = train_wrapper(params, state, conf.num_epochs, x)
print("Model training done\n")
print(f"Loss over entire dataset: {fullloss}")

# Evaluation
# print(
#     f"Mean difference b/w Ground Truth Score and Score Model: {eval(params, state, x)}"
# )

# Langevin Chain
init_x = rnd.uniform(conf.key, (num_chains, conf.data_dim), minval=-10.0, maxval=10.0)
init_x = jnp.concatenate([init_x] * conf.num_samples, axis=-1)
init_x = init_x + noise_scales[0] * rnd.normal(
    rnd.PRNGKey(3), (num_chains, conf.num_samples * conf.data_dim)
)

# key_array = rnd.split(rnd.PRNGKey(10), init_x.shape[0])
# out = chain_langevin(
#     params,
#     state,
#     key_array,
#     init_x,
#     noise_scales,
#     conf.langevin_stepsize,
#     conf.langevin_iterations,
#     conf.langevin_burnin,
# )

# # Plotting
# fig, ax = plt.subplots()
# for i, x in enumerate(out):
#     ax.scatter(x[:, 0], x[:, 1], label=i)

# plt.legend(loc="upper left")
# plt.savefig("trial.png")
