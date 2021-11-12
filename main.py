import jax.numpy as jnp
from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt
from train import f, train_wrapper
from data import get_gaussian_mixture
from evaluate import eval
import jax.random as rnd
from jax import jit


conf = get_config()

num_chains = 10

# Make Dataset
data_distribution = get_gaussian_mixture(
    means=[-4.0 * jnp.ones(conf.data_dim), 4.0 * jnp.ones(conf.data_dim)],
    std=1.0,
    weights=[0.8, 0.2],
)
x = jit(data_distribution.sample, static_argnames=("sample_shape"))(
    seed=rnd.PRNGKey(0), sample_shape=(conf.n_data,)
)


# Initialization of Params
dummy_xs, dummy_std = (
    jnp.zeros((conf.batch_size, conf.num_samples * conf.data_dim)),
    jnp.zeros((conf.batch_size, 1)),
)
params, state = f.init(rnd.PRNGKey(1), dummy_xs, dummy_std, conf.sigma)
out, state = f.apply(params, state, dummy_xs, dummy_std, conf.sigma)


# Model Training
fullloss, train_loss, params, state = train_wrapper(params, state, conf.num_epochs, x)
print("Model training done\n")
print(f"Loss over entire dataset: {fullloss}")

# Evaluation
print(
    f"Mean difference b/w Ground Truth Score and Score Model: {eval(params, state, x)}"
)

# Langevin Chain
init_x = rnd.uniform(
    conf.key, (num_chains, conf.data_dim * conf.num_samples), minval=-10.0, maxval=10.0
)
key_array = rnd.split(rnd.PRNGKey(10), init_x.shape[0])
out = chain_langevin(
    params,
    state,
    key_array,
    init_x,
    jnp.array([10.0, 5.0, 1.0, 0.1]),
    conf.langevin_stepsize,
    conf.langevin_iterations,
    conf.langevin_burnin,
)

# Plotting
fig, ax = plt.subplots()
for i, x in enumerate(out):
    ax.scatter(x[:, 0], x[:, 1], label=i)

plt.legend(loc="upper left")
plt.savefig("trial.png")
