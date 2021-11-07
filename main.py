import jax
import jax.numpy as jnp
from jax import random
from langevin import chain_langevin
from config import get_config
import matplotlib.pyplot as plt 
from train import f, train_wrapper
from makedata import make_data


conf = get_config()

num_chains = 10

# MAKE DATASET
x = make_data()


# INITIALIZATION OF PARAMS
dummy_xs, dummy_std = jnp.zeros((conf.batch_size, conf.num_samples*conf.data_dim)), jnp.zeros((conf.batch_size,1))
params, state = f.init(random.PRNGKey(1), dummy_xs, dummy_std, conf.sigma, True)
out, state= f.apply(params,state, dummy_xs, dummy_std, conf.sigma, True)


#MODEL TRAINING
train_loss, params, state = train_wrapper(params, state, conf.num_epochs, x)
print("Model training done\n")

# LANGEVIN CHAIN
init_x = jax.random.uniform(conf.key,(num_chains,conf.data_dim*conf.num_samples),minval=-10., maxval=10.)
key_array = jax.random.split(random.PRNGKey(10), init_x.shape[0])
out = chain_langevin(params, state, key_array, init_x, jnp.flip(jnp.array([0.1,1.,5.,10.])), 
conf.langevin_stepsize, conf.langevin_iterations, conf.langevin_burnin)

# PLOTTING
fig,ax = plt.subplots()
for i,x in enumerate(out):
    ax.scatter(x[:,0],x[:,1],label=i)

plt.legend(loc='upper left')
plt.savefig("trial.png")

