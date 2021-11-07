import tensorflow_probability.substrates.jax.distributions as tfd
from config import get_config
from jax import random, jit

conf = get_config()

def make_data():
    gm = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=[0.5, 0.5]), 
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-2.0] * conf.data_dim, 
             [2.0] * conf.data_dim], 
        scale_identity_multiplier=[1.0] * conf.data_dim)) 

    x = jit(gm.sample, static_argnames=("sample_shape"))(seed=random.PRNGKey(0), sample_shape=(conf.n_data,))

    return x
