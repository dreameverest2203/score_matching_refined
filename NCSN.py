import jax
import jax.numpy as jnp
import haiku as hk


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """      
    return jnp.sqrt((sigma**(2 * t) - 1.) / 2. / jnp.log(sigma))



class NCSN(hk.Module):
    def __init__(self,layer_sizes, embed_dim):
        super().__init__(name=None)
        self.layer_sizes = layer_sizes
        self.embed_dim = embed_dim

    def __call__(self, x, t, sigma, is_training=True):
        
        W = hk.get_parameter('W', (1, self.embed_dim // 2),init=hk.initializers.RandomNormal(sigma))
        W = jax.lax.stop_gradient(W)
        t_proj = t * W * 2 * jnp.pi
        t_proj = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        std_embedding3 = hk.Linear(self.embed_dim)(t_proj)

        x = jnp.concatenate([x,std_embedding3],axis=-1)
        x = hk.Linear(self.layer_sizes[0])(x)
        x = jax.nn.swish(x)
        x = hk.Linear(self.layer_sizes[1])(x)
        x = jax.nn.swish(x)
        x = hk.Linear(self.layer_sizes[2])(x)
        x = jax.nn.swish(x)
        x = hk.Linear(self.layer_sizes[3])(x)
        return x/marginal_prob_std(t,sigma)
        # return x

