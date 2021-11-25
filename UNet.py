import jax
import jax.numpy as jnp
import haiku as hk
from config import get_config

conf = get_config()


class DAE(hk.Module):
    def __init__(self):
        super().__init__(name=None)
        self.channels = conf.channels
        self.embed_dim = conf.embed_dim
        self.scale = conf.scale

    def __call__(self, x, t, sigma, is_training=True):
        # The swish activation function
        act = jax.nn.swish
        w = hk.get_parameter(
            "w", (self.embed_dim // 2,), init=hk.initializers.RandomNormal(self.scale)
        )
        w = jax.lax.stop_gradient(w)
        t_proj = t[:, None] * w[None, :] * 2 * jnp.pi
        t_proj = jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)
        std_embedding = hk.Linear(self.embed_dim)(t_proj)
        std_embedding = std_embedding[:, None, None, :]
        std_embedding = act(std_embedding)

        # Encoding path
        h1 = hk.Conv2D(
            self.channels[0], (3, 3), (1, 1), padding="VALID", with_bias=False
        )(x)
        ## Incorporate information from t
        h1 += hk.Linear(self.channels[0])(std_embedding)
        ## Group normalization
        h1 = hk.GroupNorm(4)(h1)
        h1 = act(h1)

        h2 = hk.Conv2D(
            self.channels[1], (3, 3), (2, 2), padding="VALID", with_bias=False
        )(h1)
        h2 += hk.Linear(self.channels[1])(std_embedding)
        h2 = hk.GroupNorm(32)(h2)
        h2 = act(h2)

        h3 = hk.Conv2D(
            self.channels[2], (3, 3), (2, 2), padding="VALID", with_bias=False
        )(h2)
        h3 += hk.Linear(self.channels[2])(std_embedding)
        h3 = hk.GroupNorm(32)(h3)
        h3 = act(h3)

        h4 = hk.Conv2D(
            self.channels[3], (3, 3), (2, 2), padding="VALID", with_bias=False
        )(h3)
        h4 += hk.Linear(self.channels[3])(std_embedding)
        h4 = hk.GroupNorm(32)(h4)
        h4 = act(h4)

        # Decoding path
        h = hk.Conv2DTranspose(
            self.channels[2], (3, 3), (2, 2), padding=((2, 2), (2, 2)), with_bias=False
        )(h4)
        ## Skip connection from the encoding path
        h += hk.Linear(self.channels[2])(std_embedding)
        h = hk.GroupNorm(32)(h)
        h = act(h)

        h = hk.Conv2DTranspose(
            self.channels[1], (3, 3), (2, 2), padding=((2, 3), (2, 3)), with_bias=False
        )(jnp.concatenate([h, h3], axis=-1))
        h += hk.Linear(self.channels[1])(std_embedding)
        h = hk.GroupNorm(32)(h)
        h = act(h)
        h = hk.Conv2DTranspose(
            self.channels[0], (3, 3), (2, 2), padding=((2, 3), (2, 3)), with_bias=False
        )(jnp.concatenate([h, h2], axis=-1))

        h += hk.Linear(self.channels[0])(std_embedding)
        h = hk.GroupNorm(32)(h)
        h = act(h)
        h = hk.Conv2DTranspose(1, (3, 3), (1, 1), padding=((2, 2), (2, 2)))(
            jnp.concatenate([h, h1], axis=-1)
        )

        return h


def forward_new(perturbed_x, t, sigma, is_training):
    denoiser = DAE()
    return denoiser(perturbed_x, t, sigma, is_training)


f = hk.without_apply_rng(hk.transform_with_state(forward_new))
