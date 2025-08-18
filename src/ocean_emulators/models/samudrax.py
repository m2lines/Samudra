import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from ocean_emulators.constants import Grid


class Samudrax(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d

    def __init__(
        self,
        input_channels: int = 162,
        output_channels: int = 154,
        hidden_dim: int = 512,
        *,
        key,
    ):
        keys = jax.random.split(key, 3)
        # Use 1x1 convolutions for channel-wise transformations
        self.conv1 = eqx.nn.Conv2d(
            input_channels, hidden_dim, kernel_size=1, key=keys[0]
        )
        self.conv2 = eqx.nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, key=keys[1])
        self.conv3 = eqx.nn.Conv2d(
            hidden_dim, output_channels, kernel_size=1, key=keys[2]
        )

    def __call__(
        self, x: Float[Grid, " prognostic_vars+boundary_vars"]
    ) -> Float[Grid, " prognostic_vars"]:
        # x shape: (prognostic_vars+boundary_vars, lat, lon)
        # Apply sequence of 1x1 convolutions
        x = jnp.tanh(self.conv1(x))
        x = jnp.tanh(self.conv2(x))
        x = self.conv3(x)

        return x
