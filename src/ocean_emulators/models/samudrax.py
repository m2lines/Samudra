import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from ocean_emulators.config import SamudraConfig
from ocean_emulators.constants import Grid


class CappedGELU(eqx.Module):
    def __init__(self, cap_value=10.0):
        self.cap_value = cap_value

    def __call__(self, x: ArrayLike) -> Array:
        x = jax.nn.gelu(x)
        x = jax.lax.clamp(x, x, max=self.cap_value)
        return x


class ConvNeXtBlock(eqx.Module):
    skip_module: eqx.Module
    layers: list[eqx.Module]

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: type[eqx.Module] | None = CappedGELU,
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
        *,
        key,
    ):
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        keys = jax.random.split(key, 3)
        k = 0

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad

        if in_channels == out_channels:
            self.skip_module = eqx.nn.Identity()
        else:
            # 1x1 Conv
            self.skip_module = eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="REPLICATE",  # TODO(alxmrs): Are these good padding vals?
                key=keys[k],
            )
        k += 1

        # CNN Block
        self.layers = []
        self.layers.append(
            eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                key=keys[k],
            )
        )
        k += 1

        match norm:
            case "batch":
                self.layers.append(
                    eqx.nn.BatchNorm(in_channels * upscale_factor, 0)
                )  # TODO(alxmrs): Is this the right axis??
            case "instance":
                raise NotImplementedError("No instance norm! Sorry!")
            case "nonorm":
                pass
            case _:
                typing.assert_never(norm)

        if activation is not None:
            self.layers.append(activation())

        # Linear post-processing -- 1x1 CNN
        self.layers.append(
            eqx.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="REPLICATE",
                key=keys[k],
            )
        )
        k += 1

    def __call__(self, x: ArrayLike) -> ArrayLike:
        skip = self.skip_module(x)
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d) and layer.kernel_size[0] != 1:
                # TODO(alxmrs): Verify padding is the same
                # Circular wrap (longitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (0, 0), (0, 0), (self.N_pad, self.N_pad)),
                    mode="wrap",
                )
                # Reflect around the poles (latitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (0, 0), (self.N_pad, self.N_pad), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            x = layer(x)
        return skip + x


# TODO(alxmrs): Implement checkpointing
class Samudrax(eqx.Module):
    def __init__(self, config: SamudraConfig, *, key):
        # make downscale blocks

        # middle block

        # make upscale blocks

        pass

    def __call__(self, x: Float[Grid, " channels"]) -> Float[Grid, " channels"]:
        # downscale blocks

        # middle

        # upscale blocks

        return x
