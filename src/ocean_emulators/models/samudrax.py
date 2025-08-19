import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from ocean_emulators.config import SamudraConfig
from ocean_emulators.constants import Grid
from ocean_emulators.utils.train import pairwise


class CappedGELU(eqx.Module):
    cap_value: float = eqx.field(static=True)

    def __init__(self, cap_value=10.0):
        self.cap_value = cap_value

    def __call__(self, x: ArrayLike) -> Array:
        x = jax.nn.gelu(x)
        x = jax.lax.clamp(x, x, max=self.cap_value)
        return x


class BilinearUpsample(eqx.Module):
    upsampling: int = eqx.field(static=True)

    def __init__(self, upsampling: int = 2):
        self.upsampling = upsampling

    def __call__(
        self, x: Float[Array, "channels lat lon"]
    ) -> Float[Array, "channels lat lon"]:
        return jax.image.resize(
            x,
            (x.shape[0], x.shape[1] * self.upsampling, x.shape[2] * self.upsampling),
            "bilinear",
        )


class AvgPool(eqx.Module):
    avgpool: eqx.nn.AvgPool2d = eqx.field(static=True)

    def __init__(
        self,
        pooling: int = 2,
    ):
        self.avgpool = eqx.nn.AvgPool2d(kernel_size=pooling, stride=pooling)

    def __call__(
        self, x: Float[Array, "channels lat lon"]
    ) -> Float[Array, "channels lat lon"]:
        return self.avgpool(x)


class ConvNeXtBlock(eqx.Module):
    skip_module: eqx.Module
    layers: list[eqx.Module]

    N_in: int = eqx.field(static=True)
    N_pad: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: type[eqx.Module] | None = CappedGELU,
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

        if in_channels == out_channels:
            self.skip_module = eqx.nn.Identity()
        else:
            # 1x1 Conv
            self.skip_module = eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="SAME",
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
                    eqx.nn.BatchNorm(in_channels * upscale_factor, axis_name="batch")
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
                padding="SAME",
                key=keys[k],
            )
        )
        k += 1

    def __call__(self, x: ArrayLike, state) -> tuple[ArrayLike, typing.Any]:
        skip = self.skip_module(x)
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d) and layer.kernel_size[0] != 1:
                # TODO(alxmrs): Verify padding is the same
                # Circular wrap (longitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (0, 0), (self.N_pad, self.N_pad)),
                    mode="wrap",
                )
                # Reflect around the poles (latitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (self.N_pad, self.N_pad), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state=state)
            else:
                x = layer(x)
        return skip + x, state


# TODO(alxmrs): Implement checkpointing
class Samudrax(eqx.Module):
    layers: list[eqx.Module]
    wet: ArrayLike
    block_depth: int = eqx.field(static=True)
    N_pad: int = eqx.field(static=True)

    def __init__(self, config: SamudraConfig, wet, *, key):
        ch_width = config.ch_width.copy()
        dilation = config.dilation.copy()
        self.wet = wet
        self.layers = []
        self.N_pad = int((config.last_kernel_size - 1) / 2)

        # make downscale blocks
        for i, (in_ch, out_ch) in enumerate(pairwise(ch_width)):
            cur_key, key = jax.random.split(key, 2)
            self.layers.append(
                ConvNeXtBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    activation=CappedGELU,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                    key=cur_key,
                )
            )
            self.layers.append(AvgPool())

        cur_key, key = jax.random.split(key, 2)
        # middle block
        self.layers.append(
            ConvNeXtBlock(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                activation=CappedGELU,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
                key=cur_key,
            )
        )

        self.layers.append(BilinearUpsample())

        # Reverse for upsampling path
        ch_width.reverse()
        dilation.reverse()

        # make upscale blocks
        for i, (in_ch, out_ch) in enumerate(pairwise(ch_width[:-1])):
            cur_key, key = jax.random.split(key, 2)
            self.layers.append(
                ConvNeXtBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.core_block.kernel_size,
                    dilation=dilation[i],
                    activation=CappedGELU,
                    upscale_factor=config.core_block.upscale_factor,
                    norm=config.core_block.norm,
                    key=cur_key,
                )
            )
            self.layers.append(BilinearUpsample())

        cur_key, key = jax.random.split(key, 2)
        # Final ConvBlock
        self.layers.append(
            ConvNeXtBlock(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=config.core_block.kernel_size,
                dilation=dilation[i],
                activation=CappedGELU,
                upscale_factor=config.core_block.upscale_factor,
                norm=config.core_block.norm,
                key=cur_key,
            )
        )

        cur_key, key = jax.random.split(key, 2)
        self.layers.append(
            eqx.nn.Conv2d(
                in_channels=out_ch,
                out_channels=config.n_out,
                kernel_size=config.last_kernel_size,
                key=cur_key,
            )
        )
        self.block_depth = len(config.ch_width) - 1

    def __call__(
        self, x: Float[Grid, " channels"], state
    ) -> tuple[Float[Grid, " channels"], typing.Any]:
        skips = []
        count = 0
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                # TODO(alxmrs): Verify padding is the same
                # Circular wrap (longitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (0, 0), (self.N_pad, self.N_pad)),
                    mode="wrap",
                )
                # Reflect around the poles (latitude)
                x = jnp.pad(
                    x,
                    pad_width=((0, 0), (self.N_pad, self.N_pad), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )

            if isinstance(layer, ConvNeXtBlock):
                x, state = layer(x, state=state)
            else:
                x = layer(x)

            if count < self.block_depth:
                if isinstance(layer, ConvNeXtBlock):
                    skips.append(x)
                    count += 1
                    print(f"x.shape: {x.shape}, count: {count}")
            elif count >= self.block_depth:
                if isinstance(layer, BilinearUpsample):
                    crop = np.array(x.shape[1:])
                    shape = np.array(
                        skips[int(2 * self.block_depth - count - 1)].shape[1:]
                    )
                    pads = shape - crop
                    print(
                        f"x.shape: {x.shape}, count: {count}, index: {2 * self.block_depth - count - 1}"
                    )
                    print(f"crop: {crop}, shape: {shape}, pads: {pads}")
                    # PyTorch: pads = [pads[1]//2, pads[1]-pads[1]//2, pads[0]//2, pads[0]-pads[0]//2]
                    pad_lat_before = pads[0] // 2
                    pad_lat_after = pads[0] - pad_lat_before
                    pad_lon_before = pads[1] // 2
                    pad_lon_after = pads[1] - pad_lon_before

                    x = jnp.pad(
                        x,
                        pad_width=(
                            (0, 0),  # channels
                            (pad_lat_before, pad_lat_after),  # latitude
                            (pad_lon_before, pad_lon_after),  # longitude
                        ),
                        mode="constant",
                        constant_values=0.0,
                    )
                    x += skips[int(2 * self.block_depth - count - 1)]
                    count += 1
        # TODO(alxmrs): Do we want a corrector??
        return jnp.where(self.wet, x, 0.0), state
