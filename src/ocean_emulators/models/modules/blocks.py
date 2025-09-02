from collections.abc import Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ocean_emulators.models.modules.activations import CappedGELU


class TransposedConvUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
    ):
        super().__init__()
        upsampler: list[torch.nn.Module] = []
        # Upsample transpose conv
        upsampler.append(
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
            )
        )

        if activation is not None:
            upsampler.append(activation())
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class MaxPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.maxpool(x)


class GlobePaddedConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, dilation, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Circular wrap for longitude
        x = torch.nn.functional.pad(x, (self.N_pad, self.N_pad, 0, 0), mode="circular")
        # Reflect around the poles for latitude
        x = torch.nn.functional.pad(x, (0, 0, self.N_pad, self.N_pad), mode="constant")
        x = self.conv2d(x)
        return x


class CoreBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class ConvBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        checkpoint_simple: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation)

        layers: list[torch.nn.Module] = []
        layers.append(
            GlobePaddedConv2d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(
                GlobePaddedConv2d(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(activation())

        self.layers = nn.ModuleList(layers)
        self.checkpoint_simple = checkpoint_simple

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # conv2d layers are expensive so we save their activations,
            # other (simple) layers are cheap, so we don't save their activations.
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)
            else:
                fts = layer(fts)
        return fts


class ConvNeXtBlock(CoreBlock):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix
    paper. Use of dilations here.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: Callable[[], torch.nn.Module] = CappedGELU,
        upscale_factor: int = 4,
        norm="batch",
        checkpoint_simple: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation)
        assert n_layers == 1, "Can only use a single layer here!"

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        # Convolution block
        convblock: list[torch.nn.Module] = []
        convblock.append(
            GlobePaddedConv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm2d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation())
        convblock.append(
            GlobePaddedConv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm2d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation())
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)
        self.checkpoint_simple = checkpoint_simple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for layer in self.convblock:
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return skip + x
