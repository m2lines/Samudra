import torch
from .activations import CappedGELU
from typing import Sequence

class TransposedConvUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
            )  # check padding
        )

        if activation is not None:
            upsampler.append(activation)
        self.upsampler = torch.nn.Sequential(*upsampler)

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


class BasicConvBlock(torch.nn.Module):
    """
    Convolution block consisting of n subsequent convolutions and activations
    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        latent_channels: int = None,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                torch.nn.Conv2d(
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding="same",
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        return self.convblock(x)


class ConvGRUBlock(torch.nn.Module):
    """
    Code modified from
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 1,
        downscale_factor: int = 4,
    ):
        super().__init__()

        self.channels = in_channels
        self.conv_gates = torch.nn.Conv2d(
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv_can = torch.nn.Conv2d(
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
        )
        self.h = torch.zeros(1, 1, 1, 1)

    def forward(self, inputs: Sequence) -> Sequence:
        if inputs.shape != self.h.shape:
            self.h = torch.zeros_like(inputs)
        combined = torch.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        self.h = torch.zeros_like(self.h)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in Figure 4 of https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU(),
    ):
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        # Convolution block
        convblock = []
        # 7x7 convolution increasing channels
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        )
        # LayerNorm
        # convblock.append(th.nn.LayerNorm([out_channels*upscale_factor, HW, HW]))
        if activation is not None:
            convblock.append(activation)
        # 1x1 convolution decreasing channels
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            )
        )
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        return self.skip_module(x) + self.convblock(x)
