import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import CappedGELU
from einops import rearrange


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
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = "circular"
        for n in range(n_layers):
            convblock.append(
                torch.nn.Conv2d(
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            convblock.append(
                torch.nn.BatchNorm2d(
                    out_channels if n == n_layers - 1 else latent_channels
                )
            )
            if activation is not None:
                convblock.append(activation)
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        # return self.convblock(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d):
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return x


class ConvNeXtBlockOldSwin(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper. Use of dilations here.

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
        assert n_layers == 1, "Can only use a single layer here!"
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = "circular"
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
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # LayerNorm
        # convblock.append(LayerNorm(latent_channels*upscale_factor, eps=1e-6, data_format="channels_first"))
        convblock.append(torch.nn.BatchNorm2d(latent_channels * upscale_factor))
        if activation is not None:
            convblock.append(activation)
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        # LayerNorm
        # convblock.append(LayerNorm(latent_channels*upscale_factor, eps=1e-6, data_format="channels_first"))
        convblock.append(torch.nn.BatchNorm2d(latent_channels * upscale_factor))
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
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return skip + x


class TransposedConvUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: torch.nn.Module = torch.nn.ReLU,
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
            )
        )

        if activation is not None:
            upsampler.append(activation)
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)

class BilinearUpsample3D(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=(1, upsampling, upsampling), mode="trilinear")

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
    
class AvgPool3D(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool3d((1, pooling, pooling))

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


class CoreBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, pad):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad

    def forward(self):
        raise NotImplementedError()


class ConvBlock(CoreBlock):

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU,
        pad="circular",
    ):

        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)

        layers = []
        layers.append(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(activation)
        for _ in range(n_layers - 1):
            layers.append(
                torch.nn.Conv2d(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(activation)

        self.layers = nn.ModuleList(layers)
        # self.layers = nn.ModuleList(layer)

    def forward(self, fts):
        for l in self.layers:
            if isinstance(l, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            fts = l(fts)
        return fts


class ConvNeXtBlock(CoreBlock):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper. Use of dilations here.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU,
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
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
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
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
            convblock.append(activation)
        convblock.append(
            torch.nn.Conv2d(
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
            convblock.append(activation)
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

    def forward(self, x):
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return skip + x


class ConvNeXtBlock3D(CoreBlock):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper. Use of dilations here.

    """

    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU,
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
        assert n_layers == 1, "Can only use a single layer here!"

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=(1, kernel_size, kernel_size),
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm3d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm3d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation)
        convblock.append(
            torch.nn.Conv3d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=(1, kernel_size, kernel_size),
                dilation=dilation,
            )
        )
        # BatchNorm
        if norm == "batch":
            convblock.append(torch.nn.BatchNorm3d(in_channels * upscale_factor))
        # Instance Norm
        elif norm == "instance":
            convblock.append(torch.nn.InstanceNorm3d(in_channels * upscale_factor))
        elif norm == "nonorm":
            pass
        else:
            raise NotImplementedError
        if activation is not None:
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv3d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv3d) and l.kernel_size[-1] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad, 0, 0), mode="constant"
                )
            x = l(x)
        return skip + x


class ConvNeXtTemporalBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int = 300,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = torch.nn.ReLU,
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
        assert n_layers == 1, "Can only use a single layer here!"

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv2d(
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
            convblock.append(activation)
        convblock.append(
            torch.nn.Conv2d(
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
            convblock.append(activation)
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        # return self.skip_module(x) + self.convblock(x)
        N,C,T,H,W = x.shape
        x = rearrange(x, 'n c t h w -> n (c t) h w')
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[-1] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        x = skip + x
        x = rearrange(x, 'n (c t) h w -> n c t h w', t=T)
        return x


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
