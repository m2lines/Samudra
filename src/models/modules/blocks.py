import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import CappedGELU
from typing import Sequence
from timm.models.layers import DropPath


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
            )
        )

        if activation is not None:
            upsampler.append(activation)
        self.upsampler = torch.nn.Sequential(*upsampler)

    def forward(self, x):
        return self.upsampler(x)


class BilinearUpsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3, # ignored
        out_channels: int = 1, # ignored
        upsampling: int = 2,
        activation: torch.nn.Module = None, # ignored
    ):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode='bilinear')

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


class AdamConvBlock(torch.nn.Module):

    def __init__(self, num_in = 2, num_out = 2,kernel_size = 3, num_layers=2, pad = "constant"):
        super().__init__()
        self.N_in = num_in
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad

        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_out,kernel_size))
        layers.append(torch.nn.BatchNorm2d(num_out))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(torch.nn.Conv2d(num_out,num_out,kernel_size))
            layers.append(torch.nn.BatchNorm2d(num_out))
            layers.append(torch.nn.ReLU())

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
        return fts


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
        assert kernel_size % 2 !=0, "Cannot use even kernel sizes!"
        
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        self.N_pad = int((kernel_size + (kernel_size-1)*(dilation-1) -1) / 2)
        self.pad = "circular"
        for n in range(n_layers):
            convblock.append(
                torch.nn.Conv2d(
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )
            convblock.append(torch.nn.BatchNorm2d(out_channels if n == n_layers - 1 else latent_channels))
            if activation is not None:
                # convblock.append(activation)
                convblock.append(torch.nn.ReLU())
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        # return self.convblock(x)
        for l in self.convblock:
            if isinstance(l,nn.Conv2d):
                x = torch.nn.functional.pad(x,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                x = torch.nn.functional.pad(x,(0,0,self.N_pad,self.N_pad),mode="constant")
            x= l(x)
        return x



class ConvNeXtBlock(torch.nn.Module):
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
        assert kernel_size % 2 !=0, "Cannot use even kernel sizes!"
        self.N_pad = int((kernel_size + (kernel_size-1)*(dilation-1) -1) / 2)
        self.pad = "circular"
        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"
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
        convblock.append(torch.nn.BatchNorm2d(latent_channels*upscale_factor))
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
        convblock.append(torch.nn.BatchNorm2d(latent_channels*upscale_factor))
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
            if isinstance(l, nn.Conv2d) and l.kernel_size[0]!=1:
                x = torch.nn.functional.pad(x,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                x = torch.nn.functional.pad(x,(0,0,self.N_pad,self.N_pad),mode="constant")
            x = l(x)
        return skip + x


class ConvNeXtBlockOrig(torch.nn.Module):
    """
    Actual implementation of ConvNeXt. No dilations here.

    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upscale_factor: int = 4,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        latent_channels=0,  # ignored
        dilation=1,
        n_layers=0,  # ignored
    ):
        super().__init__()
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            )
        
        self.N_pad = int((7 + (7-1)*(dilation-1) -1) / 2)
        self.pad = "circular"
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            groups=in_channels,
            dilation=dilation,
        )  # depthwise conv
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(
            in_channels, upscale_factor * in_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(upscale_factor * in_channels, out_channels)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = torch.nn.functional.pad(x,(self.N_pad,self.N_pad,0,0),mode=self.pad)
        x = torch.nn.functional.pad(x,(0,0,self.N_pad,self.N_pad),mode="constant")
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = self.skip_module(input) + self.drop_path(x)
        return x


class ConvNeXtBlockOrig2(torch.nn.Module):
    """
    Actual implementation of ConvNeXt. No dilations here.
    More layers and larger channels compared to ConvNeXtBlockOrig

    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upscale_factor: int = 4,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        latent_channels=0,  # ignored
        dilation=1,
        n_layers=0,  # ignored
    ):
        super().__init__()
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same"
            )
        self.pad = "circular"
        self.first_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=3,
            dilation=dilation,
        )
        self.N_pad1 = int((3 + (3-1)*(dilation-1) -1) / 2)

        self.dwconv = nn.Conv2d(
            in_channels * 2,
            in_channels * 2,
            kernel_size=7,
            groups=in_channels * 2,
            dilation=dilation,
        )  # depthwise conv
        self.N_pad2 = int((7 + (7-1)*(dilation-1) -1) / 2)
        self.norm = LayerNorm(in_channels * 2, eps=1e-6)
        self.pwconv1 = nn.Linear(
            in_channels * 2, upscale_factor * in_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(upscale_factor * in_channels, out_channels)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = torch.nn.functional.pad(x,(self.N_pad1,self.N_pad1,0,0),mode=self.pad)
        x = torch.nn.functional.pad(x,(0,0,self.N_pad1,self.N_pad1),mode="constant")
        x = self.first_conv(x)
        x = torch.nn.functional.pad(x,(self.N_pad2,self.N_pad2,0,0),mode=self.pad)
        x = torch.nn.functional.pad(x,(0,0,self.N_pad2,self.N_pad2),mode="constant")
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = self.skip_module(input) + self.drop_path(x)
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
