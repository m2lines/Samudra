from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
import torch.utils.checkpoint
from jaxtyping import Float

from ocean_emulators.models.modules.activations import CappedGELU


class PointwiseLinear(torch.nn.Module):
    """A 1×1 convolution implemented as nn.Linear.

    Mathematically equivalent to Conv2d(kernel_size=1), but avoids the
    non-contiguous gradient strides that 1×1 convs produce, which cause
    DDP to copy gradients instead of using zero-copy views.

    This optimization is use in the official ConvNext implementation[0].

    [0]: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L18
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(
        self, x: Float[torch.Tensor, "B C_in H W"]
    ) -> Float[torch.Tensor, "B C_out H W"]:
        return self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


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


class ZonallyPeriodicBilinearUpsample(torch.nn.Module):
    """Bilinear upsampling that enforces periodicity along the x/longitude axis."""

    def __init__(self, upsampling: int | tuple[int, int] = 2):
        super().__init__()
        if isinstance(upsampling, int):
            upsampling = (upsampling, upsampling)
        if tuple(upsampling) != (2, 2):
            raise ValueError(
                "ZonallyPeriodicBilinearUpsample only supports 2x upsampling"
            )
        self.scale_h, self.scale_w = upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample with periodic padding along longitude to avoid seams and
        # keep interpolation aligned with PyTorch's bilinear sampling grid.
        width = x.shape[-1]
        padded = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="circular")
        upsampled = torch.nn.functional.interpolate(
            padded,
            scale_factor=(self.scale_h, self.scale_w),
            mode="bilinear",
            align_corners=False,
        )
        # Crop out the extra padded columns (scaled by the upsampling factor).
        start = self.scale_w
        end = start + width * self.scale_w
        return upsampled[..., start:end]


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


class CoreBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        pad: str,
        upscale_factor: int = 1,
        norm: str = "batch",
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)
        self.pad = pad
        self.upscale_factor = upscale_factor
        self.norm = norm

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
        pad="circular",
        checkpoint_simple: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)

        layers: list[torch.nn.Module] = []
        layers.append(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation)
        )
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(activation())
        for _ in range(n_layers - 1):
            layers.append(
                torch.nn.Conv2d(
                    out_channels, out_channels, kernel_size, dilation=dilation
                )
            )
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(activation())

        self.layers = nn.ModuleList(layers)
        self.checkpoint_simple = checkpoint_simple

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
                # conv2d layers are expensive so we save their activations,
                # other (simple) layers are cheap, so we don't save their activations.
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                fts = torch.utils.checkpoint.checkpoint(layer, fts, use_reentrant=False)
            else:
                fts = layer(fts)
        return fts


def _pointwise(use_linear: bool, in_ch: int, out_ch: int) -> torch.nn.Module:
    """Create a pointwise (1×1) channel-mixing layer.

    When ``use_linear`` is True, returns a :class:`PointwiseLinear` backed by
    ``nn.Linear``, which produces 2-D weight tensors and avoids the
    non-contiguous gradient strides that ``Conv2d(kernel_size=1)`` introduces
    for degenerate spatial dimensions.  This matters for DDP, which otherwise
    falls back to copying gradients instead of using zero-copy views.

    The ``nn.Linear`` approach is also used in the official ConvNeXt
    implementation, where it is noted to be "slightly faster in PyTorch" [0].

    [0]: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    if use_linear:
        return PointwiseLinear(in_ch, out_ch)
    return torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, padding="same")


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
        pad="circular",
        upscale_factor: int = 4,
        norm="batch",
        group_norm_groups: int = 32,
        checkpoint_simple: bool = False,
        pointwise_linear: bool = False,
    ):
        super().__init__(in_channels, out_channels, kernel_size, dilation, pad)
        assert n_layers == 1, "Can only use a single layer here!"

        # Instantiate pointwise linear to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = _pointwise(pointwise_linear, in_channels, out_channels)

        # Convolution block
        convblock: list[torch.nn.Module] = []
        convblock.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        norm_layer = self._build_norm_layer(
            norm=norm,
            channels=int(in_channels * upscale_factor),
            group_norm_groups=group_norm_groups,
        )
        if norm_layer is not None:
            convblock.append(norm_layer)
        if activation is not None:
            convblock.append(activation())
        convblock.append(
            torch.nn.Conv2d(
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        norm_layer = self._build_norm_layer(
            norm=norm,
            channels=int(in_channels * upscale_factor),
            group_norm_groups=group_norm_groups,
        )
        if norm_layer is not None:
            convblock.append(norm_layer)
        if activation is not None:
            convblock.append(activation())
        # Linear postprocessing
        convblock.append(
            _pointwise(
                pointwise_linear, int(in_channels * upscale_factor), out_channels
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)
        self.checkpoint_simple = checkpoint_simple

    @staticmethod
    def _build_norm_layer(
        norm: str, channels: int, group_norm_groups: int
    ) -> torch.nn.Module | None:
        if norm == "batch":
            return torch.nn.BatchNorm2d(channels)
        elif norm == "instance":
            return torch.nn.InstanceNorm2d(channels)
        elif norm == "group":
            if group_norm_groups < 1:
                raise ValueError("group_norm_groups must be >= 1")
            num_groups = min(group_norm_groups, channels)
            while channels % num_groups != 0:
                num_groups -= 1
            return torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm == "layer":
            # LayerNorm-like behavior for NCHW tensors: one group over all channels.
            return torch.nn.GroupNorm(num_groups=1, num_channels=channels)
        elif norm == "nonorm":
            return None
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return self.skip_module(x) + self.convblock(x)
        skip = self.skip_module(x)
        for layer in self.convblock:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            if self.checkpoint_simple and not isinstance(layer, nn.Conv2d):
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return skip + x


class CoreBlockBuilder(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        n_layers: int,
        pad: str,
        checkpoint_simple: bool,
    ) -> CoreBlock: ...


class UpsamplingBlockBuilder(Protocol):
    def __call__(
        self, in_channels: int, out_channels: int
    ) -> BilinearUpsample | TransposedConvUpsample: ...
