from collections.abc import Callable
from typing import Literal, Protocol

import torch
import torch.nn as nn
import torch.utils.checkpoint
<<<<<<< HEAD
from einops import rearrange
=======
>>>>>>> 386e3796 (Added jax typing)
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
            convblock.append(activation())
        # Linear postprocessing
        convblock.append(
            _pointwise(
                pointwise_linear, int(in_channels * upscale_factor), out_channels
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)
        self.checkpoint_simple = checkpoint_simple

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


class AxialAttention(nn.Module):
    """Multi-head self-attention along a single spatial axis.

    For an input of shape ``(B, C, H, W)``:

    - ``axis='height'``: treats each column position independently, attends along H
    - ``axis='width'``: treats each row position independently, attends along W

    This decomposes full 2D attention :math:`O((HW)^2)` into two sequential 1D
    operations :math:`O(H^2 + W^2)`, making it tractable for large spatial grids.

    References:
        Axial Attention in Multidimensional Transformers (Ho et al., 2019)
        https://arxiv.org/abs/1912.12180
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        axis: Literal["height", "width"],
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # When True, the next forward pass stores attention weights in
        # ``self.last_attn_weights`` for visualization / debugging.
        self.capture_weights = False
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        B, C, H, W = x.shape

        if self.axis == "height":
            x = rearrange(x, "b c h w -> (b w) h c")
        else:
            x = rearrange(x, "b c h w -> (b h) w c")

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = rearrange(
            qkv, "batch seq three heads head_dim -> three batch heads seq head_dim"
        )
        q, k, v = qkv.unbind(0)

        # Use scaled_dot_product_attention (supports flash / memory-efficient kernels)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        if self.capture_weights:
            # Compute attention weights explicitly for visualization.
            # Detach and move to CPU to avoid holding GPU memory.
            scale = self.head_dim**-0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            # Average over heads: (batch_size, seq, seq)
            attn_avg = attn_weights.mean(dim=1).detach().cpu()
            if self.axis == "height":
                # Reshape back to (B, W, H, H) then average over batch and W
                self.last_attn_weights = attn_avg.reshape(B, W, H, H).mean(dim=(0, 1))
            else:
                # Reshape back to (B, H, W, W) then average over batch and H
                self.last_attn_weights = attn_avg.reshape(B, H, W, W).mean(dim=(0, 1))

        out = rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)")
        out = self.proj(out)
        out = self.proj_drop(out)

        if self.axis == "height":
            out = rearrange(out, "(b w) h c -> b c h w", b=B, w=W)
        else:
            out = rearrange(out, "(b h) w c -> b c h w", b=B, h=H)

        return out


class AxialAttentionBlock(nn.Module):
    """Applies axial self-attention: height-axis followed by width-axis attention.

    Uses pre-normalization (GroupNorm with ``num_groups=1``, equivalent to
    LayerNorm for spatial feature maps) and residual connections for each axis.

    This block can be inserted into any position in a U-Net to add global
    context aggregation while keeping computational cost manageable.

    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads.  Must divide *channels* evenly.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm_h = nn.GroupNorm(1, channels)
        self.attn_h = AxialAttention(
            channels,
            num_heads,
            qkv_bias=True,
            axis="height",
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm_w = nn.GroupNorm(1, channels)
        self.attn_w = AxialAttention(
            channels,
            num_heads,
            qkv_bias=True,
            axis="width",
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        # Height-axis attention with residual
        x = x + self.attn_h(self.norm_h(x))
        # Width-axis attention with residual
        x = x + self.attn_w(self.norm_w(x))
        return x


class FullAttention(nn.Module):
    """Full 2D multi-head self-attention over spatial dimensions.

    Flattens ``(H, W)`` into a single sequence of length ``H * W`` and applies
    standard scaled dot-product attention for low-resolution
    feature maps where the quadratic cost is negligible.

    Supports the same ``capture_weights`` / ``last_attn_weights`` interface
    for visualization.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.  Must divide *dim* evenly.
        qkv_bias: Whether to include bias in QKV projection.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop_p = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.capture_weights = False
        self.last_attn_weights: torch.Tensor | None = None
        self.last_spatial_shape: tuple[int, int] | None = None

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        B, C, H, W = x.shape
        seq_len = H * W

        x = rearrange(x, "b c h w -> b (h w) c")

        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, self.head_dim)
        qkv = rearrange(
            qkv, "batch seq three heads head_dim -> three batch heads seq head_dim"
        )
        q, k, v = qkv.unbind(0)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )

        if self.capture_weights:
            scale = self.head_dim**-0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale
            attn_weights = attn_weights.softmax(dim=-1)
            # Average over heads and batch: (H*W, H*W)
            self.last_attn_weights = attn_weights.mean(dim=1).mean(dim=0).detach().cpu()
            self.last_spatial_shape = (H, W)

        out = rearrange(out, "batch heads seq head_dim -> batch seq (heads head_dim)")
        out = self.proj(out)
        out = self.proj_drop(out)

        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return out


class FullAttentionBlock(nn.Module):
    """Full 2D self-attention block with pre-norm and residual connection.

    Mirrors :class:`AxialAttentionBlock` in interface so either can be
    used interchangeably in the UNet backbone.

    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads.  Must divide *channels* evenly.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = FullAttention(
            channels,
            num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "batch channels height width"],
    ) -> Float[torch.Tensor, "batch channels height width"]:
        return x + self.attn(self.norm(x))


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
