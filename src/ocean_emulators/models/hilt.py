"""Hilbert Transformer Ocean Emulator.

This module implements HilT, a simple encoder-decoder architecture with
Hilbert curve-based local attention for efficient ocean emulation.

See https://openreview.net/forum?id=ltYXDRLDGW
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import xarray as xr

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    ConvBlock,
    TransposedConvUpsample,
)
from ocean_emulators.models.modules.hilbert_vendor import (
    NATBlock,
    _global_gilbert_cache,
)

if TYPE_CHECKING:
    from ocean_emulators.config import Checkpointing


class OceanStem(nn.Module):
    """Initial stem for HilT using ocean-aware blocks.

    Converts from (B, C, H, W) Conv format to (B, H', W', embed_dim) Transformer format.
    Handles circular padding for longitude and downsamples to reduce sequence length.

    Args:
        in_channels: Number of input channels
        embed_dim: Embedding dimension for transformer
        pad: Padding mode (default: 'circular' for longitude wraparound)
        downsample_factor: Spatial downsampling factor (1=no downsample, 2=half res)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        pad: str = "circular",
        downsample_factor: int = 2,
    ):
        super().__init__()
        # Reuse existing ConvBlock (handles circular padding)
        self.conv = ConvBlock(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=3,
            dilation=1,
            n_layers=1,
            pad=pad,
        )
        # Optional downsampling
        if downsample_factor > 1:
            self.pool: nn.Module = AvgPool(pooling=downsample_factor)
        else:
            self.pool = nn.Identity()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, H', W', embed_dim) in transformer format
        """
        x = self.conv(x)  # (B, embed_dim, H, W) - circular padding applied
        x = self.pool(x)  # (B, embed_dim, H/factor, W/factor)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim) - Transformer format
        x = self.norm(x)
        return x


class SimpleUpsampler(nn.Module):
    """Simple upsampling block for encoder-decoder architecture.

    Just upsamples without skip connections or attention.

    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        upsample_type: Type of upsampling ('bilinear', 'transposed_conv')
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        upsample_type: str = "bilinear",
    ):
        super().__init__()
        if upsample_type == "bilinear":
            self.upsample: nn.Module = nn.Sequential(
                BilinearUpsample(upsampling=2),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            )
        else:
            self.upsample = TransposedConvUpsample(
                in_channels=in_dim,
                out_channels=out_dim,
                upsampling=2,
                activation=lambda: nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C) in transformer format

        Returns:
            Output tensor (B, 2H, 2W, C') in transformer format
        """
        # Convert to conv format
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Upsample
        x = self.upsample(x)  # (B, C', 2H, 2W)

        # Convert back to transformer format
        x = x.permute(0, 2, 3, 1)  # (B, 2H, 2W, C')
        return x


class HilT(BaseModel):
    """Hilbert Transformer ocean emulator with simple encoder-decoder architecture.

    Architecture:
    - Stem: Downsamples input (e.g., 360×720 → 180×360)
    - Encoder: 4 stages with Hilbert attention, progressive downsampling
    - Decoder: Progressive upsampling back to stem resolution
    - Final upsample: Back to original resolution
    - Output: Projection to prognostic variables

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        wet: Boolean mask for ocean vs land (1=ocean, 0=land)
        hist: History length
        pred_residuals: Whether to predict residuals
        last_kernel_size: Kernel size for final output layer
        pad: Padding type ('circular' for ocean)
        static_data: Static data (not used)
        checkpointing: Checkpointing strategy
        gradient_detach_interval: Gradient detaching interval
        embed_dim: Base embedding dimension
        depths: Number of attention blocks per encoder stage
        num_heads: Number of attention heads per stage
        kernel_sizes: Local attention kernel sizes per stage
        mlp_ratio: MLP expansion ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        qkv_bias: Whether to use bias in attention QKV projection
        qk_scale: Scale factor for attention scores
        norm_layer: Normalization layer
        add_3d_coordinates: Module to add 3D coordinates
        upsample_type: Type of upsampling ('bilinear' or 'transposed_conv')
        stem_downsample: Stem downsampling factor
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wet: Grid,
        hist: int,
        pred_residuals: bool = False,
        last_kernel_size: int = 3,
        pad: str = "circular",
        static_data: xr.Dataset | None = None,
        checkpointing: "Checkpointing | None" = None,
        gradient_detach_interval: int = 0,
        # HilT-specific params
        embed_dim: int = 96,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        norm_layer=nn.LayerNorm,
        add_3d_coordinates: nn.Module | None = None,
        upsample_type: str = "bilinear",
        stem_downsample: int = 2,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            wet=wet,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            static_data=static_data,
            gradient_detach_interval=gradient_detach_interval,
        )

        # Default configurations
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]
        kernel_sizes = kernel_sizes or [11, 11, 9, 7]

        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.add_3d_coordinates_module = add_3d_coordinates

        # Precompute Hilbert paths for common resolutions
        # Calculate expected resolutions based on stem downsampling
        H_start, W_start = wet.shape[-2:]  # 360, 720
        H_stem = H_start // stem_downsample
        W_stem = W_start // stem_downsample

        resolutions = [
            (H_stem // (2**i), W_stem // (2**i)) for i in range(self.num_levels)
        ]
        _global_gilbert_cache.precompute_paths(resolutions)

        # Stem: 360×720 → 180×360 (with downsample_factor=2)
        self.stem = OceanStem(
            in_channels, embed_dim, pad=pad, downsample_factor=stem_downsample
        )

        # Encoder stages with Hilbert attention
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(self.num_levels):
            stage = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=False,  # We handle downsampling separately
                sequence_order="gilbert",  # KEY: Use Hilbert ordering
            )
            self.encoder_stages.append(stage)

            # Add downsampler except for last stage
            if i < self.num_levels - 1:
                from ocean_emulators.models.modules.hilbert_vendor import (
                    ConvDownsampler,
                )

                self.downsamplers.append(
                    ConvDownsampler(dim=int(embed_dim * 2**i), norm_layer=norm_layer)
                )

        # Simple decoder: just upsample back (no attention, no skip connections)
        self.upsamplers = nn.ModuleList()

        # Build decoder in reverse (from bottleneck up)
        for i in range(self.num_levels - 2, -1, -1):
            upsample_in_dim = int(embed_dim * 2 ** (i + 1))
            upsample_out_dim = int(embed_dim * 2**i)
            self.upsamplers.append(
                SimpleUpsampler(
                    in_dim=upsample_in_dim,
                    out_dim=upsample_out_dim,
                    upsample_type=upsample_type,
                )
            )

        # Final upsampling to original resolution
        if stem_downsample > 1:
            if upsample_type == "bilinear":
                self.final_upsample: nn.Module = nn.Sequential(
                    BilinearUpsample(upsampling=stem_downsample),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                )
            else:
                self.final_upsample = TransposedConvUpsample(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    upsampling=stem_downsample,
                    activation=lambda: nn.Identity(),
                )
        else:
            self.final_upsample = nn.Identity()

        # Output projection
        self.output_proj = nn.Conv2d(
            embed_dim,
            out_channels,
            kernel_size=last_kernel_size,
            padding=last_kernel_size // 2,
        )

        # Activation checkpointing
        if checkpointing == "all":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
            )

            apply_activation_checkpointing(
                self, check_fn=lambda m: isinstance(m, NATBlock)
            )

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        """Single forward pass.

        Args:
            fts: Input tensor (B, in_channels, H, W)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # 1. Add 3D coordinates if enabled
        if self.add_3d_coordinates_module is not None:
            fts = self.add_3d_coordinates_module(fts)

        # 2. Apply wet mask to input
        fts = torch.where(self.wet, fts, 0.0)

        # 3. Stem: (B, C, 360, 720) → (B, 180, 360, embed_dim)
        x = self.stem(fts)

        # 4. Encoder path (with Hilbert attention)
        for i, stage in enumerate(self.encoder_stages[:-1]):
            x = stage(x)  # (B, H, W, C) - Hilbert attention applied
            x = self.downsamplers[i](x)  # (B, H/2, W/2, 2C)

        # Bottleneck (no downsampling)
        x = self.encoder_stages[-1](x)

        # 5. Decoder path (simple upsampling, no attention, no skip connections)
        for upsampler in self.upsamplers:
            x = upsampler(x)  # (B, 2H, 2W, C/2)

        # 6. Final upsample to original resolution
        # x is (B, H_stem, W_stem, embed_dim) → need (B, 360, 720, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H_stem, W_stem)
        x = self.final_upsample(x)  # (B, embed_dim, 360, 720)

        # 7. Output projection
        x = self.output_proj(x)  # (B, out_channels, 360, 720)

        # 8. Apply wet mask to output
        x = torch.where(self.wet, x, 0.0)

        return x
