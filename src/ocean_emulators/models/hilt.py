"""Hilbert Transformer Ocean Emulator.

This module implements HilT, a U-Net architecture with Hilbert curve-based
local attention for efficient ocean emulation.

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


class OceanUpsampler(nn.Module):
    """Upsampling block for HilT decoder using existing ocean components.

    Combines existing upsampling strategies with format conversions for transformers.

    Args:
        in_dim: Input dimension (after skip concat)
        out_dim: Output dimension
        upsample_type: Type of upsampling ('bilinear', 'transposed_conv')
        norm_layer: Normalization layer (default: LayerNorm)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        upsample_type: str = "bilinear",
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.upsample_type = upsample_type

        if upsample_type == "bilinear":
            self.upsample: nn.Module = BilinearUpsample(upsampling=2)
            # Need conv to change channels after upsampling
            self.proj: nn.Module = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        elif upsample_type == "transposed_conv":
            self.upsample = TransposedConvUpsample(
                in_channels=in_dim,
                out_channels=out_dim,
                upsampling=2,
                activation=lambda: nn.Identity(),  # No activation, will use LayerNorm
            )
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown upsample_type: {upsample_type}")

        self.norm = norm_layer(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C) in transformer format

        Returns:
            Output tensor (B, 2H, 2W, out_dim) in transformer format
        """
        # Convert to conv format
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Upsample
        x = self.upsample(x)  # (B, in_dim, 2H, 2W)
        x = self.proj(x)  # (B, out_dim, 2H, 2W)

        # Convert back to transformer format
        x = x.permute(0, 2, 3, 1)  # (B, 2H, 2W, out_dim)
        x = self.norm(x)
        return x


class HilT(BaseModel):
    """Hilbert Transformer Ocean Emulator.

    U-Net architecture with Hilbert curve-based local attention for efficient
    processing of large ocean grids. Uses space-filling curves to convert 2D
    neighborhood attention into efficient 1D sparse attention.

    Architecture:
        - Stem: Downsampling with circular padding
        - Encoder: 4 stages with Hilbert-ordered neighborhood attention
        - Decoder: 4 stages with skip connections and upsampling
        - Output: Projection to prognostic variables

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (prognostic variables)
        pred_residuals: Whether to predict residuals or absolute values
        last_kernel_size: Kernel size for final output projection
        pad: Padding mode for convolutions (default: 'circular')
        hist: History length
        wet: Ocean/land mask
        static_data: Static features (e.g., bathymetry)
        checkpointing: Activation checkpointing strategy
        gradient_detach_interval: Interval for gradient detaching in autoregressive mode
        embed_dim: Base embedding dimension (default: 96)
        depths: Number of blocks per encoder stage (default: [2, 2, 6, 2])
        num_heads: Number of attention heads per stage (default: [3, 6, 12, 24])
        kernel_sizes: Attention kernel sizes per stage (default: [11, 11, 9, 7])
        decoder_depths: Number of blocks per decoder stage (default: [2, 2, 2])
        mlp_ratio: MLP expansion ratio (default: 4.0)
        drop_rate: Dropout rate (default: 0.0)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        drop_path_rate: Stochastic depth rate (default: 0.1)
        qkv_bias: Whether to use bias in attention QKV projection (default: True)
        qk_scale: Scale factor for attention scores (default: None = auto)
        norm_layer: Normalization layer (default: LayerNorm)
        add_3d_coordinates: Module to add 3D Earth coordinates (default: None)
        upsample_type: Type of upsampling in decoder (default: 'bilinear')
        stem_downsample: Stem downsampling factor (default: 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        hist: int,
        wet: Grid,
        static_data: xr.Dataset | None,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        # HilT-specific params
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        kernel_sizes: list[int],
        decoder_depths: list[int],
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        qkv_bias: bool,
        qk_scale: float | None,
        norm_layer,
        add_3d_coordinates: nn.Module | None,
        upsample_type: str,
        stem_downsample: int,
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

        # Decoder stages with upsampling and skip connections
        self.decoder_stages = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        # Build decoder in reverse (from bottleneck up)
        for i in range(self.num_levels - 2, -1, -1):
            # Upsampler
            upsample_in_dim = int(embed_dim * 2 ** (i + 1))
            upsample_out_dim = int(embed_dim * 2**i)
            self.upsamplers.append(
                OceanUpsampler(
                    in_dim=upsample_in_dim,
                    out_dim=upsample_out_dim,
                    upsample_type=upsample_type,
                    norm_layer=norm_layer,
                )
            )

            # Decoder block (after skip connection concat)
            # Input will be: upsampled features + skip connection
            # So input dim is: upsample_out_dim + upsample_out_dim = 2 * upsample_out_dim
            decoder_depth = decoder_depths[min(i, len(decoder_depths) - 1)]
            decoder_stage = NATBlock(
                dim=upsample_out_dim * 2,  # After concat with skip
                depth=decoder_depth,
                num_heads=num_heads[i],
                kernel_size=kernel_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.0,  # No drop path in decoder
                norm_layer=norm_layer,
                downsample=False,
                sequence_order="gilbert",
            )
            self.decoder_stages.append(decoder_stage)

            # Projection to reduce channels after skip concat
            self.decoder_stages.append(
                nn.Conv2d(
                    upsample_out_dim * 2,
                    upsample_out_dim,
                    kernel_size=1,
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

        # 2. Stem: (B, C, 360, 720) → (B, 180, 360, embed_dim)
        x = self.stem(fts)

        # 4. Encoder path with skip connections
        encoder_features = []
        for i, stage in enumerate(self.encoder_stages[:-1]):
            x = stage(x)  # (B, H, W, C) - Hilbert attention applied
            encoder_features.append(x)
            x = self.downsamplers[i](x)  # (B, H/2, W/2, 2C)

        # Bottleneck (no downsampling)
        x = self.encoder_stages[-1](x)

        # 5. Decoder path with skip connections
        for i, (upsampler, skip) in enumerate(
            zip(self.upsamplers, reversed(encoder_features))
        ):
            # Upsample
            x = upsampler(x)  # (B, 2H, 2W, C)

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=-1)  # (B, H, W, 2C)

            # Get decoder stage and projection
            decoder_stage = self.decoder_stages[i * 2]
            proj = self.decoder_stages[i * 2 + 1]

            # Refine with attention (in transformer format)
            x = decoder_stage(x)  # (B, H, W, 2C)

            # Project to reduce channels (need conv format)
            x = x.permute(0, 3, 1, 2)  # (B, 2C, H, W)
            x = proj(x)  # (B, C, H, W)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # 6. Final upsample to original resolution
        # x is (B, H_stem, W_stem, embed_dim) → need (B, 360, 720, embed_dim)
        x = x.permute(0, 3, 1, 2)  # (B, embed_dim, H_stem, W_stem)
        x = self.final_upsample(x)  # (B, embed_dim, 360, 720)

        # 7. Output projection
        x = self.output_proj(x)  # (B, out_channels, 360, 720)

        # 8. Apply wet mask to output
        x = torch.where(self.wet, x, 0.0)

        return x
