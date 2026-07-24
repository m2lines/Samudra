# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _pad_spatially(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Pad latitude by replication and longitude periodically."""
    height, width = x.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_width:
        x = F.pad(x, (0, pad_width, 0, 0), mode="circular")
    if pad_height:
        x = F.pad(x, (0, 0, 0, pad_height), mode="replicate")
    return x, pad_height, pad_width


def _extend_coordinates(coordinates: torch.Tensor, size: int) -> torch.Tensor:
    if coordinates.ndim != 1:
        raise ValueError("Spatial coordinates must be one-dimensional.")
    if coordinates.numel() < 2:
        raise ValueError("At least two spatial coordinates are required.")
    if coordinates.numel() >= size:
        return coordinates[:size]

    step = (coordinates[-1] - coordinates[0]) / (coordinates.numel() - 1)
    extension = coordinates[-1] + step * torch.arange(
        1,
        size - coordinates.numel() + 1,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    return torch.cat((coordinates, extension))


def _patch_centers(
    coordinates: torch.Tensor, size: int, patch_size: int
) -> torch.Tensor:
    coordinates = _extend_coordinates(coordinates, size)
    return coordinates.reshape(-1, patch_size).mean(dim=1)


class FourierPositionEmbedding(nn.Module):
    """Absolute latitude/longitude Fourier features projected to token width."""

    def __init__(
        self,
        token_dim: int,
        num_features: int,
        min_scale: float,
        max_scale: float,
    ) -> None:
        super().__init__()
        if num_features <= 0 or num_features % 4:
            raise ValueError("num_features must be positive and divisible by four.")
        if min_scale <= 0 or max_scale <= min_scale:
            raise ValueError("Expected 0 < min_scale < max_scale.")

        scales_per_coordinate = num_features // 4
        scales = torch.logspace(
            math.log10(min_scale),
            math.log10(max_scale),
            scales_per_coordinate,
        )
        self.scales: torch.Tensor
        self.register_buffer("scales", scales, persistent=False)
        self.projection = nn.Linear(num_features, token_dim)

    def _features(self, coordinate: torch.Tensor) -> torch.Tensor:
        phase = 2 * torch.pi * coordinate[:, None] / self.scales[None, :]
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        latitude: torch.Tensor,
        longitude: torch.Tensor,
        padded_height: int,
        padded_width: int,
        patch_size: int,
    ) -> torch.Tensor:
        latitude = latitude.to(device=x.device, dtype=torch.float32)
        longitude = longitude.to(device=x.device, dtype=torch.float32)
        latitude = _patch_centers(latitude, padded_height, patch_size)
        longitude = _patch_centers(longitude, padded_width, patch_size)

        lat_features = self._features(latitude)
        lon_features = self._features(longitude)
        lat_grid = lat_features[:, None, :].expand(-1, longitude.numel(), -1)
        lon_grid = lon_features[None, :, :].expand(latitude.numel(), -1, -1)
        embedding = self.projection(torch.cat((lat_grid, lon_grid), dim=-1))
        return x + embedding.to(dtype=x.dtype).unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding2D(nn.Module):
    """Two-dimensional rotary embeddings within an attention window."""

    def __init__(self, head_dim: int, window_size: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 4:
            raise ValueError("Attention head dimension must be divisible by four.")

        axis_dim = head_dim // 2
        inverse_frequency = base ** (
            -torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
        )
        axis = torch.arange(window_size, dtype=torch.float32)
        rows, columns = torch.meshgrid(axis, axis, indexing="ij")
        rows = rows.flatten()
        columns = columns.flatten()

        row_angles = torch.repeat_interleave(
            rows[:, None] * inverse_frequency[None, :], 2, dim=-1
        )
        column_angles = torch.repeat_interleave(
            columns[:, None] * inverse_frequency[None, :], 2, dim=-1
        )
        self.row_cos: torch.Tensor
        self.row_sin: torch.Tensor
        self.column_cos: torch.Tensor
        self.column_sin: torch.Tensor
        self.register_buffer("row_cos", row_angles.cos(), persistent=False)
        self.register_buffer("row_sin", row_angles.sin(), persistent=False)
        self.register_buffer("column_cos", column_angles.cos(), persistent=False)
        self.register_buffer("column_sin", column_angles.sin(), persistent=False)

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        cosine = cosine[None, None, :, :].to(dtype=x.dtype)
        sine = sine[None, None, :, :].to(dtype=x.dtype)
        return x * cosine + _rotate_half(x) * sine

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_row, query_column = query.chunk(2, dim=-1)
        key_row, key_column = key.chunk(2, dim=-1)
        query = torch.cat(
            (
                self._apply_rotary(query_row, self.row_cos, self.row_sin),
                self._apply_rotary(query_column, self.column_cos, self.column_sin),
            ),
            dim=-1,
        )
        key = torch.cat(
            (
                self._apply_rotary(key_row, self.row_cos, self.row_sin),
                self._apply_rotary(key_column, self.column_cos, self.column_sin),
            ),
            dim=-1,
        )
        return query, key


def _partition_windows(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError(
            f"Token grid {(height, width)} must be divisible by window size "
            f"{window_size}."
        )
    x = x.reshape(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(-1, window_size * window_size, channels)


def _restore_windows(
    windows: torch.Tensor,
    batch: int,
    height: int,
    width: int,
    window_size: int,
) -> torch.Tensor:
    channels = windows.shape[-1]
    x = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(batch, height, width, channels)


class WindowAttention(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        window_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads.")
        head_dim = token_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate
        self.qkv = nn.Linear(token_dim, 3 * token_dim)
        self.projection = nn.Linear(token_dim, token_dim)
        self.projection_dropout = nn.Dropout(dropout_rate)
        self.rotary = RotaryEmbedding2D(head_dim, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_rate if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection_dropout(self.projection(attended))


class SwiGLU(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_ratio: float,
        align_to: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        hidden_dim = math.ceil(token_dim * hidden_ratio / align_to) * align_to
        self.input_projection = nn.Linear(token_dim, 2 * hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, token_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, values = self.input_projection(x).chunk(2, dim=-1)
        return self.dropout(self.output_projection(F.silu(gate) * values))


class TransformerUnit(nn.Module):
    """Pre-normalized attention and SwiGLU with batch-wise stochastic depth."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        window_size: int,
        hidden_ratio: float,
        align_to: int,
        dropout_rate: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim)
        self.attention = WindowAttention(
            token_dim, num_heads, window_size, dropout_rate
        )
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = SwiGLU(token_dim, hidden_ratio, align_to, dropout_rate)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale: torch.Tensor | float
        if self.training and self.drop_path_rate:
            keep = (torch.rand((), device=x.device) >= self.drop_path_rate).to(
                dtype=x.dtype
            )
            scale = keep / (1.0 - self.drop_path_rate)
        else:
            scale = 1.0

        # Compute dropped residuals and multiply them by zero instead of
        # short-circuiting. This keeps every parameter in the DDP graph on every
        # rank while preserving batch-wise stochastic-depth behavior.
        x = x + scale * self.attention(self.attention_norm(x))
        return x + scale * self.ffn(self.ffn_norm(x))


class OtterSwinBlock(nn.Module):
    """Window and shifted-window transformer units."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        window_size: int,
        hidden_ratio: float,
        align_to: int,
        dropout_rate: float,
        drop_path_rate: float,
    ) -> None:
        super().__init__()
        self.window_unit = TransformerUnit(
            token_dim=token_dim,
            num_heads=num_heads,
            window_size=window_size,
            hidden_ratio=hidden_ratio,
            align_to=align_to,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
        self.shifted_window_unit = TransformerUnit(
            token_dim=token_dim,
            num_heads=num_heads,
            window_size=window_size,
            hidden_ratio=hidden_ratio,
            align_to=align_to,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
        self.window_size = window_size

    def _apply_unit(self, x: torch.Tensor, unit: nn.Module) -> torch.Tensor:
        batch, height, width, _ = x.shape
        windows = _partition_windows(x, self.window_size)
        windows = unit(windows)
        return _restore_windows(windows, batch, height, width, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_unit(x, self.window_unit)
        shift = self.window_size // 2
        x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        x = self._apply_unit(x, self.shifted_window_unit)
        return torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))


class OtterBackbone(nn.Module):
    """Constant-width shifted-window Transformer U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        token_dim: int,
        stage_depths: tuple[int, ...],
        num_heads: int,
        window_size: int,
        patch_size: int,
        position_features: int,
        position_min_scale: float,
        position_max_scale: float,
        hidden_ratio: float,
        ffn_align_to: int,
        dropout_rate: float,
        drop_path_rate: float,
        checkpoint_blocks: bool,
    ) -> None:
        super().__init__()
        if not stage_depths or any(depth <= 0 for depth in stage_depths):
            raise ValueError("stage_depths must contain positive depths.")

        self.patch_size = patch_size
        self.num_downsamples = len(stage_depths) - 1
        self.required_spatial_multiple = (
            patch_size * 2**self.num_downsamples * window_size
        )
        self.checkpoint_blocks = checkpoint_blocks

        self.tokenizer = nn.Conv2d(
            in_channels, token_dim, kernel_size=patch_size, stride=patch_size
        )
        self.position_embedding = FourierPositionEmbedding(
            token_dim,
            position_features,
            position_min_scale,
            position_max_scale,
        )

        def make_stage(depth: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    OtterSwinBlock(
                        token_dim=token_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        hidden_ratio=hidden_ratio,
                        align_to=ffn_align_to,
                        dropout_rate=dropout_rate,
                        drop_path_rate=drop_path_rate,
                    )
                    for _ in range(depth)
                ]
            )

        self.encoder_stages = nn.ModuleList(
            [make_stage(depth) for depth in stage_depths[:-1]]
        )
        self.downsamplers = nn.ModuleList(
            [
                nn.Conv2d(token_dim, token_dim, kernel_size=2, stride=2)
                for _ in stage_depths[:-1]
            ]
        )
        self.bottleneck: nn.ModuleList = make_stage(stage_depths[-1])
        self.upsamplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(token_dim, token_dim, kernel_size=2, stride=2)
                for _ in stage_depths[:-1]
            ]
        )
        self.skip_fusions = nn.ModuleList(
            [
                nn.Conv2d(2 * token_dim, token_dim, kernel_size=1)
                for _ in stage_depths[:-1]
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [make_stage(depth) for depth in reversed(stage_depths[:-1])]
        )
        self.untokenizer = nn.ConvTranspose2d(
            token_dim,
            out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_blocks and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def _run_stage(self, stage: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(stage, nn.ModuleList):
            raise TypeError(f"Expected ModuleList stage, got {type(stage).__name__}.")
        for block in stage:
            x = self._run_block(block, x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        latitude: torch.Tensor,
        longitude: torch.Tensor,
    ) -> torch.Tensor:
        original_height, original_width = x.shape[-2:]
        x, _, _ = _pad_spatially(x, self.required_spatial_multiple)
        padded_height, padded_width = x.shape[-2:]

        x = self.tokenizer(x).permute(0, 2, 3, 1)
        x = self.position_embedding(
            x,
            latitude,
            longitude,
            padded_height,
            padded_width,
            self.patch_size,
        )

        skips = []
        for stage, downsample in zip(self.encoder_stages, self.downsamplers):
            x = self._run_stage(stage, x)
            skips.append(x)
            x = downsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self._run_stage(self.bottleneck, x)

        for upsample, fusion, stage, skip in zip(
            self.upsamplers,
            self.skip_fusions,
            self.decoder_stages,
            reversed(skips),
        ):
            x = upsample(x.permute(0, 3, 1, 2))
            x = fusion(torch.cat((x, skip.permute(0, 3, 1, 2)), dim=1))
            x = self._run_stage(stage, x.permute(0, 2, 3, 1))

        x = self.untokenizer(x.permute(0, 3, 1, 2))
        return x[..., :original_height, :original_width]
