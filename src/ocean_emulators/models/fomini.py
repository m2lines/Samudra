import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from perceiver_pytorch.perceiver_io import PerceiverIO
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from ocean_emulators.constants import Boundary, Prognostic
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.augment_input import make_3d_coordinate_grid
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.device import autocast

if TYPE_CHECKING:
    from ocean_emulators.config import Checkpointing

_checkpoint_types: tuple[type, ...] = (
    nn.LayerNorm,
    FeedForward,
    nn.Linear,
    PerceiverIO,
    Attention,
)

try:
    from flash_attn.modules.block import (
        Block as FlashBlock,  # type: ignore[import-not-found]
    )
    from flash_perceiver.perceiver import (
        PerceiverBase as FlashPerceiverBase,  # type: ignore[import-not-found]
    )

    _checkpoint_types = _checkpoint_types + (FlashPerceiverBase, FlashBlock)
except ImportError:
    pass


class CoordinateEmbedding(nn.Module):
    """Learned embedding for 3D Cartesian coordinates."""

    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class FactorizedChannelEmbedding(nn.Module):
    """Channel embedding built from variable, relative-time, and depth IDs."""

    def __init__(
        self,
        metadata: Sequence[tuple[int, int, int]],
        *,
        num_variables: int,
        num_times: int,
        num_depths: int,
        dim: int,
    ) -> None:
        super().__init__()
        if not metadata:
            raise ValueError("Channel metadata must be non-empty.")

        ids = torch.tensor(metadata, dtype=torch.long)
        self.register_buffer("variable_ids", ids[:, 0], persistent=False)
        self.register_buffer("time_ids", ids[:, 1], persistent=False)
        self.register_buffer("depth_ids", ids[:, 2], persistent=False)

        self.variable_embed = nn.Embedding(num_variables, dim)
        self.time_embed = nn.Embedding(num_times, dim)
        self.depth_embed = nn.Embedding(num_depths, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self) -> torch.Tensor:
        return self.norm(
            self.variable_embed(self.variable_ids)
            + self.time_embed(self.time_ids)
            + self.depth_embed(self.depth_ids)
        )


class FOMini(BaseModel):
    """Single PerceiverIO model using one token per lat/lon pixel.

    The input data is flattened to pixel tokens, and both data/query position
    information is represented via learned embeddings of 3D unit-sphere
    coordinates (x, y, z).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        input_embedding_dim: int,
        coordinate_embedding_dim: int,
        queries_dim: int,
        query_chunk_size: int | None,
        input_channel_metadata: Sequence[tuple[int, int, int]],
        output_channel_metadata: Sequence[tuple[int, int, int]],
        num_variables: int,
        num_times: int,
        num_depths: int,
        perceiver_io: nn.Module,
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError(
                f"query_chunk_size must be positive when set, got {query_chunk_size}."
            )
        if len(input_channel_metadata) != in_channels:
            raise ValueError(
                "input_channel_metadata length must match in_channels: "
                f"{len(input_channel_metadata)} != {in_channels}."
            )
        if len(output_channel_metadata) != out_channels:
            raise ValueError(
                "output_channel_metadata length must match out_channels: "
                f"{len(output_channel_metadata)} != {out_channels}."
            )

        self.input_channel_embed = FactorizedChannelEmbedding(
            input_channel_metadata,
            num_variables=num_variables,
            num_times=num_times,
            num_depths=num_depths,
            dim=input_embedding_dim,
        )
        self.mask_channel_embed = FactorizedChannelEmbedding(
            input_channel_metadata,
            num_variables=num_variables,
            num_times=num_times,
            num_depths=num_depths,
            dim=input_embedding_dim,
        )
        self.data_coordinate_embed = CoordinateEmbedding(
            coordinate_embedding_dim, input_embedding_dim
        )
        self.query_coordinate_embed = CoordinateEmbedding(
            coordinate_embedding_dim, queries_dim
        )
        self.output_channel_embed = FactorizedChannelEmbedding(
            output_channel_metadata,
            num_variables=num_variables,
            num_times=num_times,
            num_depths=num_depths,
            dim=queries_dim,
        )
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.perceiver_io = perceiver_io
        self.query_chunk_size = query_chunk_size
        self.use_bfloat16 = use_bfloat16

        if checkpointing == "all":
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(m, _checkpoint_types),
            )

    def _input_mask_tokens(
        self,
        ctx: GridContext,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        prognostic_mask = ctx.label_mask.to(device=device, dtype=dtype)
        if prognostic_mask.shape[0] != self.out_channels:
            raise ValueError(
                "Context label mask channel count must match model outputs: "
                f"{prognostic_mask.shape[0]} != {self.out_channels}."
            )

        boundary_channels = self.in_channels - self.out_channels
        if boundary_channels > 0:
            surface_mask = prognostic_mask[:1].expand(boundary_channels, -1, -1)
            input_mask = torch.cat((prognostic_mask, surface_mask), dim=0)
        else:
            input_mask = prognostic_mask

        # Signed mask keeps both wet and land distinguishable. This matters
        # because normalized land fill and normalized mean values are both zero.
        input_mask = input_mask.mul(2.0).sub(1.0)
        return rearrange(input_mask, "c h w -> (h w) c")

    def _decode(
        self,
        data_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.query_chunk_size is None:
            return self.perceiver_io(data_tokens, queries=query_tokens)

        out_chunks = []
        for start in range(0, query_tokens.shape[0], self.query_chunk_size):
            end = start + self.query_chunk_size
            out_chunks.append(
                self.perceiver_io(data_tokens, queries=query_tokens[start:end])
            )
        return torch.cat(out_chunks, dim=1)

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        # FOMini is a single-scale pixel-token model; fuse prognostic +
        # boundary into the single channel-stacked input it expects.
        fts = torch.cat((prognostic, boundary), dim=1)
        B, _, H, W = fts.shape
        lat, lon = ctx.input_resolution_cpu
        if H != len(lat) or W != len(lon):
            raise ValueError(
                "Input tensor and resolution size mismatch: "
                f"tensor has {(H, W)} but resolution has {(len(lat), len(lon))}."
            )

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            # Pixel tokens: one token per (lat, lon), channels are token features.
            tokens = rearrange(fts, "b c h w -> b (h w) c")

            coords = make_3d_coordinate_grid(lat, lon).to(
                device=fts.device, dtype=fts.dtype
            )
            coords = rearrange(coords, "d h w -> (h w) d")

            input_channel_embed = self.input_channel_embed()
            mask_channel_embed = self.mask_channel_embed()
            channel_scale = math.sqrt(self.in_channels)
            value_tokens = (
                torch.einsum("bnc,cd->bnd", tokens, input_channel_embed) / channel_scale
            )
            mask_tokens = (
                torch.einsum(
                    "nc,cd->nd",
                    self._input_mask_tokens(
                        ctx,
                        device=fts.device,
                        dtype=fts.dtype,
                    ),
                    mask_channel_embed,
                )
                / channel_scale
            )

            data_tokens = (
                value_tokens
                + mask_tokens.unsqueeze(0)
                + self.data_coordinate_embed(coords).unsqueeze(0).expand(B, -1, -1)
            )
            query_tokens = self.query_coordinate_embed(coords)

            out = self._decode(data_tokens, query_tokens)
            output_channel_embed = self.output_channel_embed()
            out = torch.einsum("bnd,cd->bnc", out, output_channel_embed) / math.sqrt(
                output_channel_embed.shape[-1]
            )
            out = out + self.output_bias

        out = out.to(torch.float32)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return torch.where(ctx.label_mask, out, 0.0)
