from typing import TYPE_CHECKING

import torch
from einops import rearrange
from perceiver_pytorch.perceiver_io import PerceiverIO
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

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
        perceiver_io: nn.Module,
        *,
        hist: int | None = None,
        num_input_states: int | None = None,
        num_output_states: int | None = None,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hist=hist,
            num_input_states=num_input_states,
            num_output_states=num_output_states,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError(
                f"query_chunk_size must be positive when set, got {query_chunk_size}."
            )

        self.input_embed = nn.Linear(in_channels, input_embedding_dim)
        self.data_coordinate_embed = CoordinateEmbedding(
            coordinate_embedding_dim, input_embedding_dim
        )
        self.query_coordinate_embed = CoordinateEmbedding(
            coordinate_embedding_dim, queries_dim
        )
        self.perceiver_io = perceiver_io
        self.query_chunk_size = query_chunk_size
        self.use_bfloat16 = use_bfloat16

        if checkpointing == "all":
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(m, _checkpoint_types),
            )

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

    def forward_once(self, fts: torch.Tensor, ctx: GridContext) -> torch.Tensor:
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

            data_tokens = self.input_embed(tokens) + self.data_coordinate_embed(
                coords
            ).unsqueeze(0).expand(B, -1, -1)
            query_tokens = self.query_coordinate_embed(coords)

            out = self._decode(data_tokens, query_tokens)

        out = out.to(torch.float32)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return torch.where(ctx.label_mask, out, 0.0)
