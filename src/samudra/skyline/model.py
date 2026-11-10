# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Dense attention over continuous ocean columns and one depth query."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ModelConfig:
    width: int = 1024
    layers: int = 16
    heads: int = 8
    bins: int = 2048
    objective: str = "absolute_ce"
    checkpoint_blocks: bool = True
    field_queries: bool = False
    coarse_bins: int = 0

    def __post_init__(self):
        if self.width % self.heads or self.layers < 1 or self.bins < 2:
            raise ValueError("Invalid transformer dimensions")
        if self.objective not in {"absolute_ce", "residual_ce", "regression"}:
            raise ValueError(self.objective)
        if self.coarse_bins and (
            self.coarse_bins < 2
            or self.bins % self.coarse_bins
            or self.bins // self.coarse_bins < 2
            or not self.field_queries
            or self.objective != "absolute_ce"
        ):
            raise ValueError("Coarse/fine requires divisible categorical field queries")


class CoarseFineHead(nn.Module):
    """Exact joint categorical masses for contiguous coarse/fine leaf IDs.

    All coarse branches are enumerated, allowing exact joint MAP decoding.
    Each conditional fine distribution sees a learned discrete coarse embedding.
    """

    def __init__(self, width, bins, coarse_bins):
        super().__init__()
        hidden = min(width, 128)
        self.coarse = nn.Linear(width, coarse_bins)
        self.context = nn.Linear(width, hidden)
        self.token = nn.Embedding(coarse_bins, hidden)
        self.fine = nn.Linear(hidden, bins // coarse_bins)

    def forward(self, x):
        coarse = self.coarse(x).float().log_softmax(-1)
        context = self.context(x).unsqueeze(-2) + self.token.weight
        fine = self.fine(F.silu(context)).float().log_softmax(-1)
        return (coarse.unsqueeze(-1) + fine).flatten(-2)


class Vocabulary(nn.Module):
    """Per-depth asinh centers; midpoint assignment in normalized physical units.

    The finite histogram has half-cell outer edges. Values beyond support are
    counted explicitly by the caller; a tail bin is not an infinite density.
    """

    centers: torch.Tensor
    edges: torch.Tensor

    def __init__(self, bounds, bins):
        super().__init__()
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        if (
            bounds.shape != (19,)
            or not torch.isfinite(bounds).all()
            or (bounds <= 0).any()
        ):
            raise ValueError("Need 19 positive finite temperature bounds")
        u = torch.linspace(-1, 1, bins)
        centers = torch.sinh(bounds.asinh()[:, None] * u[None])
        mid = (centers[:, 1:] + centers[:, :-1]) / 2
        edges = torch.cat(
            (
                centers[:, :1] - (mid[:, :1] - centers[:, :1]),
                mid,
                centers[:, -1:] + (centers[:, -1:] - mid[:, -1:]),
            ),
            dim=1,
        )
        self.register_buffer("centers", centers)
        self.register_buffer("edges", edges)

    def encode(self, values, depth):
        return torch.searchsorted(
            self.edges[depth, 1:-1].contiguous(), values[:, None].contiguous()
        ).squeeze(1)

    def outside(self, values, depth):
        return (values < self.edges[depth, 0]) | (values > self.edges[depth, -1])

    def decode(self, logits, depth):
        return self.centers[depth, logits.argmax(-1)]

    def log_density(self, logits, values, depth, physical_std):
        ids = self.encode(values, depth)
        log_mass = logits.float().log_softmax(-1).gather(1, ids[:, None]).squeeze(1)
        widths = self.edges[depth, ids + 1] - self.edges[depth, ids]
        density = log_mass - (widths * physical_std[depth]).log()
        return density.masked_fill(self.outside(values, depth), -torch.inf)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width)
        hidden = ((int(8 * width / 3) + 255) // 256) * 256
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x, valid):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :]
        )
        x = x + self.out(attended.transpose(1, 2).reshape(b, n, d))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class PointTransformer(nn.Module):
    frequencies: torch.Tensor
    head: nn.Module

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.width
        # Channels retain their physical field/depth identities in this projection.
        self.input = nn.Sequential(
            nn.Linear(80 * 2 + 19, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.geo = nn.Linear(6 + 6 * 8 * 2, d, bias=False)
        self.time = nn.Embedding(2, d)
        self.depth = nn.Embedding(19, d)
        self.query = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(
            [Block(d, config.heads) for _ in range(config.layers)]
        )
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(
            d,
            1
            if config.objective == "regression" or config.field_queries
            else config.bins,
        )
        if config.field_queries:
            if config.objective not in {"regression", "absolute_ce"}:
                raise ValueError("Field queries require regression or absolute CE")
            self.field = nn.Embedding(5, d)
            # Preserve the regression backbone AND query initialization before
            # drawing the larger categorical head from the same seeded stream.
            if config.objective == "absolute_ce":
                self.head = (
                    CoarseFineHead(d, config.bins, config.coarse_bins)
                    if config.coarse_bins
                    else nn.Linear(d, config.bins)
                )
        self.register_buffer("frequencies", 2.0 ** torch.arange(8))

    def forward(self, batch):
        values, wet = batch["values"], batch["wet"]
        geometry = batch["geometry"]
        phase = geometry[..., None] * self.frequencies
        position = torch.cat(
            (geometry, phase.sin().flatten(-2), phase.cos().flatten(-2)), -1
        )
        x = self.input(torch.cat((values, values.asinh(), wet), -1)) + self.geo(
            position
        )
        n = values.shape[1] // 2
        times = torch.arange(2, device=x.device).repeat_interleave(n)
        x = x + self.time(times)[None]
        multiple = self.config.field_queries
        z = batch["query_depth_m"] if multiple else batch["depth_m"]
        current = batch["query_current"] if multiple else batch["current"]
        depth = batch["query_depth"] if multiple else batch["depth"]
        query_values = torch.stack((z / 6000, z.log1p() / 9, current), -1)
        query = self.query(query_values) + self.depth(depth)
        if multiple:
            query = query + self.field(batch["query_field"])
        else:
            query = query[:, None]
        # The center's known geography is also available to the forecast query.
        query = query + self.geo(position[:, n // 2])[:, None]
        x = torch.cat((x, query), 1)
        valid = torch.cat(
            (
                batch["valid"],
                batch["query_valid"]
                if multiple
                else torch.ones(query.shape[:2], device=x.device, dtype=torch.bool),
            ),
            1,
        )
        for block in self.blocks:
            if self.training and self.config.checkpoint_blocks:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        if multiple:
            return self.head(self.norm(x[:, -query.shape[1] :])).squeeze(-1).float()
        return self.head(self.norm(x[:, -1])).float()
