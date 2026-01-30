from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ocean_emulators.models.modules.activations import CappedGELU

NormType = Literal["batch", "instance", "layer"]


def build_norm(norm: NormType, num_channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm == "layer":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm}")


class TokenConditioner(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_prefixes: int,
        num_timesteps: int,
        num_heads: int = 1,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_prefixes = num_prefixes
        self.num_timesteps = num_timesteps
        self.num_tokens = num_prefixes * num_timesteps

        self.tokens = nn.Parameter(
            torch.empty(num_timesteps, num_prefixes, token_dim)
        )
        nn.init.normal_(self.tokens, mean=0.0, std=init_std)

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        if bottleneck.shape[1] != self.token_dim:
            raise ValueError(
                "TokenConditioner expected bottleneck channels "
                f"{self.token_dim}, got {bottleneck.shape[1]}"
            )
        batch = bottleneck.shape[0]
        x = bottleneck.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.tokens.reshape(1, self.num_tokens, self.token_dim).expand(
            batch, -1, -1
        )
        conditioned, _ = self.attn(tokens, x, x, need_weights=False)
        return conditioned.reshape(
            batch, self.num_timesteps, self.num_prefixes, self.token_dim
        )


class TokenConcatMLP(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        out_dim: int,
        hidden_mult: int = 2,
        activation: type[nn.Module] = CappedGELU,
    ) -> None:
        super().__init__()
        in_dim = token_dim * num_tokens
        hidden_dim = max(out_dim, token_dim) * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tokens_concat: torch.Tensor) -> torch.Tensor:
        return self.net(tokens_concat)


class DecoderFiLM(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        block_channels: list[int],
        norm: NormType,
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.block_channels = block_channels
        self.norms = nn.ModuleList(
            [build_norm(norm, channels) for channels in block_channels]
        )
        self.mlps = nn.ModuleList(
            [
                TokenConcatMLP(
                    token_dim=token_dim,
                    num_tokens=num_tokens,
                    out_dim=2 * channels,
                    hidden_mult=hidden_mult,
                )
                for channels in block_channels
            ]
        )

    def forward(
        self, block_idx: int, fts: torch.Tensor, tokens_concat: torch.Tensor
    ) -> torch.Tensor:
        gamma_beta = self.mlps[block_idx](tokens_concat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        normed = self.norms[block_idx](fts)
        return (1.0 + gamma) * normed + beta
