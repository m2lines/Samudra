# SPDX-FileCopyrightText: 2021 Phil Wang
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

"""Perceiver components backed by PyTorch scaled dot product attention."""

import math
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

AttentionBackend = Literal["auto", "math", "flash"]


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = x.chunk(2, dim=-1)
        return values * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class Attention(nn.Module):
    """Projected multi-head attention using PyTorch's fused SDPA dispatcher."""

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        if heads < 1 or dim_head < 1:
            raise ValueError("heads and dim_head must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if backend not in ("auto", "math", "flash"):
            raise ValueError(f"unsupported attention backend: {backend}")

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.backend = backend
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.heads, self.dim_head).transpose(1, 2)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if self.backend == "auto":
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

        backend = (
            SDPBackend.MATH if self.backend == "math" else SDPBackend.FLASH_ATTENTION
        )
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        query = self._split_heads(self.to_q(x))
        key, value = self.to_kv(context).chunk(2, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attention_mask = None
        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                raise ValueError("mask and input batch sizes must match")
            attention_mask = mask.reshape(mask.shape[0], 1, 1, -1).bool()

        attended = self._attention(query, key, value, attention_mask)
        attended = attended.transpose(1, 2).contiguous().flatten(2)
        return self.to_out(attended)


class PreNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        fn: nn.Module,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(self, x: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


def _fourier_encode(
    positions: torch.Tensor, max_freq: float, num_bands: int
) -> torch.Tensor:
    positions = positions.unsqueeze(-1)
    scales = torch.linspace(
        1.0,
        max_freq / 2,
        num_bands,
        device=positions.device,
        dtype=positions.dtype,
    )
    scaled = positions * scales * math.pi
    return torch.cat((scaled.sin(), scaled.cos(), positions), dim=-1)


class Perceiver(nn.Module):
    """Original Perceiver latent encoder using fused PyTorch attention."""

    def __init__(
        self,
        *,
        num_freq_bands: int,
        depth: int,
        max_freq: float,
        input_channels: int = 3,
        input_axis: int = 2,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        num_classes: int = 1000,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        weight_tie_layers: bool = False,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        attention_backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        if depth < 1 or self_per_cross_attn < 1:
            raise ValueError("depth and self_per_cross_attn must be positive")
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        fourier_channels = input_axis * (2 * num_freq_bands + 1)
        input_dim = input_channels + (fourier_channels if fourier_encode_data else 0)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        def cross_block() -> nn.ModuleList:
            return nn.ModuleList(
                (
                    PreNorm(
                        latent_dim,
                        Attention(
                            latent_dim,
                            input_dim,
                            heads=cross_heads,
                            dim_head=cross_dim_head,
                            dropout=attn_dropout,
                            backend=attention_backend,
                        ),
                        context_dim=input_dim,
                    ),
                    PreNorm(
                        latent_dim,
                        FeedForward(latent_dim, dropout=ff_dropout),
                    ),
                )
            )

        def self_block() -> nn.ModuleList:
            return nn.ModuleList(
                (
                    PreNorm(
                        latent_dim,
                        Attention(
                            latent_dim,
                            heads=latent_heads,
                            dim_head=latent_dim_head,
                            dropout=attn_dropout,
                            backend=attention_backend,
                        ),
                    ),
                    PreNorm(
                        latent_dim,
                        FeedForward(latent_dim, dropout=ff_dropout),
                    ),
                )
            )

        tied_cross = cross_block() if weight_tie_layers and depth > 1 else None
        tied_self = (
            [self_block() for _ in range(self_per_cross_attn)]
            if weight_tie_layers and depth > 1
            else None
        )
        self.layers = nn.ModuleList()
        for layer_index in range(depth):
            cross = (
                tied_cross
                if layer_index > 0 and tied_cross is not None
                else cross_block()
            )
            self_attentions = nn.ModuleList(
                tied_self
                if layer_index > 0 and tied_self is not None
                else [self_block() for _ in range(self_per_cross_attn)]
            )
            self.layers.append(nn.ModuleList((cross[0], cross[1], self_attentions)))

        self.to_logits = (
            nn.Sequential(
                MeanPool(),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes),
            )
            if final_classifier_head
            else nn.Identity()
        )

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        batch = data.shape[0]
        axes = data.shape[1:-1]
        if len(axes) != self.input_axis:
            raise ValueError(
                f"expected {self.input_axis} input axes, received {len(axes)}"
            )

        if self.fourier_encode_data:
            axis_positions = [
                torch.linspace(
                    -1.0, 1.0, steps=size, device=data.device, dtype=data.dtype
                )
                for size in axes
            ]
            positions = torch.stack(
                torch.meshgrid(*axis_positions, indexing="ij"), dim=-1
            )
            encoded = _fourier_encode(
                positions, self.max_freq, self.num_freq_bands
            ).flatten(-2)
            encoded = encoded.unsqueeze(0).expand(batch, *encoded.shape)
            data = torch.cat((data, encoded), dim=-1)

        data = data.flatten(1, -2)
        x = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for untyped_layer in self.layers:
            layer = cast(nn.ModuleList, untyped_layer)
            cross_attn, cross_ff = layer[0], layer[1]
            self_attentions = cast(nn.ModuleList, layer[2])
            x = x + cross_attn(x, context=data, mask=mask)
            x = x + cross_ff(x)
            for untyped_self_attention in self_attentions:
                self_attention = cast(nn.ModuleList, untyped_self_attention)
                self_attn, self_ff = self_attention[0], self_attention[1]
                x = x + self_attn(x)
                x = x + self_ff(x)

        if return_embeddings:
            return x
        return self.to_logits(x)


class PerceiverIO(nn.Module):
    """Perceiver IO compatible latent processor and output-query decoder."""

    def __init__(
        self,
        *,
        depth: int,
        dim: int,
        queries_dim: int,
        logits_dim: int | None = None,
        num_latents: int = 512,
        latent_dim: int = 512,
        cross_heads: int = 1,
        latent_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        weight_tie_layers: bool = False,
        decoder_ff: bool = False,
        attention_backend: AttentionBackend = "auto",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList(
            (
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim,
                        dim,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        backend=attention_backend,
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            )
        )

        def latent_block() -> nn.ModuleList:
            return nn.ModuleList(
                (
                    PreNorm(
                        latent_dim,
                        Attention(
                            latent_dim,
                            heads=latent_heads,
                            dim_head=latent_dim_head,
                            backend=attention_backend,
                        ),
                    ),
                    PreNorm(latent_dim, FeedForward(latent_dim)),
                )
            )

        shared = latent_block() if weight_tie_layers else None
        self.layers = nn.ModuleList(
            shared if shared is not None else latent_block() for _ in range(depth)
        )
        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                backend=attention_backend,
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )
        self.to_logits = (
            nn.Linear(queries_dim, logits_dim)
            if logits_dim is not None
            else nn.Identity()
        )

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor | None = None,
        queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = data.shape[0]
        x = self.latents.unsqueeze(0).expand(batch, -1, -1)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = x + cross_attn(x, context=data, mask=mask)
        x = x + cross_ff(x)
        for untyped_layer in self.layers:
            layer = cast(nn.ModuleList, untyped_layer)
            self_attn, self_ff = layer[0], layer[1]
            x = x + self_attn(x)
            x = x + self_ff(x)

        if queries is None:
            return x
        if queries.ndim == 2:
            queries = queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder_cross_attn(queries, context=x)
        if self.decoder_ff is not None:
            decoded = decoded + self.decoder_ff(decoded)
        return self.to_logits(decoded)
