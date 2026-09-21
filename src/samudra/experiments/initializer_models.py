# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Wave-three surface-history initializers, including periodic shifted attention."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from samudra.experiments.surface_state import make_unet


class AttentionBlock(nn.Module):
    """Shifted windows wrap longitude, never latitude; None means global attention."""

    relative_index: torch.Tensor

    def __init__(self, width, heads, window=6, shift=0):
        super().__init__()
        self.width, self.heads, self.window, self.shift = width, heads, window, shift
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        if window is not None:
            self.relative_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
            xy = torch.stack(
                torch.meshgrid(
                    torch.arange(window), torch.arange(window), indexing="ij"
                )
            ).flatten(1)
            delta = xy[:, :, None] - xy[:, None, :] + window - 1
            self.register_buffer(
                "relative_index",
                delta[0] * (2 * window - 1) + delta[1],
                persistent=False,
            )
            nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def attend(self, tokens, mask=None):
        b, length, width = tokens.shape
        q, k, v = (
            self.qkv(tokens)
            .reshape(b, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.projection(result.transpose(1, 2).reshape(b, length, width))

    def attention(self, x):
        b, h, w, c = x.shape
        if self.window is None:
            return self.attend(x.reshape(b, h * w, c)).reshape(b, h, w, c)
        n = self.window
        ph, pw = (-h) % n, (-w) % n
        # Explicit periodic longitude extension; zero padding beyond latitude.
        xp = torch.cat((x, x[:, :, :pw]), 2) if pw else x
        xp = F.pad(xp, (0, 0, 0, 0, 0, ph))
        hp, wp = h + ph, w + pw
        y = torch.arange(hp, device=x.device)[:, None].expand(hp, wp)
        if self.shift:
            xp = torch.roll(xp, (-self.shift, -self.shift), (1, 2))
            y = torch.roll(y, (-self.shift, -self.shift), (0, 1))
        tokens = (
            xp.reshape(b, hp // n, n, wp // n, n, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, n * n, c)
        )
        ys = y.reshape(hp // n, n, wp // n, n).permute(0, 2, 1, 3).reshape(-1, n * n)
        allowed = (ys[:, None, :] < h) & ((ys[:, :, None] - ys[:, None, :]).abs() < n)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        mask = (
            bias[None]
            .expand(ys.shape[0], -1, -1, -1)
            .masked_fill(~allowed[:, None], float("-inf"))
        )
        mask = mask.repeat(b, 1, 1, 1).to(tokens.dtype)
        result = (
            self.attend(tokens, mask)
            .reshape(b, hp // n, wp // n, n, n, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, hp, wp, c)
        )
        if self.shift:
            result = torch.roll(result, (self.shift, self.shift), (1, 2))
        return result[:, :h, :w]

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SwinReconstructor(nn.Module):
    """Swin-style hierarchy with global bottleneck and dense multiscale decoder.

    Uses torch SDPA directly to control ocean boundary handling. No ImageNet
    weights, image resize, classification head, or undocumented external code.
    """

    def __init__(
        self, inputs, outputs, widths=(192, 384, 768, 1024), depths=(2, 2, 12, 2)
    ):
        super().__init__()
        self.patch = nn.Conv2d(inputs, widths[0], 2, stride=2)
        self.stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        AttentionBlock(
                            width,
                            width // 32,
                            window=None if i == 3 else 6,
                            shift=3 if j % 2 else 0,
                        )
                        for j in range(depths[i])
                    ]
                )
                for i, width in enumerate(widths)
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv2d(a, b, 2, stride=2)
                for a, b in zip(widths[:-1], widths[1:], strict=True)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(a + b, b, 1), nn.GELU(), nn.Conv2d(b, b, 1))
                for a, b in zip(widths[:0:-1], widths[-2::-1], strict=True)
            ]
        )
        self.head = nn.Conv2d(widths[0], outputs, 1)

    @staticmethod
    def even_pad(x):
        h, w = x.shape[-2:]
        if w % 2:
            x = torch.cat((x, x[..., :1]), -1)
        return F.pad(x, (0, 0, 0, h % 2))

    def forward(self, x):
        shape = x.shape[-2:]
        x = self.patch(self.even_pad(x))
        skips = []
        for i, stage in enumerate(self.stages):
            assert isinstance(stage, nn.ModuleList)
            x = x.permute(0, 2, 3, 1)
            for block in stage:
                x = (
                    checkpoint(block, x, use_reentrant=False)
                    if self.training and torch.is_grad_enabled()
                    else block(x)
                )
            x = x.permute(0, 3, 1, 2)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](self.even_pad(x))
        for block, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = block(
                torch.cat(
                    (
                        F.interpolate(
                            x,
                            size=skip.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        skip,
                    ),
                    1,
                )
            )
        return self.head(
            F.interpolate(x, size=shape, mode="bilinear", align_corners=False)
        )


class HistoryInitializer(nn.Module):
    def __init__(self, names, architecture, expanded):
        super().__init__()
        self.channels = len(names)
        self.surface = [names.index("thetao_0"), names.index("zos")]
        self.history = 19 if expanded else 6
        self.expanded = expanded
        inputs = self.history * (7 if expanded else 4) + 5
        self.net: nn.Module
        if architecture == "swin":
            self.net = SwinReconstructor(inputs, 2 * self.channels)
        else:
            widths = {"unet": [128, 192, 256, 384], "wide": [256, 384, 512, 768]}[
                architecture
            ]
            self.net = make_unet(inputs, 2 * self.channels, widths)

    def forward(self, surface, past_forcing, context, mask):
        surface = surface[:, -self.history :]
        b, _, _, h, w = surface.shape
        masks = mask[self.surface].expand(b, self.history, 2, h, w)
        pieces = [surface.flatten(1, 2), masks.flatten(1, 2), context]
        if self.expanded:
            pieces.append(past_forcing[:, -self.history :].flatten(1, 2))
        result = (
            self.net(torch.cat(pieces, 1)).float().reshape(b, 2, self.channels, h, w)
        )
        result[:, :, self.surface] = surface[:, -2:]
        return result * mask
