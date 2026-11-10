# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-channel asinh categories with explicit open-ended edge classes."""

import torch
from torch import nn
from torch.nn import functional as F


class ChannelVocabulary(nn.Module):
    centers: torch.Tensor
    boundaries: torch.Tensor

    def __init__(self, bounds, bins):
        super().__init__()
        bound = torch.as_tensor(bounds, dtype=torch.float32)
        if (
            bound.shape != (77,)
            or not torch.isfinite(bound).all()
            or (bound <= 0).any()
            or bins < 2
        ):
            raise ValueError("Need 77 positive finite bounds and at least two bins")
        centers = torch.sinh(bound.asinh()[:, None] * torch.linspace(-1, 1, bins))
        self.register_buffer("centers", centers)
        self.register_buffer("boundaries", (centers[:, 1:] + centers[:, :-1]) / 2)

    def encode(self, values, channels):
        return torch.searchsorted(
            self.boundaries[channels].contiguous(), values.unsqueeze(-1).contiguous()
        ).squeeze(-1)

    def nll(self, logits, values, channels):
        ids = self.encode(values, channels)
        return F.cross_entropy(
            logits.flatten(0, 1), ids.flatten(), reduction="none"
        ).reshape_as(values)

    def two_hot_loss(self, logits, values, channels):
        """Adjacent physical-center CE; hard-category likelihood stays separate."""
        centers = self.centers[channels]
        upper = (
            torch.searchsorted(centers.contiguous(), values.unsqueeze(-1).contiguous())
            .squeeze(-1)
            .clamp(1, centers.shape[-1] - 1)
        )
        lower = upper - 1
        lo = centers.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        hi = centers.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        fraction = ((values - lo) / (hi - lo)).clamp(0, 1)
        logp = logits.float().log_softmax(-1)
        low_logp = logp.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
        high_logp = logp.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
        return -(1 - fraction) * low_logp - fraction * high_logp

    def predictions(self, logits, channels):
        centers = self.centers[channels]
        mode = centers.gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
        mean = (centers * logits.float().softmax(-1)).sum(-1)
        return mode, mean

    def tails(self, values, channels):
        ids = self.encode(values, channels)
        return (ids == 0) | (ids == self.centers.shape[-1] - 1)

    def quantized(self, values, channels):
        return (
            self.centers[channels]
            .gather(-1, self.encode(values, channels).unsqueeze(-1))
            .squeeze(-1)
        )
