# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch import nn

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.experiments.velocity_transfer.data import HISTORY


class VelocitySamudra(nn.Module):
    """Two small source adapters surrounding the original, fully shared Samudra U-Net."""

    def __init__(
        self, widths=(64, 96, 128, 192), checkpointing="all", use_geometry=True
    ):
        super().__init__()
        self.use_geometry = use_geometry
        self.stems = nn.ModuleDict(
            {name: nn.Conv2d(27, widths[0], 1) for name in ("duacs", "om4")}
        )
        self.backbone = UNetBackboneConfig(
            ch_width=list(widths),
            dilation=[1] * len(widths),
            n_layers=[1] * len(widths),
            core_block=BlockConfig(upscale_factor=2, norm="batch"),
        ).build(widths[0], pad="circular", checkpointing=checkpointing)
        self.heads = nn.ModuleDict(
            {name: nn.Conv2d(widths[0], 2, 1) for name in ("duacs", "om4")}
        )
        for head in self.heads.values():
            assert isinstance(head, nn.Conv2d)
            assert head.bias is not None
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        history,
        history_valid,
        static,
        lags,
        year_day,
        lead,
        source: str,
        periodic=True,
    ):
        batch, _, _, height, width = history.shape
        if history.shape[1:3] != (HISTORY, 2):
            raise ValueError("Expected four historical velocity pairs")
        conditioning = static
        if not self.use_geometry:
            conditioning = torch.cat(
                (torch.zeros_like(static[:, :5]), static[:, 5:]), dim=1
            )
        phase = 2 * math.pi * year_day / 365.25
        times = torch.cat(
            (lags / 30, phase.sin()[:, None], phase.cos()[:, None], lead[:, None] / 30),
            dim=1,
        )
        times = times[:, :, None, None].expand(batch, -1, height, width)
        features = torch.cat(
            (
                history.flatten(1, 2),
                history_valid.flatten(1, 2).float(),
                conditioning,
                times,
            ),
            dim=1,
        )
        features = self.stems[source](features)
        features = self.backbone(features, pad="circular" if periodic else "constant")
        predicted = history[:, -1] + self.heads[source](features)
        return torch.where(static[:, 5:6].bool(), predicted, 0)


def rollout(model, batch, source: str, steps: int, periodic=True):
    """Roll history with predictions only; target maps never enter forecast inputs."""
    history, valid = batch["history"], batch["history_valid"]
    offsets = batch["day_offsets"]
    lags = offsets[:, :HISTORY]
    current_day = torch.zeros_like(offsets[:, 0])
    predictions = []
    for step in range(steps):
        next_day = offsets[:, HISTORY + step]
        predicted = model(
            history,
            valid,
            batch["static"],
            lags,
            batch["year_day"] + current_day,
            next_day - current_day,
            source,
            periodic,
        )
        predictions.append(predicted)
        history = torch.cat((history[:, 1:], predicted[:, None]), dim=1)
        valid = torch.cat((valid[:, 1:], valid[:, -1:]), dim=1)
        lags = torch.cat(
            (
                lags[:, 1:] - (next_day - current_day)[:, None],
                torch.zeros_like(current_day[:, None]),
            ),
            dim=1,
        )
        current_day = next_day
    return torch.stack(predictions, dim=1)


def weighted_loss(predictions, batch):
    weights = batch["target_valid"].float() * batch["area"][:, None]
    denominator = weights.sum() * 2
    if denominator.item() <= 0:
        raise ValueError("No valid target ocean in this batch")
    return (
        (predictions.float() - batch["targets"]).square() * weights
    ).sum() / denominator


class TrainingRollout(nn.Module):
    """Keep a multi-step unroll within one DDP forward/reducer lifecycle."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, source, steps, periodic):
        return rollout(self.model, batch, source, steps, periodic)
