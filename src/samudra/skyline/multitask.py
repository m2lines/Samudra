# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched output-task comparison with Samudra's full-grid channel MSE."""

import numpy as np
import torch

from .data import DEPTHS, OceanSource

ARMS = ("temperature", "all_fields", "all_depths")


def query_channels(arm, depths):
    depths = np.asarray(depths)
    if arm == "temperature":
        return (38 + depths)[:, None]
    if arm == "all_fields":
        return np.column_stack([depths + 19 * f for f in range(4)] + [depths * 0 + 76])
    if arm == "all_depths":
        return np.broadcast_to(np.arange(77), (len(depths), 77)).copy()
    raise ValueError(arm)


def full_grid_weights(arm, depths, rows, cols, channels, wet, area, totals=None):
    """Horvitz-Thompson weights for the ORIGINAL depth/area proposal.

    Coverage resampling has its own independent finite-pool correction.
    Base P(z,x)=area(x)/(19 A_z) on depth-z wet cells. For a column,
    marginalize the auxiliary depth: g(x)=area(x)*sum_z wet(z,x)/A_z/19.
    Desired objective is sum_{channel,x} wet*error^2/(C*H*W).
    """
    if totals is None:
        totals = (wet * area[None]).sum((1, 2))
    n = area.size
    qdepth = np.where(channels == 76, 0, channels % 19)
    valid = wet[qdepth, rows[:, None], cols[:, None]].copy()
    if arm == "all_depths":
        marginal = area[rows, cols] * (wet[:, rows, cols] / totals[:, None]).sum(0) / 19
        weights = valid / (77 * n * marginal[:, None])
    else:
        if arm == "all_fields":
            valid &= (channels != 76) | (depths[:, None] == 0)
        count = 19 if arm == "temperature" else 77
        weights = (
            valid * (19 * totals[depths] / (count * n * area[rows, cols]))[:, None]
        )
    return weights.astype(np.float32), valid


class ComparisonSource(OceanSource):
    target_channels = tuple(range(77))

    def __init__(self, *args, arm, **kwargs):
        super().__init__(*args, **kwargs)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.wet_area = (self.wet * self.area[None]).sum((1, 2))

    def sample(self, frame, rng, size, crop, depth_sequence=None, points=None):
        # Reuse precisely the same crop extraction and location random stream.
        temperature_frame = dict(frame, target=frame["target"][38:57])
        batch = super().sample(
            temperature_frame, rng, size, crop, depth_sequence, points
        )
        depths, rows, cols = (batch[k].numpy() for k in ("depth", "row", "col"))
        channels = query_channels(self.arm, depths)
        qdepth = np.where(channels == 76, 0, channels % 19)
        weights, valid = full_grid_weights(
            self.arm, depths, rows, cols, channels, self.wet, self.area, self.wet_area
        )
        extra = dict(
            query_channel=channels,
            query_field=np.where(channels == 76, 4, channels // 19),
            query_depth=qdepth,
            query_depth_m=DEPTHS[qdepth],
            query_current=frame["input"][1, channels, rows[:, None], cols[:, None]],
            query_valid=valid,
            target=frame["target"][channels, rows[:, None], cols[:, None]],
            loss_weight=weights,
        )
        batch.update({k: torch.from_numpy(v.copy()) for k, v in extra.items()})
        return batch


def comparison_losses(prediction, batch):
    if prediction.shape != batch["target"].shape:
        raise ValueError("Prediction/target layout mismatch")
    return ((prediction - batch["target"]).square() * batch["loss_weight"]).sum(-1)


def learning_rate_factor(step, warmup, updates):
    if not 0 < warmup < updates or not 1 <= step <= updates:
        raise ValueError("Invalid update schedule")
    if step <= warmup:
        return step / warmup
    fraction = (step - warmup) / (updates - warmup)
    return 0.1 + 0.9 * (1 + np.cos(np.pi * fraction)) / 2
