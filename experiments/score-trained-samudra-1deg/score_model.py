# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-local ensemble model and scores; no production model changes."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from samudra.config import SamudraConfig
from samudra.models.base import BaseModel
from samudra.models.samudra import Samudra


def crps(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Almost-fair CRPS; member axis first, remaining axes match truth.

    alpha=0 gives empirical CRPS, alpha=1 fair CRPS. The pair formulation
    avoids subtracting two large, almost-equal reductions. All arithmetic FP32.
    """
    x, y = x.float(), y.float()
    m = x.shape[0]
    if m == 1:
        return (x[0] - y).abs()
    distances = (x - y).abs()
    result = torch.zeros_like(y)
    coefficient = 1 - (1 - alpha) / m
    for i in range(m):
        for j in range(i + 1, m):
            result = (
                result + distances[i] + distances[j] - coefficient * (x[i] - x[j]).abs()
            )
    return result / (m * (m - 1))


def weights(ctx, channels: int, *, area: bool) -> torch.Tensor:
    wet = ctx.label_mask[:channels].float()
    if area:
        lat = ctx.output_resolution_cpu[0].to(wet.device).float()
        wet = wet * torch.cos(torch.deg2rad(lat)).clamp_min(0)[None, :, None]
    return wet


def reduce_field(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Average contexts and valid area separately for every channel."""
    return (x * w).sum(dim=(-2, -1)).mean(dim=0) / w.sum(dim=(-2, -1)).clamp_min(1e-12)


def pool_field(x: torch.Tensor, w: torch.Tensor, size: int):
    """Area/wet weighted nonoverlapping blocks, with empty edge padding.

    The grid starts at its stored southwest corner. Partial northern/eastern
    blocks are retained; padding never duplicates or wraps real ocean cells.
    """
    h, width = x.shape[-2:]
    padding = (0, (-width) % size, 0, (-h) % size)
    denominator = F.avg_pool2d(F.pad(w, padding), size) * size**2
    flat = (x * w).flatten(0, -4)
    numerator = F.avg_pool2d(F.pad(flat, padding), size) * size**2
    result = numerator / denominator.clamp_min(1e-12)
    return result.reshape(*x.shape[:-2], *result.shape[-2:]), denominator


def depth_difference(x: torch.Tensor):
    """Adjacent depth differences for T/S/U/V, independently for each time."""
    frames = x.unflatten(-3, (-1, 77))
    tsuv = frames[..., :76, :, :].unflatten(-3, (4, 19))
    return (tsuv[..., 1:, :, :] - tsuv[..., :-1, :, :]).flatten(-5, -3)


def depth_weights(w: torch.Tensor):
    frames = w.unflatten(0, (-1, 77))[:, :76].unflatten(1, (4, 19))
    return torch.minimum(frames[:, :, 1:], frames[:, :, :-1]).flatten(0, 2)


def time_difference(x: torch.Tensor, previous: torch.Tensor):
    frames = x.unflatten(-3, (2, 77))
    earlier = torch.stack((previous, frames[..., 0, :, :, :]), dim=-4)
    return (frames - earlier).flatten(-4, -3)


class NoiseProjection(nn.Module):
    """Independent spatial Gaussian channels mixed after an encoder block."""

    def __init__(self, channels: int, noise_channels: int = 8):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(channels, noise_channels) * (0.02 / math.sqrt(noise_channels))
        )

    def forward(self, x):
        noise = torch.randn(
            x.shape[0],
            self.weight.shape[1],
            *x.shape[-2:],
            device=x.device,
            dtype=x.dtype,
        )
        return x + torch.einsum("bnhw,cn->bchw", noise, self.weight.to(x.dtype))


def add_noise(module, args, output):
    return module.score_noise(output)


class ScoreModel(BaseModel):
    spatial_rms: torch.Tensor
    depth_rms: torch.Tensor
    time_rms: torch.Tensor

    def __init__(
        self,
        base: Samudra,
        arm: str,
        widths: list[int],
        spatial_weight=0.1,
        difference_weight=0.1,
    ):
        super().__init__(
            in_channels=base.in_channels,
            out_channels=base.out_channels,
            input_steps=base.input_steps,
            pred_residuals=base.pred_residuals,
            last_kernel_size=2 * base.N_pad + 1,
            pad=base.pad,
            gradient_detach_interval=base.gradient_detach_interval,
        )
        self.base = base
        self.arm = arm
        self.members = 1 if arm == "B" else 4
        self.spatial_weight = spatial_weight
        self.difference_weight = difference_weight
        # A dedicated initialization stream leaves the common backbone unchanged.
        if arm != "B":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(150001)
                for i, width in enumerate(widths):
                    layer = base.unet.layers[2 * i]
                    layer.add_module("score_noise", NoiseProjection(width))
                    layer.register_forward_hook(add_noise)
        # Saved in every checkpoint, including EMA (these are fixed buffers).
        self.register_buffer("spatial_rms", torch.ones(2, 154))
        self.register_buffer("depth_rms", torch.ones(144))
        self.register_buffer("time_rms", torch.ones(154))

    def forward_once(self, prognostic, boundary, ctx):
        return self.base.forward_once(prognostic, boundary, ctx)

    def sample_step(self, history, boundary, ctx):
        """Keep [member, context] order when folding members into batch."""
        m, b = history.shape[:2]
        forcing = boundary.unsqueeze(0).expand(m, *boundary.shape).flatten(0, 1)
        pred = self.forward_once(history.flatten(0, 1), forcing, ctx)
        return self._assemble_prediction(history.flatten(0, 1), pred).unflatten(
            0, (m, b)
        )

    def score_components(self, pred, target, previous_pred, previous_true, ctx):
        # Same land-inclusive spatial reduction as the existing MSE control.
        wet = weights(ctx, 154, area=False)
        grid = (crps(pred, target) * wet).mean(dim=(0, 2, 3))
        if self.arm != "D":
            return {"grid": grid}
        area = weights(ctx, 154, area=True)
        spatial = []
        for i, size in enumerate((4, 16)):
            px, pw = pool_field(pred, area, size)
            py, _ = pool_field(target, area, size)
            rms = self.spatial_rms[i]
            values = reduce_field(crps(px, py), pw) / rms.clamp_min(1e-6)
            spatial.append(values[rms > 1e-6].mean())
        dx, dy = depth_difference(pred), depth_difference(target)
        dw = depth_weights(area)
        depth = reduce_field(crps(dx, dy), dw) / self.depth_rms.clamp_min(1e-6)
        tx = time_difference(pred, previous_pred)
        ty = time_difference(target, previous_true)
        temporal = reduce_field(crps(tx, ty), area) / self.time_rms.clamp_min(1e-6)
        differences = (
            depth[self.depth_rms > 1e-6].mean() + temporal[self.time_rms > 1e-6].mean()
        ) / 2
        return {
            "grid": grid,
            "spatial": torch.stack(spatial).mean(),
            "differences": differences,
        }

    def score(self, pred, target, previous_pred, previous_true, ctx):
        terms = self.score_components(pred, target, previous_pred, previous_true, ctx)
        if self.arm != "D":
            return terms["grid"]
        return (
            terms["grid"]
            + self.spatial_weight * terms["spatial"]
            + self.difference_weight * terms["differences"]
        )

    def forward(self, batch, loss_fn=None):
        if self.arm == "B":
            return super().forward(batch, loss_fn=loss_fn)
        initial, _ = batch.get_initial_input()
        history = initial.unsqueeze(0).expand(self.members, *initial.shape)
        previous_true = initial[:, -77:]
        outputs = []
        losses = []
        for step in range(len(batch)):
            if (
                step
                and self.gradient_detach_interval > 0
                and step % self.gradient_detach_interval == 0
            ):
                history = history.detach()
            _, boundary = batch.get_input(step)
            pred = self.sample_step(history, boundary, batch.ctx)
            target = batch.get_label(step)
            if loss_fn is not None:
                losses.append(
                    self.score(
                        pred, target, history[:, :, -77:], previous_true, batch.ctx
                    )
                )
            outputs.append(pred.mean(0))
            history = torch.cat((history[:, :, self.out_channels :], pred), dim=2)
            previous_true = target[:, -77:]
        return torch.stack(losses).sum(0) if loss_fn is not None else outputs


class ScoreModelConfig(SamudraConfig):
    arm: Literal["B", "C", "D"] = "B"
    spatial_weight: float = 0.1
    difference_weight: float = 0.1

    def build(self, *args, **kwargs):
        return ScoreModel(
            super().build(*args, **kwargs),
            self.arm,
            self.unet.ch_width,
            self.spatial_weight,
            self.difference_weight,
        )
