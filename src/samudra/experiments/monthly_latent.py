# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared masked initialization and direct-window physical diffusion.

The processor has no physical completion interface. Missing target entries are
excluded from both corrupted decoder inputs and scores; this is marginal
masked-target denoising, not a complete-data likelihood or imputed supervision.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from samudra.experiments.diffusion_latent import (
    PersistentLatentProcessor,
    SamudraLatentProcessor,
)
from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.joint_diffusion import JointInteriorDecoder, sample_joint
from samudra.experiments.normalization import configure_normalization
from samudra.experiments.surface_state import make_unet


def calendar_weights(starts, ends, first, last):
    """Exact interval intersections; works with datetime and cftime calendars."""
    intervals = list(zip(starts, ends, strict=True))
    cursor = first
    for a, b in intervals:
        if b <= a:
            raise ValueError("Invalid interval bounds")
        lo, hi = max(a, first), min(b, last)
        if hi > lo:
            if lo != cursor:
                raise ValueError("Intervals overlap or leave a gap")
            cursor = hi
    if cursor != last:
        raise ValueError("Intervals do not exactly cover requested month")
    spans = [
        max(0.0, (min(b, last) - max(a, first)).total_seconds()) for a, b in intervals
    ]
    duration = (last - first).total_seconds()
    if duration <= 0 or abs(sum(spans) - duration) > 1e-6:
        raise ValueError("Intervals do not exactly cover requested month")
    return [s / duration for s in spans]


def initializer_visibility(wet, task, generator):
    """One physical availability mask, applied to every historical input frame."""
    visible = wet.bool().clone()
    if task == "surface":
        visible.zero_()
        visible[[38, 76]] = wet[[38, 76]].bool()
    elif task == "variable":
        groups = [(0, 19), (19, 38), (38, 57), (57, 76), (76, 77)]
        start, end = groups[
            int(torch.randint(5, (), generator=generator, device=wet.device))
        ]
        visible[start:end] = False
    elif task == "gap":
        h, w = wet.shape[-2:]
        cells = wet[38].nonzero()
        if not len(cells):
            raise ValueError("No wet cell for gap placement")
        y, x = cells[
            int(torch.randint(len(cells), (), generator=generator, device=wet.device))
        ]
        height, width = max(1, h // 4), max(1, w // 4)
        low = max(0, min(h - height, int(y) - height // 2))
        cols = (torch.arange(width, device=wet.device) + int(x) - width // 2) % w
        visible[:, low : low + height, cols] = False
    else:
        raise ValueError("Unknown initialization task")
    return visible


class WindowDecoder(nn.Module):
    """Same denoiser for five-day fields and calendar-month target windows."""

    def __init__(self, latent_width=128, width=192):
        super().__init__()
        # Two latent memory slots, support masks, log duration and position.
        self.activation_checkpointing = True
        self._compiled_training = None
        self.core = JointInteriorDecoder(2 * latent_width + 77 + 2, 77, width, 0.5)

    def compile_training(self):
        """Keep inference eager: training graph caching must not alter sampling."""
        self._compiled_training = torch.compile(
            self.core.forward, mode="default", fullgraph=False
        )

    def training_denoise(self, noisy, sigma, conditioning, support):
        function = (
            self._compiled_training
            if self.training and self._compiled_training is not None
            else self.core
        )
        return function(noisy, sigma, conditioning, support)

    def condition(self, latent, support, duration_days, position_days):
        h, w = latent.shape[-2:]
        mask = F.adaptive_avg_pool2d(support.float(), (h, w))
        duration = latent.new_full((latent.shape[0], 1, h, w), duration_days).log() / 4
        position = latent.new_full((latent.shape[0], 1, h, w), position_days / 365.0)
        return torch.cat((latent.flatten(1, 2), mask, duration, position), 1)

    def loss(
        self,
        latent,
        target,
        support,
        area,
        generator,
        *,
        duration_days=5.0,
        position_days=0.0,
        channel_weights=None,
    ):
        support = support.bool().expand_as(target) & torch.isfinite(target)
        if not support.any():
            raise ValueError("No observed target support")
        clean = torch.where(support, target, 0).float()
        sigma = (
            torch.randn(target.shape[0], device=target.device, generator=generator)
            * 1.2
            - 1.2
        ).exp()
        noise = diffusion_noise(
            target.shape, support, generator, self.core.noise_correlation
        )
        noisy = (clean + sigma[:, None, None, None] * noise) * support
        conditioning = self.condition(latent, support, duration_days, position_days)
        denoiser = (
            self._compiled_training
            if self.training and self._compiled_training is not None
            else self.core
        )
        predicted = (
            checkpoint(
                denoiser, noisy, sigma, conditioning, support, use_reentrant=False
            )
            if self.training and self.activation_checkpointing
            else denoiser(noisy, sigma, conditioning, support)
        )
        weights = support * area
        denominator = weights.sum((-2, -1))
        present = denominator > 0
        errors = ((predicted - clean).square() * weights).sum(
            (-2, -1)
        ) / denominator.clamp_min(1e-12)
        channel = (
            torch.ones_like(errors)
            if channel_weights is None
            else channel_weights.expand_as(errors)
        )
        channel = channel * present
        if not (channel.sum(-1) > 0).all():
            raise ValueError("No positively weighted target channel")
        error = (errors * channel).sum(-1) / channel.sum(-1)
        return (error * (1 + sigma.square()) / sigma.square()).mean()

    def sample(
        self,
        latent,
        support,
        generator,
        *,
        duration_days=5.0,
        position_days=0.0,
        steps=32,
    ):
        support = support.expand(latent.shape[0], 77, *support.shape[-2:])
        conditioning = self.condition(latent, support, duration_days, position_days)
        return sample_joint(self.core, conditioning, support, generator, steps=steps)


class MonthlyLatentOcean(nn.Module):
    """One initialization per forecast, with two memory slots in its latent state."""

    def __init__(
        self,
        latent_width=128,
        decoder_width=192,
        shape=(180, 360),
        history=19,
        encoder_width=64,
        latent_stride=4,
        processor_forcing=True,
        processor_kind="small",
        processor_widths=(64, 128, 256),
    ):
        super().__init__()
        self.history, self.shape = history, shape
        self.activation_checkpointing = True
        self.latent_width = latent_width
        if latent_stride not in (1, 2, 4) or any(s % latent_stride for s in shape):
            raise ValueError("Latent stride must divide the physical grid")
        self.latent_stride = latent_stride
        self.latent_shape = tuple(s // latent_stride for s in shape)
        self.frame = nn.Conv2d(154, 16, 1)
        self.initializer = make_unet(
            history * (16 + 3) + 5,
            2 * latent_width,
            [encoder_width, 2 * encoder_width, 4 * encoder_width],
        )
        configure_normalization(self.initializer, "instance")
        self.processor = PersistentLatentProcessor(
            latent_width, 4, self.latent_shape, shape, forcing_enabled=processor_forcing
        )
        self.decoder = WindowDecoder(latent_width, decoder_width)
        self.atmosphere_adapter = nn.Sequential(
            nn.Conv2d(8, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1)
        )

        if processor_kind not in ("small", "samudra"):
            raise ValueError("Unknown latent processor")
        self.processor_kind = processor_kind
        if processor_kind == "samudra":
            # Preserve initializer/decoder/adapter initialization and the global
            # RNG stream when replacing only the processor in a fresh model.
            with torch.random.fork_rng(devices=[]):
                self.processor = SamudraLatentProcessor(
                    latent_width,
                    self.latent_shape,
                    shape,
                    forcing_enabled=processor_forcing,
                    widths=processor_widths,
                )

    def adapt(self, atmosphere):
        b, t, c, h, w = atmosphere.shape
        return self.atmosphere_adapter(atmosphere.reshape(b * t, c, h, w)).reshape(
            b, t, 3, h, w
        )

    def initialize(self, fields, available, forcing, context):
        if fields.shape[1:3] != (self.history, 77) or forcing.shape[1:3] != (
            self.history,
            3,
        ):
            raise ValueError("Initialization history/channel layout differs")
        available = available.bool().expand_as(fields) & torch.isfinite(fields)
        clean = torch.where(available, fields, 0)
        b, t, _, h, w = clean.shape
        frames = self.frame(
            torch.cat((clean, available.float()), 2).reshape(b * t, 154, h, w)
        )
        inputs = torch.cat(
            (frames.reshape(b, -1, h, w), forcing.flatten(1, 2), context), 1
        )
        latent = (
            checkpoint(self.initializer, inputs, use_reentrant=False)
            if self.training and self.activation_checkpointing
            else self.initializer(inputs)
        )
        latent = F.avg_pool2d(latent, self.latent_stride)
        return latent.reshape(b, 2, self.latent_width, *self.latent_shape)

    def trajectory(self, fields, available, forcing, contexts):
        initial = self.initialize(
            fields[:, : self.history],
            available[:, : self.history],
            forcing[:, : self.history],
            contexts[:, self.history - 1],
        )
        return self.processor.rollout(
            initial, forcing[:, self.history :], contexts[:, self.history :]
        )

    @staticmethod
    def monthly_condition(states, weights):
        if len(states) != len(weights) or not torch.isclose(
            weights.sum(), weights.new_tensor(1.0)
        ):
            raise ValueError("Monthly latent weights differ from forecast window")
        return sum(
            state * weight for state, weight in zip(states, weights, strict=True)
        )
