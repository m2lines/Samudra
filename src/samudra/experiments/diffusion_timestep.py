# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""OM4 endpoint consistency for two five-day advances versus one ten-day advance.

Only the newest latent slots represent the same time in the two branches. The
physical ten-day auxiliary readout uses the SHORT branch's day-five slot as
scaffolding, so the existing decoder still receives a five-day-spaced pair.
This does not implement autonomous variable-spacing decoder trajectories.
"""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from samudra.experiments.diffusion_latent import PersistentLatentProcessor
from samudra.experiments.joint_diffusion import denoising_loss
from samudra.experiments.missingness import structured_visibility
from samudra.experiments.surface_state import advance_season


class TimestepLatentProcessor(PersistentLatentProcessor):
    """Shared five/ten-day dynamics; inherited forward remains exactly five-day.

    The ten-day path has no private dynamics head. A zero-initialized conditioning
    adapter receives both ordered forcing/context intervals and normalized elapsed
    time. Shared residual updates are scaled by elapsed time / five days.
    Existing parameter names stay unchanged, allowing five-day evaluators to omit
    only ``timestep_adapter.*`` when loading an auxiliary-training checkpoint.
    """

    def __init__(self, base):
        super().__init__(
            base.width,
            len(base.blocks),
            base.latent_shape,
            base.conditioning_shape,
        )
        if hasattr(base, "input_adapters"):
            self.input_adapters = deepcopy(base.input_adapters)
        first = next(base.parameters())
        self.to(device=first.device, dtype=first.dtype)
        self.load_state_dict(base.state_dict(), strict=True)
        self.timestep_adapter = nn.Conv2d(17, self.width, 1).to(
            device=first.device, dtype=first.dtype
        )
        nn.init.zeros_(self.timestep_adapter.weight)
        assert self.timestep_adapter.bias is not None
        nn.init.zeros_(self.timestep_adapter.bias)
        self.train(base.training)

    def advance_ten_days(self, latent, forcing, contexts, task="om4"):
        """Return [day 0, day 10] from memory [day -5, day 0].

        ``forcing`` and ``contexts`` contain the SAME two ordered five-day
        conditioning blocks used by the short path, never an averaged substitute.
        """
        if tuple(latent.shape[1:]) != (2, self.width, *self.latent_shape):
            raise ValueError("Latent memory shape changed")
        if tuple(forcing.shape[1:]) != (2, 3, *self.conditioning_shape):
            raise ValueError("Ten-day advance requires two ordered forcing blocks")
        if tuple(contexts.shape[1:]) != (2, 5, *self.conditioning_shape):
            raise ValueError("Ten-day advance requires two ordered context blocks")
        if forcing.shape[0] != latent.shape[0] or contexts.shape[0] != latent.shape[0]:
            raise ValueError("Conditioning and latent batches must match")
        condition = F.adaptive_avg_pool2d(
            torch.cat((forcing, contexts), 2).flatten(1, 2), self.latent_shape
        )
        inputs = torch.cat((latent.flatten(1, 2), condition[:, :8]), 1)
        if hasattr(self, "input_adapters"):
            inputs = self.input_adapters[task](inputs)
        elapsed = condition.new_full((condition.shape[0], 1, *self.latent_shape), 2)
        features = self.stem(inputs) + self.timestep_adapter(
            torch.cat((condition, elapsed), 1)
        )
        update = self.head(self.blocks(features))
        return torch.stack((latent[:, -1], latent[:, -1] + 2 * update), 1)


def install_timestep_processor(model):
    """Install before constructing the optimizer; repeated calls are harmless."""
    if not isinstance(model.processor, TimestepLatentProcessor):
        model.processor = TimestepLatentProcessor(model.processor)
    return model.processor


def endpoint_consistency(short, direct, origin):
    """Both endpoint branches receive gradients; historical slots are excluded.

    Normalize each channel by detached short-endpoint RMS, clamped at 0.1. This keeps the
    coefficient interpretable across channel amplitudes without optimizing the
    denominator. It is not a guarantee against latent collapse or instability.
    """
    if short.shape != direct.shape or short.shape != origin.shape:
        raise ValueError("Endpoint memories and origin must have matching shapes")
    if short.ndim != 5 or short.shape[1] != 2:
        raise ValueError("Expected two-slot spatial latent memories")
    scale = short[:, -1].detach().float().square().mean((-2, -1), keepdim=True)
    scale = scale.sqrt().clamp_min(0.1)
    return ((short[:, -1].float() - direct[:, -1].float()) / scale).square().mean()


def auxiliary_loss(
    model, wave, index, coverage, seed, *, physical_weight=0.5, consistency_weight=0.1
):
    """One additional day-10 physical readout, optionally with latent consistency.

    Add this to the unchanged native OM4 objective. Pass consistency_weight=0
    for the supervised-five/ten-day arm; omit the auxiliary for the control.
    Returns scalar auxiliary loss and detached scalar diagnostic tensors.
    Target interiors NEVER enter either latent branch. Neither branch is detached.
    The physical target pair is [day 5, day 10]; its conditioning pair combines
    short-path day 5 with direct-path day 10, preserving decoder slot semantics.
    This auxiliary is only for native OM4 examples, not observation examples.
    """
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("Consistency weight must be finite and nonnegative")
    if not math.isfinite(physical_weight) or physical_weight < 0:
        raise ValueError("Physical weight must be finite and nonnegative")
    if not isinstance(model.processor, TimestepLatentProcessor):
        raise TypeError("Install the timestep processor before training")
    surface, past, context, _, forcing, targets = wave.model_sample(
        wave.trainset, [index]
    )
    if forcing.shape[1] < 2 or targets.shape[1] < 2:
        raise ValueError("Auxiliary requires at least two five-day OM4 targets")
    available = wave.mask[model.surface].bool().expand_as(surface)
    visible = structured_visibility(available & coverage.bool(), seed)
    origin, _, _ = model.encode_native(
        torch.where(visible, surface, 0), past, context, wave.mask, visible, task="om4"
    )
    forcing = forcing[:, :2]
    contexts = torch.stack([advance_season(context, days) for days in (5, 10)], 1)
    short_states = model.processor.rollout(origin, forcing, contexts, task="om4")
    direct = model.processor.advance_ten_days(origin, forcing, contexts, task="om4")
    consistent = endpoint_consistency(short_states[-1], direct, origin)
    decoder_pair = torch.stack((short_states[1][:, -1], direct[:, -1]), 1)
    generator = torch.Generator(device=origin.device).manual_seed(seed + 6000000)
    supervised = denoising_loss(
        model.decoder,
        decoder_pair.flatten(1, 2),
        targets[:, :2].flatten(1, 2),
        wave.mask.repeat(2, 1, 1),
        wave.weights.repeat(2, 1, 1),
        generator,
        checkpoint_denoiser=model.training,
        spatial_weight=0.5,
    )
    return physical_weight * supervised + consistency_weight * consistent, {
        "ten_day_denoising": supervised.detach(),
        "endpoint_consistency": consistent.detach(),
    }
