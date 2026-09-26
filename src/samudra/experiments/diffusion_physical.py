# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched deterministic/diffusion initialization with frozen physical dynamics."""

import torch
from torch import nn

from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    channel_balanced_mse,
    denoising_loss,
    sample_joint,
)


class JointPhysicalForecast(nn.Module):
    """A/B share an initializer, decoder capacity, and frozen physical stepper.

    The initializer's unconstrained output is a conditioning representation, not
    a prescribed interior label. A decodes once from zero at sigma=1; B jointly
    samples the two historical states. Known historical surfaces are anchored in
    both. Future surface fields are predicted, never supplied to the stepper.
    """

    def __init__(self, core, *, stochastic, width=64, sampling_steps=16):
        super().__init__()
        self.core = core
        self.stochastic = stochastic
        self.sampling_steps = sampling_steps
        fields = 2 * core.initializer.channels
        self.decoder = JointInteriorDecoder(fields, fields, width=width)
        self.core.evolution.requires_grad_(False)

    def encode(self, surface, atmosphere, contexts, mask, validity):
        count = self.core.initializer.history
        if surface.shape[1] < count or atmosphere.shape[1] < count:
            raise ValueError("Insufficient initializer history")
        return self.encode_native(
            surface[:, :count],
            self.core.adapt(atmosphere[:, :count]),
            contexts[:, count - 1],
            mask,
            validity[:, :count],
        )

    def encode_native(self, surface, past, context, mask, validity=None):
        """Consume already-normalized OM4 fluxes or adapted ERA5; never re-adapt fluxes."""
        count = self.core.initializer.history
        if surface.shape[1] != count or past.shape[1:3] != (count, 3):
            raise ValueError(
                "Native conditioning must have the declared history and three fluxes"
            )
        if validity is None:
            validity = mask[self.core.initializer.surface].expand_as(surface)
        if validity.shape != surface.shape:
            raise ValueError("Surface validity shape differs from supplied history")
        inputs = torch.cat(
            (
                surface.flatten(1, 2),
                validity.flatten(1, 2),
                context,
                past.flatten(1, 2),
            ),
            1,
        )
        latent = self.core.call(self.core.initializer.net, inputs)
        b, _, h, w = latent.shape
        known = latent.new_zeros(b, 2, self.core.initializer.channels, h, w)
        indices = self.core.initializer.surface
        known[:, :, indices] = surface[:, count - 2 : count]
        known_mask = torch.zeros_like(mask, dtype=torch.bool)
        known_mask[indices] = True
        return latent, known.flatten(1, 2), known_mask.repeat(2, 1, 1)

    def pretraining_loss(self, surface, past, context, truth, mask, weights, generator):
        """Use InitializerWave.model_sample outputs and its shared observation state scales.

        Full two-time OM4 interiors supervise initialization, including velocity.
        Original coarse OM4 forcings pass directly to the conditioning network.
        Known surfaces follow the same anchoring convention as inference.
        """
        latent, known, known_mask = self.encode_native(surface, past, context, mask)
        target = truth.flatten(1, 2)
        joint_mask = mask.repeat(2, 1, 1)
        joint_weights = weights.repeat(2, 1, 1) * ~known_mask
        if target.shape != known.shape:
            raise ValueError("OM4 target must contain the two historical full states")
        if self.stochastic:
            return denoising_loss(
                self.decoder,
                latent,
                target,
                joint_mask,
                joint_weights,
                generator,
                known_mask=known_mask,
            )
        prediction = self.initialize(
            latent, known, known_mask, mask, generator
        ).flatten(1, 2)
        return channel_balanced_mse(prediction, target, joint_weights).mean()

    def initialize(self, latent, known, known_mask, mask, generator):
        joint_mask = mask.repeat(2, 1, 1)
        if self.stochastic:
            state = sample_joint(
                self.decoder,
                latent,
                joint_mask,
                generator,
                steps=self.sampling_steps,
                known_values=known,
                known_mask=known_mask,
                checkpoint_denoiser=self.training,
            )
        else:
            noisy = torch.where(known_mask, known, torch.zeros_like(known))
            state = self.core.call(
                self.decoder,
                noisy,
                latent.new_ones(latent.shape[0]),
                latent,
                joint_mask,
            )
            state = torch.where(known_mask, known, state) * joint_mask
        return state.reshape(
            state.shape[0], 2, self.core.initializer.channels, *state.shape[-2:]
        )

    def forecast(
        self, surface, atmosphere, contexts, mask, validity, *, generator, members=1
    ):
        """Return member trajectories and initial pairs; caller controls independent RNG draws."""
        if members < 1 or (not self.stochastic and members != 1):
            raise ValueError(
                "Deterministic arm has one member; sampling needs positive count"
            )
        count = self.core.initializer.history
        if atmosphere.shape[1] <= count or contexts.shape[1] != atmosphere.shape[1]:
            raise ValueError("Need aligned future forcing and context frames")
        latent, known, known_mask = self.encode(
            surface, atmosphere, contexts, mask, validity
        )
        future = self.core.adapt(atmosphere[:, count:])
        trajectories, initials = [], []
        for _ in range(members):
            states = self.initialize(latent, known, known_mask, mask, generator)
            initials.append(states)
            predicted = []
            for step in range(future.shape[1]):
                value = self.core.call(
                    self.core.evolution,
                    states,
                    future[:, step : step + 1],
                    contexts[:, count + step],
                    mask,
                    1,
                )
                predicted.append(value)
                states = torch.stack((states[:, -1], value), 1)
            trajectories.append(torch.stack(predicted, 1))
        return torch.stack(trajectories), torch.stack(initials)
