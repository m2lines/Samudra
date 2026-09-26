# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""C/D: deterministic latent evolution with deterministic or diffusion readouts.

D samples conditional readouts independently at each lead. These are not claims
of coherent uncertain trajectories; sampled latent initialization is arm E.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from samudra.experiments.diffusion_latent import (
    LatentHistoryEncoder,
    PersistentLatentProcessor,
)
from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    channel_balanced_mse,
    denoising_loss,
    sample_joint,
)


class LatentOceanForecast(nn.Module):
    """Initializer/processor never receive interior targets or decoded states.

    Pass the chosen pretrained initializer and forcing adapter. The physical
    stepper is intentionally absent from this architecture. C and D share all
    trainable components and differ in readout/supervision only.
    """

    def __init__(
        self,
        initializer,
        adapter,
        *,
        stochastic,
        latent_width=128,
        latent_shape=(45, 90),
        conditioning_shape=(180, 360),
        processor_depth=4,
        decoder_width=64,
        sampling_steps=16,
    ):
        super().__init__()
        self.channels, self.history, self.surface = (
            initializer.channels,
            initializer.history,
            initializer.surface,
        )
        self.stochastic, self.sampling_steps = stochastic, sampling_steps
        self.adapter = adapter
        self.encoder = LatentHistoryEncoder(
            initializer.net, 2 * self.channels, latent_width, latent_shape
        )
        self.processor = PersistentLatentProcessor(
            latent_width, processor_depth, latent_shape, conditioning_shape
        )
        self.decoder = JointInteriorDecoder(
            2 * latent_width, 2 * self.channels, decoder_width
        )

    def call(self, module, *args):
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def adapt(self, atmosphere):
        b, t, c, h, w = atmosphere.shape
        return self.adapter(atmosphere.reshape(b * t, c, h, w)).reshape(b, t, 3, h, w)

    def encode_native(self, surface, past, context, mask, validity=None):
        if surface.shape[1:3] != (self.history, len(self.surface)) or past.shape[
            1:3
        ] != (self.history, 3):
            raise ValueError(
                "Initializer needs its fixed coarse surface/forcing history"
            )
        if validity is None:
            validity = mask[self.surface].expand_as(surface)
        if validity.shape != surface.shape:
            raise ValueError("Validity shape differs from the supplied history")
        inputs = torch.cat(
            (
                surface.flatten(1, 2),
                validity.flatten(1, 2),
                context,
                past.flatten(1, 2),
            ),
            1,
        )
        latent = self.encoder(inputs)
        known = surface.new_zeros(
            surface.shape[0], 2, self.channels, *surface.shape[-2:]
        )
        known[:, :, self.surface] = surface[:, -2:]
        anchor = torch.zeros_like(known, dtype=torch.bool)
        anchor[:, :, self.surface] = validity[:, -2:].bool() & mask[self.surface].bool()
        return latent, known.flatten(1, 2), anchor.flatten(1, 2)

    def readout(self, latent, mask, generator, known=None, anchor=None):
        flat = latent.flatten(1, 2)
        joint_mask = mask.repeat(2, 1, 1)
        if self.stochastic:
            result = sample_joint(
                self.decoder,
                flat,
                joint_mask,
                generator,
                steps=self.sampling_steps,
                known_values=known,
                known_mask=anchor,
                checkpoint_denoiser=self.training,
            )
        else:
            noisy = torch.zeros(
                (flat.shape[0], 2 * self.channels, *mask.shape[-2:]),
                device=flat.device,
                dtype=torch.float32,
            )
            if anchor is not None:
                noisy = torch.where(anchor, known, noisy)
            result = self.call(
                self.decoder, noisy, flat.new_ones(flat.shape[0]), flat, joint_mask
            )
            if anchor is not None:
                result = torch.where(anchor, known, result)
            result = result * joint_mask
        return result.reshape(result.shape[0], 2, self.channels, *result.shape[-2:])

    def pretraining_loss(
        self,
        surface,
        past,
        context,
        truth,
        forcing,
        contexts,
        targets,
        mask,
        weights,
        generator,
    ):
        if truth.shape[1:3] != (2, self.channels) or truth.shape[0] != surface.shape[0]:
            raise ValueError("Initial target must be the two full historical states")
        if targets.shape[1] != forcing.shape[1] or targets.shape[2:] != truth.shape[2:]:
            raise ValueError("Future interior targets and forcing leads must align")
        initial, known, anchor = self.encode_native(surface, past, context, mask)
        # Compute the complete latent trajectory without any interior targets.
        states = self.processor.rollout(initial, forcing, contexts)
        sequence = torch.cat((truth, targets), 1)
        joint_mask, joint_weights = mask.repeat(2, 1, 1), weights.repeat(2, 1, 1)
        losses = []
        for step, latent in enumerate(states):
            target = sequence[:, step : step + 2].flatten(1, 2)
            fixed = anchor if step == 0 else None
            if self.stochastic:
                value = denoising_loss(
                    self.decoder,
                    latent.flatten(1, 2),
                    target,
                    joint_mask,
                    joint_weights,
                    generator,
                    known_mask=fixed,
                )
            else:
                predicted = self.readout(
                    latent, mask, generator, known if step == 0 else None, fixed
                )
                current_weights = (
                    joint_weights if fixed is None else joint_weights * ~fixed
                )
                value = channel_balanced_mse(
                    predicted.flatten(1, 2), target, current_weights
                ).mean()
            losses.append(value)
        return torch.stack(losses).mean()

    def forecast(
        self, surface, atmosphere, contexts, mask, validity, *, generator, members=1
    ):
        if members < 1 or (not self.stochastic and members != 1):
            raise ValueError(
                "Deterministic readout has one member; diffusion needs a positive count"
            )
        count = self.history
        if atmosphere.shape[1] <= count or contexts.shape[:2] != atmosphere.shape[:2]:
            raise ValueError("Aligned future forcing and contexts required")
        adapted = self.adapt(atmosphere)
        initial, known, anchor = self.encode_native(
            surface[:, :count],
            adapted[:, :count],
            contexts[:, count - 1],
            mask,
            validity[:, :count],
        )
        states = self.processor.rollout(
            initial, adapted[:, count:], contexts[:, count:]
        )
        trajectories, initials = [], []
        for _ in range(members):
            initials.append(self.readout(states[0], mask, generator, known, anchor))
            trajectories.append(
                torch.stack(
                    [
                        self.readout(state, mask, generator)[:, -1]
                        for state in states[1:]
                    ],
                    1,
                )
            )
        return torch.stack(trajectories), torch.stack(initials)
