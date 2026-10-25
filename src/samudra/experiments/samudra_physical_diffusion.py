# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical-state conditional diffusion using Samudra ConvNeXt U-Nets.

OM4 supplies denoising targets. Observation supervision differentiates through
sampled trajectories; the state has identical physical meaning on both tasks.
"""

import math

import torch
from torch import nn

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.experiments.joint_diffusion import denoising_loss, sample_joint


class SamudraDenoiser(nn.Module):
    """EDM preconditioning around the repository Samudra U-Net backbone."""

    def __init__(self, conditions, fields, widths, noise_correlation=0.5):
        super().__init__()
        self.fields = fields
        self.noise_correlation = noise_correlation
        self.paired_noise_draws = False
        self.embedding = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 16))
        # Instance normalization matches the current latent-diffusion baseline and
        # avoids coupling otherwise independent ensemble members through batch statistics.
        backbone = UNetBackboneConfig(
            ch_width=list(widths),
            dilation=[2**i for i in range(len(widths))],
            n_layers=[1] * len(widths),
            core_block=BlockConfig(upscale_factor=2, norm="instance"),
        ).build(conditions + 2 * fields + 16, pad="circular", checkpointing=None)
        self.net = nn.Sequential(backbone, nn.Conv2d(widths[0], fields, 1))

    def enable_compile(self, mode="default"):
        # Compile a callable, not a registered Module: stable state_dict keys.
        # Evaluation bypasses this cache and remains eager.
        self._compiled_net = torch.compile(self.net.forward, mode=mode)

    def __getstate__(self):
        # EMA deepcopies and whole-module serialization must not capture a bound
        # compiled callable referring to another model's parameters.
        state = super().__getstate__()
        state.pop("_compiled_net", None)
        return state

    def forward(self, noisy, sigma, conditioning, mask):
        sigma = sigma.reshape(-1, 1, 1, 1)
        frequency = torch.exp(
            torch.arange(16, device=noisy.device).float() * (-math.log(10000) / 15)
        )
        phase = sigma.flatten()[:, None].log() / 4 * frequency[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos()), 1))
        planes = embedding[:, :, None, None].expand(-1, -1, *noisy.shape[-2:])
        inputs = torch.cat(
            (
                noisy * torch.rsqrt(1 + sigma.square()),
                conditioning,
                mask.expand_as(noisy).to(noisy.dtype),
                planes,
            ),
            dim=1,
        )
        network = (
            getattr(self, "_compiled_net", self.net) if self.training else self.net
        )
        correction = network(inputs).float()
        return (
            noisy / (1 + sigma.square())
            + sigma * torch.rsqrt(1 + sigma.square()) * correction
        ) * mask


class SamudraPhysicalDiffusion(nn.Module):
    """Two diffusers, persistent physical member states, no learned decoder.

    ``initializer`` provides only channel/history metadata. Its existing weights
    are not retained. Both data sources use the same eight normalized ERA5
    channels directly; no learned mixing or flux adapter is present.
    """

    def __init__(
        self,
        initializer,
        *,
        widths=(64, 128, 256),
        sampling_steps=8,
        noise_correlation=0.5,
        checkpoint_denoiser=True,
        vectorize_members=True,
    ):
        super().__init__()
        if sampling_steps < 2:
            raise ValueError("Heun sampling requires at least two noise levels")
        self.channels = initializer.channels
        self.history = initializer.history
        self.surface = list(initializer.surface)
        self.stochastic = True
        self.sampling_steps = sampling_steps
        self.checkpoint_denoiser = checkpoint_denoiser
        self.vectorize_members = vectorize_members
        self.initializer = SamudraDenoiser(
            self.history * (2 * len(self.surface) + 8) + 5,
            2 * self.channels,
            widths,
            noise_correlation,
        )
        self.processor = SamudraDenoiser(
            2 * self.channels + 8 + 5,
            self.channels,
            widths,
            noise_correlation,
        )

    def enable_compile(self, mode="default"):
        self.initializer.enable_compile(mode)
        self.processor.enable_compile(mode)

    def encode_native(self, surface, past, context, mask, validity=None, task="om4"):
        del task  # Shared representation and shared parameters across data sources.
        if surface.shape[1:3] != (self.history, len(self.surface)):
            raise ValueError("Incorrect surface history shape")
        if past.shape[1:3] != (self.history, 8):
            raise ValueError("Incorrect forcing history shape")
        if validity is None:
            validity = mask[self.surface].expand_as(surface)
        if validity.shape != surface.shape:
            raise ValueError("Surface validity must match surface history")
        valid = validity.bool() & mask[self.surface].bool()
        # Invalid surface values cannot leak through conditioning or anchors.
        surface = torch.where(valid, surface, torch.zeros_like(surface))
        conditioning = torch.cat(
            (
                surface.flatten(1, 2),
                valid.flatten(1, 2).float(),
                context,
                past.flatten(1, 2),
            ),
            1,
        )
        known = surface.new_zeros(
            surface.shape[0], 2, self.channels, *surface.shape[-2:]
        )
        anchor = torch.zeros_like(known, dtype=torch.bool)
        known[:, :, self.surface] = surface[:, -2:]
        anchor[:, :, self.surface] = valid[:, -2:]
        return conditioning, known.flatten(1, 2), anchor.flatten(1, 2)

    initializer_condition = encode_native

    @staticmethod
    def processor_condition(states, forcing, context):
        if forcing.ndim != 4 or forcing.shape[1] != 8:
            raise ValueError("Processor requires eight shared ERA5 channels")
        return torch.cat((states.flatten(1, 2), forcing, context), 1)

    def initialize(self, conditioning, known, anchor, mask, generator):
        result = sample_joint(
            self.initializer,
            conditioning,
            mask.repeat(2, 1, 1),
            generator,
            steps=self.sampling_steps,
            known_values=known,
            known_mask=anchor,
            checkpoint_denoiser=self.training and self.checkpoint_denoiser,
        )
        return result.reshape(result.shape[0], 2, self.channels, *result.shape[-2:])

    def advance(self, states, forcing, context, mask, generator):
        """Sample a new state; retain the old latest state exactly in slot zero."""
        conditioning = self.processor_condition(states, forcing, context)
        following = sample_joint(
            self.processor,
            conditioning,
            mask,
            generator,
            steps=self.sampling_steps,
            checkpoint_denoiser=self.training and self.checkpoint_denoiser,
        )
        return torch.stack((states[:, -1], following), 1)

    def pretraining_loss(self, surface, past, context, truth, mask, weights, generator):
        conditioning, known, anchor = self.encode_native(surface, past, context, mask)
        target = truth.flatten(1, 2)
        if target.shape != known.shape:
            raise ValueError("Initializer target requires two complete physical states")
        return denoising_loss(
            self.initializer,
            conditioning,
            target,
            mask.repeat(2, 1, 1),
            weights.repeat(2, 1, 1),
            generator,
            known_mask=anchor,
            checkpoint_denoiser=self.training and self.checkpoint_denoiser,
        )

    def transition_loss(
        self, states, forcing, context, target, mask, weights, generator
    ):
        conditioning = self.processor_condition(states, forcing, context)
        return denoising_loss(
            self.processor,
            conditioning,
            target,
            mask,
            weights,
            generator,
            checkpoint_denoiser=self.training and self.checkpoint_denoiser,
        )

    def forecast(
        self,
        surface,
        atmosphere,
        contexts,
        mask,
        validity,
        *,
        generator,
        members=1,
        decode_initial=True,
    ):
        if members < 1:
            raise ValueError("At least one member is required")
        count = self.history
        if atmosphere.shape[1] <= count or contexts.shape[:2] != atmosphere.shape[:2]:
            raise ValueError("Aligned future forcing and context frames required")
        if atmosphere.ndim != 5 or atmosphere.shape[2] != 8:
            raise ValueError("Forecast requires eight shared ERA5 channels")
        conditioning, known, anchor = self.encode_native(
            surface[:, :count],
            atmosphere[:, :count],
            contexts[:, count - 1],
            mask,
            validity[:, :count],
            task="observation",
        )
        if self.vectorize_members:
            # Member-major (M, B) layout is retained at every physical step.
            # Each sampler draws independent noise for all M*B entries. This
            # changes RNG draw ordering versus serial trajectories, not its law.
            batch = surface.shape[0]

            def repeat(value):
                return value.unsqueeze(0).expand(members, *value.shape).flatten(0, 1)

            states = self.initialize(
                repeat(conditioning), repeat(known), repeat(anchor), mask, generator
            )
            initial = states.reshape(members, batch, *states.shape[1:])
            predictions = []
            for step in range(count, atmosphere.shape[1]):
                states = self.advance(
                    states,
                    repeat(atmosphere[:, step]),
                    repeat(contexts[:, step]),
                    mask,
                    generator,
                )
                predictions.append(
                    states[:, -1].reshape(
                        members, batch, self.channels, *states.shape[-2:]
                    )
                )
            return torch.stack(predictions, 2), initial if decode_initial else None
        trajectories, initials = [], []
        for _ in range(members):
            states = self.initialize(conditioning, known, anchor, mask, generator)
            if decode_initial:
                initials.append(states)
            predictions = []
            for step in range(count, atmosphere.shape[1]):
                states = self.advance(
                    states, atmosphere[:, step], contexts[:, step], mask, generator
                )
                predictions.append(states[:, -1])
            trajectories.append(torch.stack(predictions, 1))
        return torch.stack(trajectories), torch.stack(
            initials
        ) if decode_initial else None
