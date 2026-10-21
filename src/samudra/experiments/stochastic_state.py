# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Member-specific latent trajectories with deterministic physical readouts.

Randomness enters the initial state and optionally each transition, never the
readout. AR noise state belongs to a rollout, not to the model; callers must
carry it when splitting a rollout into chunks. All arms have identical parameters.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast


class ConditionalPerturbation(nn.Module):
    """Learned state-dependent amplitude for global and coarse spatial noise."""

    def __init__(self, width, outputs, noise_channels, scale):
        super().__init__()
        self.projection = nn.Conv2d(2 * noise_channels, outputs, 1, bias=False)
        self.amplitude = nn.Conv2d(2 * width, outputs, 1)
        nn.init.zeros_(self.amplitude.weight)
        assert self.amplitude.bias is not None
        nn.init.zeros_(self.amplitude.bias)
        # Nonzero noise projection avoids identical members at initialization.
        nn.init.normal_(self.projection.weight, std=scale / math.sqrt(noise_channels))

    def forward(self, latent, noise):
        global_noise, spatial_noise = noise
        shape = latent.shape[-2:]
        # Periodic longitude interpolation: extend one coarse column on either
        # side and evaluate at the original grid's cell centers.
        nh, nw = spatial_noise.shape[-2:]
        extended = F.pad(spatial_noise, (1, 1, 0, 0), mode="circular")
        yy = (torch.arange(shape[0], device=latent.device).float() + 0.5) / shape[0]
        xx = (torch.arange(shape[1], device=latent.device).float() + 0.5) / shape[1]
        yy, xx = torch.meshgrid(
            2 * yy - 1, 2 * (1 + xx * nw) / (nw + 2) - 1, indexing="ij"
        )
        grid = torch.stack((xx, yy), -1).expand(latent.shape[0], -1, -1, -1)
        spatial = F.grid_sample(
            extended.float(), grid, padding_mode="border", align_corners=False
        )
        noise_features = torch.cat((global_noise.expand(-1, -1, *shape), spatial), 1)
        return self.projection(noise_features) * (
            2 * self.amplitude(latent.flatten(1, 2)).sigmoid()
        )


class StochasticStateForecast(LatentOceanForecast):
    """Four controlled arms: deterministic, initial, fresh, and ar1.

    `encode_native` yields the conditioning mean. `sample_initial_members`
    samples joint two-slot memory. `advance_members` shifts the old final slot
    exactly and perturbs only the predicted next slot. State shape is
    [member, batch, two history slots, latent channels, latitude, longitude].
    """

    def __init__(
        self,
        initializer,
        adapter,
        *,
        arm="initial",
        ar1_rho=0.9,
        noise_scale=0.05,
        noise_channels=8,
        noise_shape=(6, 12),
        **kwargs,
    ):
        if arm not in {"deterministic", "initial", "fresh", "ar1"}:
            raise ValueError(f"Unknown stochastic state arm: {arm}")
        if (
            not 0 <= ar1_rho < 1
            or noise_scale <= 0
            or noise_channels < 1
            or min(noise_shape) < 1
        ):
            raise ValueError("Invalid noise configuration")
        super().__init__(initializer, adapter, stochastic=False, **kwargs)
        self.arm, self.ar1_rho = arm, ar1_rho
        self.noise_channels, self.noise_shape = noise_channels, tuple(noise_shape)
        width = self.processor.width
        self.initial_noise = ConditionalPerturbation(
            width, 2 * width, noise_channels, noise_scale
        )
        self.transition_noise = ConditionalPerturbation(
            width, width, noise_channels, noise_scale
        )

    def _draw_noise(self, count, device, generator):
        return tuple(
            torch.randn(
                count, self.noise_channels, *shape, device=device, generator=generator
            )
            for shape in ((1, 1), self.noise_shape)
        )

    def sample_initial_members(self, initial, generator, members=2):
        if members < 1:
            raise ValueError("Need at least one member")
        state = initial.unsqueeze(0).expand(members, *initial.shape).clone()
        if self.arm != "deterministic":
            flat = state.flatten(0, 1)
            noise = self._draw_noise(flat.shape[0], flat.device, generator)
            delta = self.initial_noise(flat, noise).reshape_as(state)
            state = state + delta
        return state

    def advance_members(
        self, state, forcing, context, *, generator, noise_state=None, task="om4"
    ):
        members, batch = state.shape[:2]
        flat = state.flatten(0, 1)
        force = forcing.unsqueeze(0).expand(members, *forcing.shape).flatten(0, 1)
        ctx = context.unsqueeze(0).expand(members, *context.shape).flatten(0, 1)
        # Random samples are made outside checkpointed calls; recomputation does
        # not advance the caller's explicit generator.
        result = self.call(self.processor, flat, force, ctx, task)
        if self.arm in {"fresh", "ar1"}:
            noise = self._draw_noise(members * batch, flat.device, generator)
            if self.arm == "ar1" and noise_state is not None:
                if any(
                    a.shape != b.shape for a, b in zip(noise, noise_state, strict=True)
                ):
                    raise ValueError(
                        "AR noise state belongs to a different member/batch shape"
                    )
                noise = tuple(
                    self.ar1_rho * old + math.sqrt(1 - self.ar1_rho**2) * new
                    for old, new in zip(noise_state, noise, strict=True)
                )
            delta = self.transition_noise(flat, noise)
            result = torch.stack((result[:, 0], result[:, 1] + delta), 1)
            noise_state = noise if self.arm == "ar1" else None
        return result.reshape(members, batch, *result.shape[1:]), noise_state

    def rollout_members(
        self, initial, forcing, contexts, *, generator, members=2, task="om4"
    ):
        if forcing.shape[:2] != contexts.shape[:2]:
            raise ValueError("Future forcing/context intervals must align")
        current = self.sample_initial_members(initial, generator, members)
        states, noise_state = [current], None
        for step in range(forcing.shape[1]):
            current, noise_state = self.advance_members(
                current,
                forcing[:, step],
                contexts[:, step],
                generator=generator,
                noise_state=noise_state,
                task=task,
            )
            states.append(current)
        return states

    def decode_members(self, state, mask, known=None, anchor=None):
        if (known is None) != (anchor is None):
            raise ValueError("Known surface values and anchor mask must be paired")
        members, batch = state.shape[:2]

        def repeat(value):
            return (
                None
                if value is None
                else value.unsqueeze(0).expand(members, *value.shape).flatten(0, 1)
            )

        decoded = self.readout(
            state.flatten(0, 1), mask, None, repeat(known), repeat(anchor)
        )
        return decoded.reshape(members, batch, *decoded.shape[1:])

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
            task="observation",
        )
        states = self.rollout_members(
            initial,
            adapted[:, count:],
            contexts[:, count:],
            generator=generator,
            members=members,
            task="observation",
        )
        forecasts = torch.stack(
            [self.decode_members(state, mask)[:, :, -1] for state in states[1:]], 2
        )
        decoded_initial = (
            self.decode_members(states[0], mask, known, anchor)
            if decode_initial
            else None
        )
        return forecasts, decoded_initial

    def pretraining_loss(self, *args, **kwargs):
        raise NotImplementedError(
            "Use member rollout and ensemble scoring; inherited mean-state MSE is not the stochastic objective"
        )
