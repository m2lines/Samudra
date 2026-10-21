# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Detached member-state OM4 replay in six-step chunks through day 360."""

from collections import defaultdict

import torch

from samudra.experiments.diffusion_spatial import spatial_crps
from samudra.experiments.diffusion_structure_losses import excess_latent_loss
from samudra.experiments.missingness import structured_visibility
from samudra.experiments.surface_state import advance_season


class OM4ReplayPool:
    """Uniform occupied-lead sampling with original-initialization expiration.

    Dataset indices must advance by exactly one native five-day interval. The
    conservative origin check keeps the full replay horizon inside trainset.
    """

    def __init__(self, train_windows, per_lead=8, max_age=250, max_lead=72, chunk=6):
        if (
            min(train_windows, per_lead, max_age, max_lead, chunk) < 1
            or max_lead % chunk
        ):
            raise ValueError("Invalid native replay configuration")
        self.train_windows, self.per_lead = train_windows, per_lead
        self.max_age, self.max_lead, self.chunk = max_age, max_lead, chunk
        self.entries = defaultdict(list)

    def add_next(self, origin_index, start_lead, states, step, *, origin_step=None):
        if origin_index < 0 or origin_index + self.max_lead >= self.train_windows:
            return
        if start_lead < 0 or start_lead % self.chunk:
            raise ValueError("Native replay lead must align to chunk boundaries")
        next_lead = start_lead + self.chunk
        if next_lead + self.chunk > self.max_lead:
            return
        latent = states[self.chunk].detach().cpu().clone()
        if latent.ndim != 6 or latent.shape[2] != 2 or not torch.isfinite(latent).all():
            raise ValueError("Invalid member-specific native replay state")
        self.entries[next_lead].append(
            dict(
                origin_index=int(origin_index),
                start_lead=next_lead,
                created_step=step,
                origin_step=step if origin_step is None else origin_step,
                latent=latent,
            )
        )
        self.entries[next_lead] = self.entries[next_lead][-self.per_lead :]

    def draw(self, rng, step):
        for lead in list(self.entries):
            self.entries[lead] = [
                entry
                for entry in self.entries[lead]
                if 0 <= step - entry["origin_step"] <= self.max_age
            ]
        leads = sorted(lead for lead, entries in self.entries.items() if entries)
        if not leads:
            return None
        entries = self.entries[leads[int(rng.integers(len(leads)))]]
        return entries[int(rng.integers(len(entries)))]

    def state_dict(self):
        return {
            key: getattr(self, key)
            for key in (
                "train_windows",
                "per_lead",
                "max_age",
                "max_lead",
                "chunk",
            )
        } | {"entries": dict(self.entries)}

    def load_state_dict(self, saved):
        for key in ("train_windows", "per_lead", "max_age", "max_lead", "chunk"):
            if saved[key] != getattr(self, key):
                raise ValueError("Native replay configuration changed")
        self.entries = defaultdict(list, saved["entries"])


def native_replay_objective(model, wave, entry, coverage, seed, capture=None):
    """Continue cached OM4 members; fresh fields enter only auxiliary branches."""
    if model.arm == "ar1":
        raise ValueError("AR1 replay requires cached innovation history")
    index = entry["origin_index"] + entry["start_lead"]
    if not 0 <= index < len(wave.trainset) or not 0 < entry["start_lead"] <= 66:
        raise ValueError("Native replay target outside training horizon")
    surface, past, context, truth, forcing, targets = wave.model_sample(
        wave.trainset, [index]
    )
    if forcing.shape[1] != 6:
        raise ValueError("Native replay requires exactly six five-day steps")
    generator = torch.Generator(device=wave.device).manual_seed(seed + 4000000)
    states = [entry["latent"].detach().to(wave.device)]
    for i in range(6):
        state, _ = model.advance_members(
            states[-1],
            forcing[:, i],
            advance_season(context, (i + 1) * 5),
            generator=generator,
            task="om4",
        )
        states.append(state)
    if capture is not None:
        capture(states)

    def score(prediction, target, support):
        values, present = spatial_crps(prediction, target, support, wave.weights, 1.0)
        return values[present].mean()

    loss = torch.stack(
        [
            score(
                model.decode_members(state, wave.mask)[:, :, -1],
                targets[:, i],
                wave.mask,
            )
            for i, state in enumerate(states[1:])
        ]
    ).mean() + 0.1 * excess_latent_loss(states, states[0])

    # This fresh initializer learns from the current window but never resets or
    # corrects the replay trajectory above.
    available = wave.mask[model.surface].bool().expand_as(surface)
    visible = structured_visibility(available & coverage.bool(), seed)
    initial, known, anchor = model.encode_native(
        torch.where(visible, surface, 0),
        past,
        context,
        wave.mask,
        visible,
        task="om4",
    )
    members = model.sample_initial_members(initial, generator, members=2)
    decoded = model.decode_members(members, wave.mask, known, anchor)
    interior_mask = wave.mask.clone()
    interior_mask[model.surface] = 0
    loss = loss + 0.1 * score(decoded, truth, interior_mask)
    hidden = available[:, -2:] & ~visible[:, -2:]
    if bool(hidden.any()):
        values, present = spatial_crps(
            decoded[:, :, :, model.surface],
            truth[:, :, model.surface],
            hidden,
            wave.weights[model.surface],
            1.0,
        )
        loss = loss + 0.1 * values[present].mean()
    return loss
