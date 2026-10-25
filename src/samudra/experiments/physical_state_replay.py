# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Detached physical-member replay on exact observation and OM4 month windows."""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from samudra.experiments.diffusion_spatial import forecast_crps
from samudra.experiments.observation_data import SPLITS
from samudra.experiments.stochastic_state_replay import ReplayIntervals, window_layout

__all__ = [
    "PhysicalReplayPool",
    "ReplayIntervals",
    "states_from_predictions",
    "sample_native_trajectory",
    "observation_replay_objective",
    "native_replay_objective",
]


def states_from_predictions(initial, predictions):
    """Recover two-slot member histories without resampling any member."""
    states = [initial]
    for lead in range(predictions.shape[2]):
        states.append(torch.stack((states[-1][:, :, -1], predictions[:, :, lead]), 2))
    return states


class PhysicalReplayPool:
    """Uniform occupied-month sampling; age measured from original initialization."""

    def __init__(self, source, per_lead=2, max_age=250, max_months=11, seed=1729001):
        if (
            source not in ("observation", "om4")
            or min(per_lead, max_age, max_months) < 1
        ):
            raise ValueError("Invalid physical replay configuration")
        self.source, self.per_lead, self.max_age, self.max_months = (
            source,
            per_lead,
            max_age,
            max_months,
        )
        self.rng = np.random.default_rng(seed)
        self.entries = defaultdict(list)

    def _add(self, depth, state, step, origin_step, **metadata):
        if depth > self.max_months:
            return
        state = state.detach().cpu().clone()
        if state.ndim != 6 or state.shape[2] != 2 or not torch.isfinite(state).all():
            raise ValueError(
                "Replay requires finite member/batch/two-slot physical states"
            )
        self.entries[depth].append(
            dict(
                state=state,
                source=self.source,
                created_step=step,
                origin_step=step if origin_step is None else origin_step,
                **metadata,
            )
        )
        self.entries[depth] = self.entries[depth][-self.per_lead :]

    def add_observation(self, origin, month, states, step, *, origin_step=None):
        if self.source != "observation":
            raise ValueError("Wrong replay source")
        origin, month = pd.Timestamp(origin), pd.Timestamp(month)
        next_month = month + pd.offsets.MonthBegin(1)
        depth = (next_month.year - origin.year) * 12 + next_month.month - origin.month
        if str(next_month.date())[:7] > SPLITS["train"][1] or depth > self.max_months:
            return
        current_start = window_layout(origin, month)[0]
        start, end, *_ = window_layout(origin, next_month)
        if end > 73:
            return
        self._add(
            depth,
            states[start - current_start],
            step,
            origin_step,
            origin=str(origin.date()),
            month=str(next_month.date())[:7],
            bin_index=start,
        )

    def add_native(
        self, reader, window_index, states, step, *, origin_index=None, origin_step=None
    ):
        if self.source != "om4":
            raise ValueError("Wrong replay source")
        origin_index = window_index if origin_index is None else origin_index
        next_index = window_index + 1
        if next_index >= len(reader.windows):
            return
        start, stop, *_ = reader.windows[window_index]
        next_start, next_stop, *_ = reader.windows[next_index]
        origin_start = reader.windows[origin_index][0]
        current_month = reader.bounds[start, 1]
        next_month = reader.bounds[next_start, 1]
        depth = next_index - origin_index
        # Skipped/gapped months must not silently join a physical trajectory.
        if (
            next_month.year - current_month.year
        ) * 12 + next_month.month - current_month.month != 1:
            return
        if (
            next_start > stop
            or next_stop - origin_start > 73
            or depth > self.max_months
        ):
            return
        if any(
            reader.bounds[i, 1] != reader.bounds[i + 1, 0]
            for i in range(start - 2, next_stop - 1)
        ):
            return
        self._add(
            depth,
            states[next_start - start],
            step,
            origin_step,
            origin_index=origin_index,
            window_index=next_index,
            start_index=next_start,
            lead_days=int((next_start - origin_start) * 5),
        )

    def draw(self, step):
        for depth in self.entries:
            self.entries[depth] = [
                entry
                for entry in self.entries[depth]
                if 0 <= step - entry["origin_step"] <= self.max_age
            ]
        depths = sorted(depth for depth, values in self.entries.items() if values)
        if not depths:
            return None
        values = self.entries[depths[int(self.rng.integers(len(depths)))]]
        return values[int(self.rng.integers(len(values)))]

    def state_dict(self):
        return dict(
            source=self.source,
            per_lead=self.per_lead,
            max_age=self.max_age,
            max_months=self.max_months,
            entries=dict(self.entries),
            rng=self.rng.bit_generator.state,
        )

    def load_state_dict(self, saved):
        for name in ("source", "per_lead", "max_age", "max_months"):
            if saved[name] != getattr(self, name):
                raise ValueError("Replay configuration changed")
        self.entries = defaultdict(list, saved["entries"])
        self.rng.bit_generator.state = saved["rng"]


def repeat_members(value, members):
    return value.repeat((members,) + (1,) * (value.ndim - 1))


def advance_members(model, states, forcing, context, mask, generator):
    members, batch = states.shape[:2]
    advanced = model.advance(
        states.flatten(0, 1),
        repeat_members(forcing, members),
        repeat_members(context, members),
        mask,
        generator,
    )
    return advanced.reshape(members, batch, *advanced.shape[1:])


def continue_members(model, initial, forcing, contexts, mask, generator):
    """Sample member continuations; only forcing and context can enter."""
    states = [initial.detach()]
    if forcing.shape[:2] != contexts.shape[:2] or forcing.shape[2] != 8:
        raise ValueError("Aligned eight-channel ERA5 required")
    for lead in range(forcing.shape[1]):
        states.append(
            advance_members(
                model, states[-1], forcing[:, lead], contexts[:, lead], mask, generator
            )
        )
    return states


@torch.no_grad()
def sample_native_trajectory(model, batch, generator, members=2):
    n = model.history
    fields, mask, atmosphere = batch["fields"], batch["mask"], batch["atmosphere"]
    contexts = batch["contexts"]
    conditioning, known, anchor = model.encode_native(
        fields[:, :n, model.surface], atmosphere[:, :n], contexts[:, n - 1], mask
    )
    initial = model.initialize(
        repeat_members(conditioning, members),
        repeat_members(known, members),
        repeat_members(anchor, members),
        mask,
        generator,
    )
    initial = initial.reshape(members, fields.shape[0], *initial.shape[1:])
    return continue_members(
        model, initial, atmosphere[:, n:], contexts[:, n:], mask, generator
    )


def observation_replay_objective(model, data, intervals, entry, generator):
    month, origin = entry["month"], entry["origin"]
    if (
        entry["source"] != "observation"
        or not SPLITS["train"][0] <= month <= SPLITS["train"][1]
    ):
        raise ValueError("Observation replay outside training source")
    start, end, starts, weights, *_ = window_layout(origin, month)
    if entry["bin_index"] != start or end > 73:
        raise ValueError("Cached physical timestamp/horizon mismatch")
    later = intervals.load(data, starts)
    sample = data.load(data.root / "train" / (month + ".npz"))
    states = continue_members(
        model,
        entry["state"].to(data.device),
        later["atmosphere"],
        later["contexts"],
        data.mask,
        generator,
    )
    prediction = torch.stack([state[:, :, -1] for state in states[1:]], 2)
    target = {
        **sample,
        "raw_surface": later["raw_surface"],
        "month_weights": data.tensor(weights),
    }
    return forecast_crps(data, prediction, target), states


def native_replay_objective(model, data, reader, forcing_cache, entry, generator):
    """Denoising targets remain gold; each condition is a sampled detached past."""
    index = entry["window_index"]
    if entry["source"] != "om4" or not 0 <= index < len(reader.windows):
        raise ValueError("OM4 replay outside training source")
    start, stop, *_ = reader.windows[index]
    origin_start = reader.windows[entry["origin_index"]][0]
    if start != entry["start_index"] or stop - origin_start > 73:
        raise ValueError("OM4 replay timestamp/horizon mismatch")
    batch = reader.sample(index, data.device)
    atmosphere = forcing_cache.load(reader, index, data)
    contexts, mask = batch["contexts"], batch["mask"]
    states = [entry["state"].to(data.device).detach()]
    losses = []
    members = states[0].shape[0]
    for lead in range(model.history, batch["fields"].shape[1]):
        losses.append(
            model.transition_loss(
                states[-1].flatten(0, 1),
                repeat_members(atmosphere[:, lead], members),
                repeat_members(contexts[:, lead], members),
                repeat_members(batch["fields"][:, lead], members),
                mask,
                mask * data.area,
                generator,
            )
        )
        with torch.no_grad():
            states.append(
                advance_members(
                    model,
                    states[-1],
                    atmosphere[:, lead],
                    contexts[:, lead],
                    mask,
                    generator,
                )
            )
    return torch.stack(losses).mean(), states
