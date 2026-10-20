# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar-exact observation replay with detached, versioned latent states."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.monthly_annual import month_layout
from samudra.experiments.monthly_pilot_data import (
    observation_batch,
    observation_window_loss,
)
from samudra.experiments.observation_data import SPLITS, context_planes


def window_layout(origin, month):
    """Whole native bins spanning one calendar month on the origin's time grid."""
    origin, first = pd.Timestamp(origin), pd.Timestamp(month)
    end = first + pd.offsets.MonthBegin(1)
    if first < origin or first.day != 1 or origin.day != 1:
        raise ValueError("Replay requires ordered monthly origins")
    start_index = (first - origin).days // 5
    end_index = int(np.ceil((end - origin).days / 5))
    starts = pd.date_range(
        origin + pd.Timedelta(days=5 * start_index),
        periods=end_index - start_index,
        freq="5D",
    )
    _, weights, duration, _ = month_layout(starts, str(first.date()))
    position = (first - origin).days + duration / 2
    return start_index, end_index, starts, weights, duration, position


class ReplayPool:
    """Eight recent entries per lead-month; uniform sampling across occupied leads.

    Creation age counts optimizer updates, including OM4 updates. Stale entries
    are rejected rather than silently supplied as current-model hidden states.
    """

    def __init__(self, per_lead=8, max_age=250, max_months=11):
        self.per_lead, self.max_age, self.max_months = per_lead, max_age, max_months
        self.entries = defaultdict(list)

    def add_next(self, origin, month, states, step, *, origin_step=None):
        origin, month = pd.Timestamp(origin), pd.Timestamp(month)
        next_month = month + pd.offsets.MonthBegin(1)
        depth = (next_month.year - origin.year) * 12 + next_month.month - origin.month
        if depth > self.max_months or str(next_month.date())[:7] > SPLITS["train"][1]:
            return
        current_start = window_layout(origin, month)[0]
        next_start, next_end, *_ = window_layout(origin, next_month)
        if next_end > 73:
            return
        latent = states[next_start - current_start].detach().cpu().clone()
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite replay state")
        self.entries[depth].append(
            dict(
                origin=str(origin.date()),
                month=str(next_month.date())[:7],
                bin_index=next_start,
                created_step=step,
                origin_step=step if origin_step is None else origin_step,
                latent=latent,
            )
        )
        self.entries[depth] = self.entries[depth][-self.per_lead :]

    def draw(self, rng, step):
        for depth in list(self.entries):
            self.entries[depth] = [
                e
                for e in self.entries[depth]
                if 0 <= step - e["origin_step"] <= self.max_age
            ]
        depths = sorted(d for d, entries in self.entries.items() if entries)
        if not depths:
            return None
        depth = depths[int(rng.integers(len(depths)))]
        candidates = self.entries[depth]
        return candidates[int(rng.integers(len(candidates)))]

    def state_dict(self):
        return dict(
            per_lead=self.per_lead,
            max_age=self.max_age,
            max_months=self.max_months,
            entries=dict(self.entries),
        )

    def load_state_dict(self, state):
        for key in ("per_lead", "max_age", "max_months"):
            if state[key] != getattr(self, key):
                raise ValueError("Replay configuration changed")
        self.entries = defaultdict(list, state["entries"])


class ReplayIntervals:
    """Read-only raw surface/forcing bins; normalized with the original statistics."""

    def __init__(self, root):
        self.root = Path(root)
        self.starts = np.load(self.root / "starts.npy")
        self.values = np.load(self.root / "values.npy", mmap_mode="r")
        if len(self.starts) != len(self.values) or not np.all(
            np.diff(self.starts) == 1
        ):
            raise ValueError("Replay interval index is not contiguous")

    def load(self, data, starts):
        days = np.asarray(starts, dtype="datetime64[D]").astype(np.int64)
        indices = days - self.starts[0]
        if np.any(indices < 0) or np.any(indices >= len(self.values)):
            raise ValueError("Replay request exceeds the training cache")
        raw = np.array(self.values[indices])
        validity = np.isfinite(raw[:, :2]) & data.grid["mask"][[38, 76]]
        surface = np.where(
            validity,
            (raw[:, :2] - data.grid["mean"][[38, 76], None, None])
            / data.grid["std"][[38, 76], None, None],
            0,
        )
        atmosphere = (
            raw[:, 2:] - data.stats["atmosphere_mean"][None, :, None, None]
        ) / data.stats["atmosphere_std"][None, :, None, None]
        if not np.isfinite(atmosphere).all():
            raise ValueError("Nonfinite replay forcing")
        return dict(
            surface=data.tensor(surface)[None],
            validity=data.tensor(validity)[None],
            atmosphere=data.tensor(atmosphere)[None],
            contexts=data.tensor(
                context_planes(
                    data.grid["lat"], data.grid["lon"], starts + pd.Timedelta(days=2.5)
                )
            )[None],
        )


def replay_loss(model, data, intervals, entry, generator, *, capture=None):
    """Truncated BPTT through one later month, with absolute decoder lead times."""
    origin, month = entry["origin"], entry["month"]
    if not SPLITS["train"][0] <= month <= SPLITS["train"][1]:
        raise ValueError("Replay target is outside the training split")
    start, _, starts, weights, _, position = window_layout(origin, month)
    if entry["bin_index"] != start:
        raise ValueError("Cached latent timestamp does not match the target window")
    later = intervals.load(data, starts)
    # This fresh history is used solely for the original reconstruction loss.
    # It never initializes or corrects the continued forecast.
    sample = data.load(data.root / "train" / (month + ".npz"))
    batch = observation_batch(data, sample, "surface", generator)
    initial_forcing = model.adapt(sample["atmosphere"][:, :19])
    reconstruction_state = model.initialize(
        batch["fields"][:, :19],
        batch["available"][:, :19],
        initial_forcing,
        sample["contexts"][:, 18],
    )
    forcing = model.adapt(later["atmosphere"])
    initial = entry["latent"].detach().to(data.device)
    states = model.processor.rollout(initial, forcing, later["contexts"])
    if capture is not None:
        capture(states)
    target_sample = {
        **sample,
        "surface": torch.cat((sample["surface"][:, :19], later["surface"]), 1),
        "validity": torch.cat((sample["validity"][:, :19], later["validity"]), 1),
        "month_weights": data.tensor(weights),
    }
    return observation_window_loss(
        model,
        data,
        target_sample,
        batch,
        states,
        generator,
        surface_offset_days=5 * start,
        monthly_position_days=position,
        reconstruction_state=reconstruction_state,
    )
