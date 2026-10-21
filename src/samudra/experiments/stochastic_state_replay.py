# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar-exact observation replay with detached, versioned latent states."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from samudra.experiments.observation_annual import month_weights
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
    weights = month_weights(starts, str(first.date())).astype(np.float32)
    duration = float(first.days_in_month)
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
        if latent.ndim != 6 or latent.shape[2] != 2:
            raise ValueError("Replay requires member/batch/two-slot latent history")
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
            raw_surface=data.tensor(raw[:, :2])[None],
            surface=data.tensor(surface)[None],
            validity=data.tensor(validity)[None],
            atmosphere=data.tensor(atmosphere)[None],
            contexts=data.tensor(
                context_planes(
                    data.grid["lat"], data.grid["lon"], starts + pd.Timedelta(days=2.5)
                )
            )[None],
        )


def continue_members(model, initial, forcing, contexts, generator):
    """Continue cached members without resampling or averaging their histories."""
    if model.arm == "ar1":
        raise ValueError("AR1 replay also needs cached innovation history")
    if initial.ndim != 6 or initial.shape[2] != 2:
        raise ValueError("Replay requires member/batch/two-slot latent history")
    if forcing.shape[:2] != contexts.shape[:2]:
        raise ValueError("Replay forcing/context intervals must align")
    states = [initial.detach()]
    for i in range(forcing.shape[1]):
        state, _ = model.advance_members(
            states[-1],
            forcing[:, i],
            contexts[:, i],
            generator=generator,
            task="observation",
        )
        states.append(state)
    return states


def replay_parts(model, data, intervals, entry, seed, *, capture=None):
    """One calendar month of truncated BPTT plus separate fresh auxiliaries.

    Only future forcing and calendar context enter the continued trajectory.
    Current-month surface history is confined to completion and reconstruction.
    This decoder has no lead embedding; absolute lead is maintained by state,
    forcing timestamps and calendar-month overlaps, not a reset decoder clock.
    """
    from samudra.experiments.diffusion_spatial import (
        completion_crps,
        forecast_crps,
        interior_crps,
    )
    from samudra.experiments.diffusion_structure_losses import excess_latent_loss
    from samudra.experiments.missingness import corrupt_sample
    from samudra.experiments.stochastic_state_train import coherent_reconstruction, rng

    origin, month = entry["origin"], entry["month"]
    if not SPLITS["train"][0] <= month <= SPLITS["train"][1]:
        raise ValueError("Replay target is outside the training split")
    start, _, starts, weights, _, _ = window_layout(origin, month)
    if entry["bin_index"] != start:
        raise ValueError("Cached latent timestamp does not match target window")
    later = intervals.load(data, starts)
    sample = corrupt_sample(data.load(data.root / "train" / (month + ".npz")), seed)
    states = continue_members(
        model,
        entry["latent"].to(data.device),
        model.adapt(later["atmosphere"]),
        later["contexts"],
        rng(data.device, seed + 2000000),
    )
    if capture is not None:
        capture(states)
    predictions = torch.stack(
        [model.decode_members(state, data.mask)[:, :, -1] for state in states[1:]], 2
    )
    target = {
        **sample,
        "raw_surface": later["raw_surface"],
        "month_weights": data.tensor(weights),
    }
    forecast = forecast_crps(data, predictions, target)
    bound = excess_latent_loss(states, states[0])
    yield (
        "forecast_replay",
        forecast + 0.1 * bound,
        {
            "forecast": float(forecast.detach()),
            "bound": float(bound.detach()),
        },
    )
    del predictions, forecast, bound, states, target, later

    adapted = model.adapt(sample["atmosphere"][:, :19])
    latent, known, anchor = model.encode_native(
        sample["surface"][:, :19],
        adapted,
        sample["contexts"][:, 18],
        data.mask,
        sample["validity"][:, :19],
        task="observation",
    )
    members = model.sample_initial_members(latent, rng(data.device, seed + 2000000), 2)
    initial = model.decode_members(members, data.mask, known, anchor)
    completion = completion_crps(
        data,
        initial[:, :, :, model.surface],
        sample["completion_target"],
        sample["completion_valid"],
    )
    yield "completion", 0.1 * completion, {"completion": float(completion.detach())}
    del initial, completion, members, latent, adapted

    monthly, bound = coherent_reconstruction(
        model, data, sample, rng(data.device, seed + 3000000)
    )
    reconstruction = interior_crps(data, monthly, sample)
    yield (
        "reconstruction",
        0.1 * reconstruction + 0.1 * bound,
        {
            "reconstruction": float(reconstruction.detach()),
        },
    )
