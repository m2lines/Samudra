# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Detached physical-state replay, independently sampled for each data source."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ReplayEntry:
    source: str
    origin: str | int
    month: int
    start_lead: int
    origin_update: int
    states: torch.Tensor


def states_at(initial: torch.Tensor, predictions: torch.Tensor, steps: int):
    """Two states after ``steps`` calls, including a recomputable overlap bin.

    A 31-day month requires seven five-day predictions. The next month's first
    bin starts at day 30, so its cache is taken after six calls, not seven.
    """
    if initial.ndim != 5 or initial.shape[1] != 2:
        raise ValueError("Replay needs exactly two physical states")
    if not 0 <= steps <= predictions.shape[1]:
        raise ValueError("Replay boundary is outside the computed trajectory")
    if steps == 0:
        return initial
    if steps == 1:
        return torch.stack((initial[:, -1], predictions[:, 0]), dim=1)
    return predictions[:, steps - 2 : steps]


class ReplayPool:
    """Uniform occupied-month sampling, bounded by original trajectory age.

    Two entries per lead month keep full-resolution physical-state storage
    bounded (~0.9 GiB per source at 1 degree / float32). Entries stay on CPU in
    their original dtype. Replaying never refreshes the original model age.
    """

    def __init__(self, source: str, seed: int, *, per_month=2, max_age=250):
        if per_month < 1 or max_age < 0:
            raise ValueError("Invalid replay capacity or maximum age")
        self.source, self.per_month, self.max_age = source, per_month, max_age
        self.rng = np.random.default_rng(seed)
        self.entries: dict[int, list[ReplayEntry]] = {}

    def add(self, entry: ReplayEntry):
        if entry.source != self.source:
            raise ValueError("Cannot mix sources in a replay pool")
        if not 1 <= entry.month <= 11 or not 0 < entry.start_lead < 73:
            raise ValueError("Replay continuation is outside the annual horizon")
        if entry.states.ndim != 5 or entry.states.shape[1] != 2:
            raise ValueError("Replay requires two physical states")
        if not torch.isfinite(entry.states).all():
            raise FloatingPointError("Cannot cache nonfinite physical states")
        saved = ReplayEntry(
            entry.source,
            entry.origin,
            entry.month,
            entry.start_lead,
            entry.origin_update,
            entry.states.detach().cpu().clone(),
        )
        bucket = self.entries.setdefault(entry.month, [])
        bucket.append(saved)
        del bucket[: -self.per_month]

    def draw(self, update: int) -> ReplayEntry | None:
        self.entries = {
            month: [e for e in entries if 0 <= update - e.origin_update <= self.max_age]
            for month, entries in self.entries.items()
        }
        months = sorted(month for month, entries in self.entries.items() if entries)
        if not months:
            return None
        month = months[int(self.rng.integers(len(months)))]
        bucket = self.entries[month]
        return bucket[int(self.rng.integers(len(bucket)))]

    def state_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "per_month": self.per_month,
            "max_age": self.max_age,
            "rng": deepcopy(self.rng.bit_generator.state),
            "entries": deepcopy(self.entries),
        }

    def load_state_dict(self, state):
        for key in ("source", "per_month", "max_age"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Replay contract changed: {key}")
        self.entries = {}
        for entries in state["entries"].values():
            for entry in entries:
                self.add(entry)
        self.rng.bit_generator.state = deepcopy(state["rng"])
