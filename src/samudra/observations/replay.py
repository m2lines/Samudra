# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Detached physical-state replay, independently sampled for each data source."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import torch

from samudra.observations.constants import (
    FORECAST_STEPS,
    MAX_REPLAY_MONTHS,
    STATE_HISTORY_STEPS,
)


@dataclass
class ReplayEntry:
    source: str
    origin: str | int
    month: int
    start_lead: int
    origin_update: int
    states: torch.Tensor  # [batch, state_history, physical_channel, y, x]


class ReplayPoolState(TypedDict):
    """Checkpoint payload; NumPy defines the bit-generator-specific RNG mapping."""

    source: str
    per_month: int
    max_age: int
    rng: Mapping[str, Any]
    entries: dict[int, list[ReplayEntry]]


def states_at(
    initial: torch.Tensor, predictions: torch.Tensor, steps: int
) -> torch.Tensor:
    """Two states after ``steps`` calls, including a recomputable overlap bin.

    Inputs have shapes [batch, state_history, channel, y, x] and
    [batch, forecast_step, channel, y, x]; the result has the initial shape.
    A 31-day month requires seven five-day predictions. The next month's first
    bin starts at day 30, so its cache is taken after six calls, not seven.
    """
    if initial.ndim != 5 or initial.shape[1] != STATE_HISTORY_STEPS:
        raise ValueError("Replay needs exactly two physical states")
    if not 0 <= steps <= predictions.shape[1]:
        raise ValueError("Replay boundary is outside the computed trajectory")
    if steps == 0:
        return initial
    if steps == 1:
        return torch.stack((initial[:, -1], predictions[:, 0]), dim=1)
    return predictions[:, steps - STATE_HISTORY_STEPS : steps]


class ReplayPool:
    """Uniform occupied-month sampling, bounded by original trajectory age.

    Capacity is bounded per lead month. Entries stay on CPU in their original
    dtype. Age counts optimizer updates since the original initialization.
    """

    def __init__(
        self, source: str, seed: int, *, per_month: int = 2, max_age: int = 250
    ) -> None:
        if per_month < 1 or max_age < 0:
            raise ValueError("Invalid replay capacity or maximum age")
        self.source, self.per_month, self.max_age = source, per_month, max_age
        self.rng = np.random.default_rng(seed)
        self.entries: dict[int, list[ReplayEntry]] = {}

    def add(self, entry: ReplayEntry) -> None:
        if entry.source != self.source:
            raise ValueError("Cannot mix sources in a replay pool")
        if (
            not 1 <= entry.month < MAX_REPLAY_MONTHS
            or not 0 < entry.start_lead < FORECAST_STEPS
        ):
            raise ValueError("Replay continuation is outside the annual horizon")
        if entry.states.ndim != 5 or entry.states.shape[1] != STATE_HISTORY_STEPS:
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

    def state_dict(self) -> ReplayPoolState:
        return {
            "source": self.source,
            "per_month": self.per_month,
            "max_age": self.max_age,
            "rng": deepcopy(self.rng.bit_generator.state),
            "entries": deepcopy(self.entries),
        }

    def load_state_dict(self, state: ReplayPoolState) -> None:
        if (state["source"], state["per_month"], state["max_age"]) != (
            self.source,
            self.per_month,
            self.max_age,
        ):
            raise ValueError("Replay contract changed: source, capacity or maximum age")
        self.entries = {}
        for entries in state["entries"].values():
            for entry in entries:
                self.add(entry)
        self.rng.bit_generator.state = deepcopy(state["rng"])
